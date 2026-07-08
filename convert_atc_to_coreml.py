#!/usr/bin/env python3
"""
Convert MLX fine-tuned ATC Whisper model → Core ML encoder (.mlmodelc)
for Apple Neural Engine acceleration in the iOS app.

Size-agnostic: encoder dims are read from the MLX config.json, so large-v2 /
medium / small all work. Naming rule (per whisper.cpp's
whisper_get_coreml_path_encoder): take the GGML .bin name, drop ".bin", strip a
trailing quant suffix like "-q5_1"/"-q8_0", then append "-encoder.mlmodelc".
So flightai-asr-v1-q5_1.bin → flightai-asr-v1-encoder.mlmodelc. The encoder is
architecture-only (independent of GGML quantization), so the same encoder
pairs with any quant of a given model.

Usage:
    /opt/miniconda3/bin/python3 convert_atc_to_coreml.py \
        [--mlx-dir DIR] [--out-mlmodelc PATH]
"""

import argparse
import json
import sys
import os
import subprocess
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import coremltools as ct
from pathlib import Path
from torch import nn, Tensor
from typing import Optional, Dict

DEFAULT_MLX_DIR      = "/Users/alexhan/Projects/atc-python/models/flightai-asr-v1-mlx"
# NOTE: whisper.cpp strips a trailing quant suffix ("-q5_1", "-q8_0", …) from
# the .bin name before appending "-encoder.mlmodelc", so the encoder for
# flightai-asr-v1-q5_1.bin must be named flightai-asr-v1-encoder.mlmodelc.
DEFAULT_OUT_MLMODELC = "/Users/alexhan/Projects/atc-ios/whisper.swiftui.demo/Resources/models/flightai-asr-v1-encoder.mlmodelc"

_ap = argparse.ArgumentParser(description="Convert an MLX Whisper encoder (any size) to a Core ML .mlmodelc for ANE.")
_ap.add_argument("--mlx-dir", default=DEFAULT_MLX_DIR,
                 help="MLX model dir containing weights.safetensors + config.json")
_ap.add_argument("--out-mlmodelc", default=DEFAULT_OUT_MLMODELC,
                 help="output .mlmodelc path; basename must match the shipped GGML .bin")
_ap.add_argument("--palettize", type=int, default=0, metavar="BITS",
                 help="palettize (k-means LUT) weights to N bits (e.g. 6). Shrinks the "
                      "encoder so it fits the iPhone ANE program limit — an uncompressed "
                      "F16 large encoder (~1.2GB) fails with ANE error 0x20004. "
                      "0 = off (plain F16).")
_args = _ap.parse_args()

MLX_DIR      = Path(_args.mlx_dir)
OUT_MLMODELC = Path(_args.out_mlmodelc)
# Intermediate .mlpackage in /tmp, named to match the final .mlmodelc stem so
# coremlc's "<stem>.mlmodelc" output lines up with OUT_MLMODELC.
OUT_MLPKG    = Path("/tmp") / (OUT_MLMODELC.stem + ".mlpackage")

# ── ANE-optimised Whisper classes (from whisper.cpp convert-whisper-to-coreml.py) ──

from coremltools.models.neural_network.quantization_utils import quantize_weights
from whisper.model import (
    Whisper, AudioEncoder, TextDecoder,
    ResidualAttentionBlock, MultiHeadAttention, ModelDimensions
)
import whisper.model
whisper.model.MultiHeadAttention.use_sdpa = False

try:
    from ane_transformers.reference.layer_norm import LayerNormANE as LayerNormANEBase
    HAS_ANE = True
except ImportError:
    HAS_ANE = False
    print("WARNING: ane_transformers not found, falling back to standard encoder (no ANE path)")


def linear_to_conv2d_map(state_dict, prefix, local_metadata, strict,
                         missing_keys, unexpected_keys, error_msgs):
    for k in state_dict:
        is_attention = all(s in k for s in ['attn', '.weight'])
        is_mlp = any(k.endswith(s) for s in ['mlp.0.weight', 'mlp.2.weight'])
        if (is_attention or is_mlp) and len(state_dict[k].shape) == 2:
            state_dict[k] = state_dict[k][:, :, None, None]


def correct_for_bias_scale_order_inversion(state_dict, prefix, local_metadata,
                                            strict, missing_keys,
                                            unexpected_keys, error_msgs):
    state_dict[prefix + 'bias'] = state_dict[prefix + 'bias'] / state_dict[prefix + 'weight']
    return state_dict


if HAS_ANE:
    class LayerNormANE(LayerNormANEBase):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._register_load_state_dict_pre_hook(correct_for_bias_scale_order_inversion)

    class MultiHeadAttentionANE(MultiHeadAttention):
        def __init__(self, n_state: int, n_head: int):
            super().__init__(n_state, n_head)
            self.query = nn.Conv2d(n_state, n_state, kernel_size=1)
            self.key   = nn.Conv2d(n_state, n_state, kernel_size=1, bias=False)
            self.value = nn.Conv2d(n_state, n_state, kernel_size=1)
            self.out   = nn.Conv2d(n_state, n_state, kernel_size=1)

        def forward(self, x, xa=None, mask=None, kv_cache=None):
            q = self.query(x)
            if kv_cache is None or xa is None or self.key not in kv_cache:
                k = self.key(x if xa is None else xa)
                v = self.value(x if xa is None else xa)
            else:
                k = kv_cache[self.key]
                v = kv_cache[self.value]
            wv, qk = self.qkv_attention_ane(q, k, v, mask)
            return self.out(wv), qk

        def qkv_attention_ane(self, q, k, v, mask=None):
            _, dim, _, seqlen = q.size()
            dim_per_head = dim // self.n_head
            scale = float(dim_per_head) ** -0.5
            q = q * scale
            mh_q = q.split(dim_per_head, dim=1)
            mh_k = k.transpose(1, 3).split(dim_per_head, dim=3)
            mh_v = v.split(dim_per_head, dim=1)
            mh_qk = [torch.einsum('bchq,bkhc->bkhq', qi, ki) for qi, ki in zip(mh_q, mh_k)]
            if mask is not None:
                for i in range(self.n_head):
                    mh_qk[i] = mh_qk[i] + mask[:, :seqlen, :, :seqlen]
            attn_w = [aw.softmax(dim=1) for aw in mh_qk]
            attn = [torch.einsum('bkhq,bchk->bchq', wi, vi) for wi, vi in zip(attn_w, mh_v)]
            return torch.cat(attn, dim=1), torch.cat(mh_qk, dim=1).float().detach()

    class ResidualAttentionBlockANE(ResidualAttentionBlock):
        def __init__(self, n_state, n_head, cross_attention=False):
            super().__init__(n_state, n_head, cross_attention)
            self.attn        = MultiHeadAttentionANE(n_state, n_head)
            self.attn_ln     = LayerNormANE(n_state)
            self.cross_attn    = MultiHeadAttentionANE(n_state, n_head) if cross_attention else None
            self.cross_attn_ln = LayerNormANE(n_state) if cross_attention else None
            n_mlp = n_state * 4
            self.mlp    = nn.Sequential(nn.Conv2d(n_state, n_mlp, 1), nn.GELU(), nn.Conv2d(n_mlp, n_state, 1))
            self.mlp_ln = LayerNormANE(n_state)

    class AudioEncoderANE(AudioEncoder):
        def __init__(self, n_mels, n_ctx, n_state, n_head, n_layer):
            super().__init__(n_mels, n_ctx, n_state, n_head, n_layer)
            self.blocks  = nn.ModuleList([ResidualAttentionBlockANE(n_state, n_head) for _ in range(n_layer)])
            self.ln_post = LayerNormANE(n_state)

        def forward(self, x: Tensor):
            x = F.gelu(self.conv1(x))
            x = F.gelu(self.conv2(x))
            assert x.shape[1:] == self.positional_embedding.shape[::-1], "incorrect audio shape"
            x = (x + self.positional_embedding.transpose(0, 1)).to(x.dtype).unsqueeze(2)
            for block in self.blocks:
                x = block(x)
            x = self.ln_post(x)
            return x.squeeze(2).transpose(1, 2)

    class WhisperANE(Whisper):
        def __init__(self, dims: ModelDimensions):
            super().__init__(dims)
            self.encoder = AudioEncoderANE(dims.n_mels, dims.n_audio_ctx, dims.n_audio_state, dims.n_audio_head, dims.n_audio_layer)
            self._register_load_state_dict_pre_hook(linear_to_conv2d_map)


def convert_encoder(hparams, encoder, quantize=False):
    encoder.eval()
    input_shape = (1, hparams.n_mels, 3000)
    traced = torch.jit.trace(encoder, torch.randn(input_shape))
    # compute_precision=FLOAT16 stores and runs in F16 (works on ANE); replaces
    # the legacy quantize_weights(nbits=16) call which doesn't support mlprogram.
    precision = ct.precision.FLOAT16 if quantize else ct.precision.FLOAT32
    model = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="logmel_data", shape=input_shape)],
        outputs=[ct.TensorType(name="output")],
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=precision,
    )
    return model


# ── 1. Load MLX safetensors ──────────────────────────────────────────────────

from safetensors import safe_open

print("Loading MLX safetensors …")
tensors = {}
with safe_open(MLX_DIR / "weights.safetensors", framework="np") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)

if "encoder._positional_embedding" in tensors:
    tensors["encoder.positional_embedding"] = tensors.pop("encoder._positional_embedding")

tensors = {
    k.replace(".mlp1.", ".mlp.0.").replace(".mlp2.", ".mlp.2."): v
    for k, v in tensors.items()
}
print(f"Loaded {len(tensors)} tensors")

# ── 2. Build ModelDimensions ─────────────────────────────────────────────────

mlx_cfg = json.loads((MLX_DIR / "config.json").read_text())
n_vocab = tensors["decoder.token_embedding.weight"].shape[0]

dims = ModelDimensions(
    n_mels        = mlx_cfg["n_mels"],
    n_audio_ctx   = mlx_cfg["n_audio_ctx"],
    n_audio_state = mlx_cfg["n_audio_state"],
    n_audio_head  = mlx_cfg["n_audio_head"],
    n_audio_layer = mlx_cfg["n_audio_layer"],
    n_vocab       = n_vocab,
    n_text_ctx    = mlx_cfg["n_text_ctx"],
    n_text_state  = mlx_cfg.get("n_text_state", mlx_cfg["n_audio_state"]),
    n_text_head   = mlx_cfg.get("n_text_head",  mlx_cfg["n_audio_head"]),
    n_text_layer  = mlx_cfg.get("n_text_layer", mlx_cfg["n_audio_layer"]),
)
print("ModelDimensions:", dims)

# ── 3. Load weights into model ───────────────────────────────────────────────

print("Building model and loading weights …")

state_dict = {}
for k, v in tensors.items():
    t = torch.from_numpy(v.astype(np.float32))
    # MLX conv weights: [out, kernel, in] → PyTorch Conv1d: [out, in, kernel]
    if "conv" in k and "weight" in k and t.ndim == 3:
        t = t.permute(0, 2, 1)
    state_dict[k] = t

# Build ANE encoder and load only encoder weights (avoids decoder Linear vs Conv2d conflict)
if HAS_ANE:
    encoder_model = AudioEncoderANE(
        dims.n_mels, dims.n_audio_ctx, dims.n_audio_state, dims.n_audio_head, dims.n_audio_layer
    ).cpu().eval()
    encoder_model._register_load_state_dict_pre_hook(linear_to_conv2d_map)
else:
    encoder_model = AudioEncoder(
        dims.n_mels, dims.n_audio_ctx, dims.n_audio_state, dims.n_audio_head, dims.n_audio_layer
    ).cpu().eval()

# Strip "encoder." prefix
encoder_sd = {k[len("encoder."):]: v for k, v in state_dict.items() if k.startswith("encoder.")}
missing, unexpected = encoder_model.load_state_dict(encoder_sd, strict=True)
if missing:    print(f"  Missing keys ({len(missing)}): {missing[:3]} …")
if unexpected: print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:3]} …")

# ── 4. Convert encoder ───────────────────────────────────────────────────────

print("\nConverting encoder to Core ML (may take several minutes) …")
coreml_model = convert_encoder(dims, encoder_model, quantize=True)

if _args.palettize > 0:
    import coremltools.optimize as cto
    print(f"\nPalettizing weights to {_args.palettize}-bit (k-means LUT) — this can take a while …")
    opt_config = cto.coreml.OptimizationConfig(
        global_config=cto.coreml.OpPalettizerConfig(mode="kmeans", nbits=_args.palettize)
    )
    coreml_model = cto.coreml.palettize_weights(coreml_model, opt_config)

print(f"Saving .mlpackage → {OUT_MLPKG}")
if OUT_MLPKG.exists():
    shutil.rmtree(OUT_MLPKG)
coreml_model.save(str(OUT_MLPKG))

# ── 5. Compile to .mlmodelc ──────────────────────────────────────────────────

print(f"\nCompiling → {OUT_MLMODELC}")
OUT_MLMODELC.parent.mkdir(parents=True, exist_ok=True)

result = subprocess.run(
    ["xcrun", "coremlc", "compile", str(OUT_MLPKG), str(OUT_MLMODELC.parent)],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)
    sys.exit(1)

# Compiler outputs <stem>.mlmodelc inside the parent dir
compiled = OUT_MLMODELC.parent / (OUT_MLPKG.stem + ".mlmodelc")
if compiled.resolve() != OUT_MLMODELC.resolve():
    if OUT_MLMODELC.exists():
        shutil.rmtree(OUT_MLMODELC)
    compiled.rename(OUT_MLMODELC)

print(f"\nDone! → {OUT_MLMODELC}")
print("Rebuild Xcode to pick up the new Core ML encoder.")
