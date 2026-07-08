#!/usr/bin/env python3
"""
Convert MLX whisper safetensors → GGML F16 directly in Python.

Source: an MLX Whisper model fine-tuned on ATCO2 + ATCOSIM (large-v2 today,
but any size works — dims are read from config.json). Writes an F16 GGML
model (~3 GB for large-v2); quantize it afterwards with whisper-quantize:
    whisper-quantize flightai-asr-v1-f16.bin flightai-asr-v1-q5_1.bin q5_1

For the full one-command pipeline (MLX → f16 → quantize → ANE encoder),
use build_ios_model.sh, which chains this + whisper-quantize +
convert_atc_to_coreml.py with matching output names.

Usage:
    /opt/miniconda3/bin/python3 convert_mlx_to_ggml.py \
        [--mlx-dir DIR] [--out PATH] [--whisper-assets DIR]
"""

import argparse
import json
import struct
import sys
import numpy as np
from pathlib import Path

# Defaults preserve the original behaviour; override per model size via CLI.
DEFAULT_MLX_DIR        = "/Users/alexhan/Projects/atc-python/models/flightai-asr-v1-mlx"
DEFAULT_OUT_F16        = "/Users/alexhan/Projects/atc-ios/whisper.swiftui.demo/Resources/models/flightai-asr-v1-f16.bin"
DEFAULT_WHISPER_ASSETS = "/opt/miniconda3/lib/python3.13/site-packages/whisper/assets"

_ap = argparse.ArgumentParser(description="Convert an MLX Whisper model (any size) to a GGML F16 model.")
_ap.add_argument("--mlx-dir", default=DEFAULT_MLX_DIR,
                 help="MLX model dir containing weights.safetensors + config.json")
_ap.add_argument("--out", default=DEFAULT_OUT_F16, help="output GGML F16 .bin path")
_ap.add_argument("--whisper-assets", default=DEFAULT_WHISPER_ASSETS,
                 help="dir with mel_filters.npz + *.tiktoken")
_args = _ap.parse_args()

MLX_DIR        = Path(_args.mlx_dir)
OUT_F16        = Path(_args.out)
WHISPER_ASSETS = Path(_args.whisper_assets)
# All architecture dims (n_mels, n_layer, n_state, …) are read from config.json
# below, so this script is size-agnostic: large-v2 / medium / small all work.

# GGML type IDs (per-tensor ftype in header)
GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1

# ── 1. Load safetensors ──────────────────────────────────────────────────────

from safetensors import safe_open

print("Loading MLX safetensors …")
tensors = {}
with safe_open(MLX_DIR / "weights.safetensors", framework="np") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)

# MLX positional embedding has a leading underscore
if "encoder._positional_embedding" in tensors:
    tensors["encoder.positional_embedding"] = tensors.pop("encoder._positional_embedding")

# MLX uses mlp1/mlp2; whisper.cpp expects mlp.0/mlp.2
tensors = {
    k.replace(".mlp1.", ".mlp.0.").replace(".mlp2.", ".mlp.2."): v
    for k, v in tensors.items()
}

# MLX Conv1d weights are [out, kernel, in]; GGML/PyTorch needs [out, in, kernel]
# (stored reversed in GGML → ne = [kernel, in, out] which matches whisper.cpp)
for key in list(tensors.keys()):
    if (key in ("encoder.conv1.weight", "encoder.conv2.weight")
            and tensors[key].ndim == 3):
        tensors[key] = tensors[key].transpose(0, 2, 1).copy()
        print(f"  transposed {key}: {tensors[key].shape}")

print(f"Loaded {len(tensors)} tensors")

# ── 2. Config ────────────────────────────────────────────────────────────────

mlx_cfg = json.loads((MLX_DIR / "config.json").read_text())
hparams = {
    "n_vocab":       tensors["decoder.token_embedding.weight"].shape[0],
    "n_audio_ctx":   mlx_cfg["n_audio_ctx"],
    "n_audio_state": mlx_cfg["n_audio_state"],
    "n_audio_head":  mlx_cfg["n_audio_head"],
    "n_audio_layer": mlx_cfg["n_audio_layer"],
    "n_text_ctx":    mlx_cfg["n_text_ctx"],
    "n_text_state":  mlx_cfg.get("n_text_state", mlx_cfg["n_audio_state"]),
    "n_text_head":   mlx_cfg.get("n_text_head",  mlx_cfg["n_audio_head"]),
    "n_text_layer":  mlx_cfg.get("n_text_layer", mlx_cfg["n_audio_layer"]),
    "n_mels":        mlx_cfg["n_mels"],
}
print("hparams:", hparams)

# ── 3. Mel filters ───────────────────────────────────────────────────────────

mel_npz = WHISPER_ASSETS / "mel_filters.npz"
if not mel_npz.exists():
    sys.exit(f"mel_filters.npz not found at {mel_npz}")

with np.load(mel_npz) as f:
    filters = f[f"mel_{hparams['n_mels']}"].astype(np.float32)
print(f"Mel filters: {filters.shape}")

# ── 4. Tokenizer ─────────────────────────────────────────────────────────────

import base64

multilingual = hparams["n_vocab"] >= 51865
tiktoken_file = WHISPER_ASSETS / ("multilingual.tiktoken" if multilingual else "gpt2.tiktoken")
if not tiktoken_file.exists():
    sys.exit(f"Tiktoken file not found: {tiktoken_file}")

tokens: list[bytes] = []
with open(tiktoken_file, "rb") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        token_b64, _rank = line.split()
        tokens.append(base64.b64decode(token_b64))

special_tokens = ["<|endoftext|>", "<|startoftranscript|>"]
if multilingual:
    langs = [
        "en","zh","de","es","ru","ko","fr","ja","pt","tr","pl","ca","nl","ar",
        "sv","it","id","hi","fi","vi","iw","uk","el","ms","cs","ro","da","hu",
        "ta","no","th","ur","hr","bg","lt","la","mi","ml","cy","sk","te","fa",
        "lv","bn","sr","az","sl","kn","et","mk","br","eu","is","hy","ne","mn",
        "bs","kk","sq","sw","gl","mr","pa","si","km","sn","yo","so","af","oc",
        "ka","be","tg","sd","gu","am","yi","lo","uz","fo","ht","ps","tk","nn",
        "mt","sa","lb","my","bo","tl","mg","as","tt","haw","ln","ha","ba","jw","su",
    ]
    for lang in langs:
        special_tokens.append(f"<|{lang}|>")
special_tokens += [
    "<|translate|>", "<|transcribe|>", "<|startoflm|>", "<|startofprev|>",
    "<|nospeech|>", "<|notimestamps|>",
]
for i in range(1501):
    special_tokens.append(f"<|{i * 0.02:.2f}|>")
for st in special_tokens:
    tokens.append(st.encode("utf-8"))

tokens = tokens[:hparams["n_vocab"]]
while len(tokens) < hparams["n_vocab"]:
    tokens.append(b"")
print(f"Vocab: {len(tokens)} tokens")

# ── 5. Write GGML F16 ────────────────────────────────────────────────────────

print(f"\nWriting GGML F16 → {OUT_F16} …")
OUT_F16.parent.mkdir(parents=True, exist_ok=True)

def write_aligned(fout, data: bytes):
    """Write data, padding before it to reach 32-byte alignment."""
    cur = fout.tell()
    pad = (32 - cur % 32) % 32
    if pad:
        fout.write(b'\x00' * pad)
    fout.write(data)

with open(OUT_F16, "wb") as fout:
    # magic + hparams
    fout.write(struct.pack("i", 0x67676d6c))
    for key in ["n_vocab","n_audio_ctx","n_audio_state","n_audio_head","n_audio_layer",
                "n_text_ctx","n_text_state","n_text_head","n_text_layer","n_mels"]:
        fout.write(struct.pack("i", hparams[key]))
    fout.write(struct.pack("i", 1))  # GGML_FTYPE_MOSTLY_F16

    # mel filters
    fout.write(struct.pack("i", filters.shape[0]))
    fout.write(struct.pack("i", filters.shape[1]))
    fout.write(filters.tobytes())

    # vocab
    fout.write(struct.pack("i", len(tokens)))
    for tok in tokens:
        fout.write(struct.pack("i", len(tok)))
        fout.write(tok)

    # weights
    for name, data in tensors.items():
        data = data.squeeze()

        # whisper.cpp creates conv biases via ggml_new_tensor_2d(F32, 1, n_audio_state),
        # giving ne = [1, 1280, 1, 1]. Write as 2D (1280, 1) so file ne = [1, 1280].
        if name in ("encoder.conv1.bias", "encoder.conv2.bias") and data.ndim == 1:
            data = data.reshape(-1, 1)

        # 1D tensors, conv biases, positional embeddings → keep as f32
        keep_f32 = (
            data.ndim < 2 or
            name in ("encoder.conv1.bias", "encoder.conv2.bias",
                     "encoder.positional_embedding", "decoder.positional_embedding")
        )
        # whisper.cpp allocates conv weights as vtype (F16 in Q8 mode), not wtype
        keep_f16 = name in ("encoder.conv1.weight", "encoder.conv2.weight")

        if keep_f32:
            arr = data.astype(np.float32)
            ftype = GGML_TYPE_F32
            tensor_bytes = arr.tobytes()
        else:
            arr = data.astype(np.float16)
            ftype = GGML_TYPE_F16
            tensor_bytes = arr.tobytes()

        n_dims = data.ndim
        name_enc = name.encode("utf-8")
        fout.write(struct.pack("iii", n_dims, len(name_enc), ftype))
        for i in range(n_dims):
            fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
        fout.write(name_enc)
        fout.write(tensor_bytes)

        label = "f32" if ftype == GGML_TYPE_F32 else "f16"
        print(f"  {name:60s} {str(data.shape):20s} {label}")

size_gb = OUT_F16.stat().st_size / 1e9
print(f"\nDone! F16 model → {OUT_F16} ({size_gb:.2f} GB)")
print("Next: quantize to Q5_1 with whisper.cpp, then rebuild Xcode.")
