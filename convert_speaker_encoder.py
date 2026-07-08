#!/usr/bin/env python3
"""
Convert resemblyzer's VoiceEncoder (GE2E speaker embedding) to Core ML for
on-device speaker labeling (ATC vs PLT), mirroring atc-python's SpeakerTracker.

The mel-spectrogram front-end (librosa: n_fft=400, hop=160, n_mels=40,
power=2) is baked INTO the model as fixed conv kernels (DFT-as-Conv1d), so
Swift feeds raw 16 kHz waveform — no DSP code on the app side.

Input : [1, 25840] float32 waveform (1.615 s → exactly 160 mel frames)
Output: [1, 256]   L2-normalized speaker embedding

Usage:
    /opt/miniconda3/bin/python3 convert_speaker_encoder.py

Output:
    whisper.swiftui.demo/Resources/models/speaker-id-v1.mlmodelc
"""

import numpy as np
import torch
import torch.nn as nn
import librosa
import coremltools as ct
import subprocess, shutil, sys
from pathlib import Path
from resemblyzer import VoiceEncoder

OUT_MLMODELC = Path("/Users/alexhan/Projects/atc-ios/whisper.swiftui.demo/Resources/models/speaker-id-v1.mlmodelc")
OUT_MLPKG    = Path("/tmp/speaker-id-v1.mlpackage")

N_FFT, HOP, N_MELS, SR = 400, 160, 40, 16000
WIN_SAMPLES = 25840          # (L - N_FFT)/HOP + 1 = 160 frames, no padding

# ── wrap GE2E with a conv-DFT mel front-end ──────────────────────────────────

class SpeakerNet(nn.Module):
    def __init__(self, ve: VoiceEncoder):
        super().__init__()
        hann = np.hanning(N_FFT + 1)[:-1]           # librosa periodic hann
        k = np.arange(N_FFT // 2 + 1)[:, None]      # [201,1]
        n = np.arange(N_FFT)[None, :]               # [1,400]
        cos_k = (hann * np.cos(2 * np.pi * k * n / N_FFT)).astype(np.float32)
        sin_k = (-(hann * np.sin(2 * np.pi * k * n / N_FFT))).astype(np.float32)
        self.conv_re = nn.Conv1d(1, N_FFT // 2 + 1, N_FFT, stride=HOP, bias=False)
        self.conv_im = nn.Conv1d(1, N_FFT // 2 + 1, N_FFT, stride=HOP, bias=False)
        self.conv_re.weight.data = torch.from_numpy(cos_k).unsqueeze(1)
        self.conv_im.weight.data = torch.from_numpy(sin_k).unsqueeze(1)

        mel_fb = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS).astype(np.float32)
        self.mel = nn.Linear(N_FFT // 2 + 1, N_MELS, bias=False)
        self.mel.weight.data = torch.from_numpy(mel_fb)

        self.lstm   = ve.lstm      # 3-layer LSTM(40→256), batch_first
        self.linear = ve.linear    # 256→256
        self.relu   = ve.relu

    def forward(self, wav):                       # [1, WIN_SAMPLES]
        x = wav.unsqueeze(1)                      # [1,1,T]
        power = self.conv_re(x) ** 2 + self.conv_im(x) ** 2   # [1,201,160]
        mels = self.mel(power.transpose(1, 2))    # [1,160,40]
        _, (hidden, _) = self.lstm(mels)
        e = self.relu(self.linear(hidden[-1]))    # [1,256]
        return e / (e.norm(dim=1, keepdim=True) + 1e-5)

print("Loading resemblyzer VoiceEncoder …")
ve = VoiceEncoder("cpu")
net = SpeakerNet(ve).eval()

# ── verify against resemblyzer on real-ish audio ─────────────────────────────

rng = np.random.default_rng(0)
wav = (rng.standard_normal(WIN_SAMPLES) * 0.1).astype(np.float32)
with torch.no_grad():
    ours = net(torch.from_numpy(wav).unsqueeze(0)).numpy()[0]
mel_ref = librosa.feature.melspectrogram(
    y=wav, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, center=False
).astype(np.float32).T                            # [160,40]
with torch.no_grad():
    ref = ve.forward(torch.from_numpy(mel_ref[None])).numpy()[0]
cos = float(np.dot(ours, ref) / (np.linalg.norm(ours) * np.linalg.norm(ref)))
print(f"parity vs resemblyzer: cosine = {cos:.6f}")
assert cos > 0.999, "front-end mismatch"

# ── convert ──────────────────────────────────────────────────────────────────

print("Converting to Core ML …")
traced = torch.jit.trace(net, torch.from_numpy(wav).unsqueeze(0))
mlmodel = ct.convert(
    traced,
    convert_to="mlprogram",
    inputs=[ct.TensorType(name="waveform", shape=(1, WIN_SAMPLES))],
    outputs=[ct.TensorType(name="embedding")],
    compute_units=ct.ComputeUnit.ALL,
    compute_precision=ct.precision.FLOAT16,
)
if OUT_MLPKG.exists():
    shutil.rmtree(OUT_MLPKG)
mlmodel.save(str(OUT_MLPKG))

print(f"Compiling → {OUT_MLMODELC}")
r = subprocess.run(["xcrun", "coremlc", "compile", str(OUT_MLPKG), str(OUT_MLMODELC.parent)],
                   capture_output=True, text=True)
if r.returncode != 0:
    sys.exit(r.stderr)
compiled = OUT_MLMODELC.parent / (OUT_MLPKG.stem + ".mlmodelc")
if compiled.resolve() != OUT_MLMODELC.resolve():
    if OUT_MLMODELC.exists():
        shutil.rmtree(OUT_MLMODELC)
    compiled.rename(OUT_MLMODELC)
print("Done!")
