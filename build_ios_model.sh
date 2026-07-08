#!/usr/bin/env bash
#
# One-command, size-agnostic pipeline: MLX Whisper model → iOS-ready assets.
#
#   MLX safetensors  ──convert_mlx_to_ggml.py──►  GGML F16 (intermediate)
#                    ──whisper-quantize────────►  GGML <quant>  (decoder, shipped)
#                    ──convert_atc_to_coreml.py►  ANE encoder .mlmodelc (shipped)
#
# Works for any model size (large-v2 / medium / small): all architecture dims
# are read from the MLX config.json. The GGML .bin and the .mlmodelc are named
# with the SAME basename so whisper.cpp auto-pairs them on device
# (<basename>.bin  ↔  <basename>-encoder.mlmodelc).
#
# Usage:
#   ./build_ios_model.sh [MLX_DIR] [BASENAME] [QUANT]
#
#   MLX_DIR   MLX model dir (weights.safetensors + config.json)
#             default: /Users/alexhan/Projects/atc-python/models/flightai-asr-v1-mlx
#   BASENAME  shipped file stem (default: flightai-asr-v1-q5_1)
#             Keep the default to swap models with ZERO Xcode/app changes —
#             WhisperState auto-loads this name from the bundle. Use a distinct
#             stem only for in-app A/B (then update WhisperState/ContentView).
#   QUANT     whisper-quantize type (default: q5_1; e.g. q8_0, q5_0, q4_1)
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLX_DIR="${1:-/Users/alexhan/Projects/atc-python/models/flightai-asr-v1-mlx}"
BASENAME="${2:-flightai-asr-v1-q5_1}"
QUANT="${3:-q5_1}"

PY="/opt/miniconda3/bin/python3"
[ -x "$PY" ] || PY="$(command -v python3)"

MODELS_DIR="$REPO_DIR/whisper.swiftui.demo/Resources/models"
QUANTIZE_BIN="$REPO_DIR/whisper.cpp/build/bin/whisper-quantize"

F16="/tmp/${BASENAME}.f16.bin"
FINAL_BIN="$MODELS_DIR/${BASENAME}.bin"
# whisper.cpp strips a trailing quant suffix (-q5_1/-q8_0/…) from the .bin name
# before appending -encoder.mlmodelc, so do the same here.
ENCODER_STEM="$(echo "$BASENAME" | sed -E 's/-q[0-9]_[0-9k]$//')"
ENCODER="$MODELS_DIR/${ENCODER_STEM}-encoder.mlmodelc"

echo "▸ MLX_DIR   : $MLX_DIR"
echo "▸ basename  : $BASENAME   (quant: $QUANT)"
echo "▸ models dir: $MODELS_DIR"
echo

[ -f "$MLX_DIR/weights.safetensors" ] || { echo "ERROR: no weights.safetensors in $MLX_DIR"; exit 1; }
[ -x "$QUANTIZE_BIN" ] || { echo "ERROR: whisper-quantize not built at $QUANTIZE_BIN"; exit 1; }

# 1) MLX → GGML F16 (intermediate, decoder + vocab + mel filters)
echo "==> [1/3] MLX → GGML F16"
"$PY" "$REPO_DIR/convert_mlx_to_ggml.py" --mlx-dir "$MLX_DIR" --out "$F16"

# 2) F16 → quantized GGML (the decoder model the app ships)
echo "==> [2/3] quantize → $QUANT"
# whisper-quantize was built with an absolute rpath; point dyld at the dylibs.
LIBDIRS="$(find "$REPO_DIR/whisper.cpp/build" -name '*.dylib' -exec dirname {} \; 2>/dev/null | sort -u | paste -sd: -)"
DYLD_LIBRARY_PATH="$LIBDIRS" "$QUANTIZE_BIN" "$F16" "$FINAL_BIN" "$QUANT"

# 3) MLX → Core ML ANE encoder (.mlmodelc), named to match FINAL_BIN
echo "==> [3/3] MLX → ANE encoder .mlmodelc"
"$PY" "$REPO_DIR/convert_atc_to_coreml.py" --mlx-dir "$MLX_DIR" --out-mlmodelc "$ENCODER"

# 4) cleanup the big intermediate
rm -f "$F16"

echo
echo "✅ Done. Shipped into the app bundle (folder reference, no Xcode change needed):"
echo "   decoder : $FINAL_BIN"
echo "   encoder : $ENCODER"
echo "   Rebuild in Xcode; whisper.cpp will auto-load the ANE encoder on device."
