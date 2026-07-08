# ATC Copilot iOS

ATC Copilot iOS is an internal FLAI iOS prototype for offline air-traffic-control radio transcription, ATC text cleanup, safety-keyword highlighting, and sample-clip benchmarking.

The app is based on a SwiftUI front end, a `whisper.cpp` Swift bridge, bundled ATC sample clips, a configurable ATC normalization layer, and model conversion scripts for preparing on-device Whisper assets.

## Current Repository State

This repository contains the app source and supporting tooling. It does not include the large production ASR decoder/encoder model binaries. Those are generated into `whisper.swiftui.demo/Resources/models/` by the model build pipeline.

Included:

- SwiftUI iOS app project: `whisper.swiftui.xcodeproj`.
- App source: `whisper.swiftui.demo/`.
- Swift bridge to `whisper.cpp`: `whisper.cpp.swift/`.
- Prebuilt `whisper.xcframework` used by the Xcode project.
- Speaker ID Core ML bundle: `speaker-id-v1.mlmodelc`.
- Bundled ATC sample clips and `test_clips.json`.
- ATC config: `whisper.swiftui.demo/Resources/atc_config.json`.
- Model conversion scripts: `build_ios_model.sh`, `convert_mlx_to_ggml.py`, `convert_atc_to_coreml.py`, `convert_speaker_encoder.py`.
- Architecture and pipeline notes in `docs/`.

Not included:

- Production Whisper GGML decoder model, such as `flightai-asr-v1-q5_1.bin`.
- Matching Core ML encoder bundle, such as `flightai-asr-v1-encoder.mlmodelc`.
- Training data, proprietary flight data, or server-side training code.

## What The App Does

- Records microphone audio or plays bundled sample WAV clips.
- Resamples audio to 16 kHz mono.
- Runs local Whisper inference through `whisper.cpp`.
- Uses a sliding-window stream for live preview and committed transcript entries.
- Normalizes ATC phraseology, numbers, runway identifiers, frequencies, callsigns, and common recognition errors.
- Highlights safety-relevant entities such as runways, altitudes, headings, frequencies, squawks, QNH, and hold-short phrases.
- Tracks the latest key instruction state as chips in the UI.
- Optionally labels adjacent mic utterances as `ATC` / `PLT` using a small speaker embedding model.
- Runs bundled test clips through the same streaming path for local smoke benchmarking.

## Quick Start

1. Open `whisper.swiftui.xcodeproj` in Xcode 16 or newer.
2. Select a physical iOS device for full model testing. Simulator builds are useful for UI and smaller-model checks only.
3. Generate or place the ASR model assets in:

```text
whisper.swiftui.demo/Resources/models/
```

Expected production-style names:

```text
flightai-asr-v1-q5_1.bin
flightai-asr-v1-encoder.mlmodelc/
ggml-silero-v5.1.2.bin
```

4. Build and run from Xcode.

The app can launch without the production ASR model, but transcription is unavailable until a compatible model is present.

## Model Build Pipeline

The intended local conversion path is:

```bash
./build_ios_model.sh /path/to/mlx-model-dir flightai-asr-v1-q5_1 q5_1
```

The MLX directory is expected to contain:

```text
weights.safetensors
config.json
```

The script performs:

1. MLX safetensors to GGML F16.
2. GGML F16 to quantized GGML via `whisper-quantize`.
3. MLX encoder to Core ML `.mlmodelc`.

Before running the script, build `whisper-quantize` from `whisper.cpp` and install the Python packages required by the conversion scripts.

## Main Code Paths

| Area | Files |
| --- | --- |
| App shell and UI | `whisper.swiftui.demo/UI/ContentView.swift` |
| Streaming state and transcript control | `whisper.swiftui.demo/Models/WhisperState.swift` |
| Whisper C bridge | `whisper.cpp.swift/LibWhisper.swift` |
| Recorder and audio buffering | `whisper.swiftui.demo/Utils/Recorder.swift` |
| WAV decoding | `whisper.swiftui.demo/Utils/RiffWaveUtils.swift` |
| ATC normalization, highlighting, key-state extraction | `whisper.swiftui.demo/Models/ATCNormalizer.swift` |
| Config loading and remote override | `whisper.swiftui.demo/Models/ATCConfigManager.swift` |
| Speaker embedding and adjacent-speaker tracker | `whisper.swiftui.demo/Models/SpeakerID.swift` |
| Model list/download UI support | `whisper.swiftui.demo/Models/Model.swift`, `whisper.swiftui.demo/UI/DownloadButton.swift` |
| Xcode project | `whisper.swiftui.xcodeproj/` |

## Internal Docs

- [Internal guide](docs/INTERNAL_GUIDE.md)
- [Architecture notes](docs/01-architecture.md)
- [Model and inference notes](docs/02-model-and-inference.md)
- [Streaming pipeline notes](docs/03-streaming-pipeline.md)
- [Text normalization notes](docs/04-text-normalization.md)
- [Speaker labeling notes](docs/05-speaker-labeling.md)
- [Build and deploy notes](docs/06-build-and-deploy.md)

## Accuracy And Benchmark Notes

Any WER, RTF, battery, memory, or latency number should be treated as environment-specific until re-run on the target model, target device, and target iOS version. The bundled sample clips are useful for smoke tests and A/B checks, not for final product claims.

## Safety Boundary

This is an internal transcription and decision-support prototype. It is not certified avionics, not navigation software, and not an aircraft control system.
