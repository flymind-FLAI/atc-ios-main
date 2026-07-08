# Internal Guide

This guide maps the repository to the functions the internal team will touch most often. It is based on the current code in this repository.

## Product Function

ATC Copilot iOS is a local iOS app for:

- Capturing ATC radio audio from the device input.
- Running local Whisper transcription through `whisper.cpp`.
- Showing live tentative text and committed transcript entries.
- Normalizing ATC-specific text.
- Highlighting safety-relevant values.
- Running bundled sample clips through the same pipeline for smoke testing.

The app should be treated as an internal prototype unless a later release process says otherwise.

## Runtime Flow

```text
microphone or sample WAV
  -> Recorder / simulated sample buffer
  -> WhisperState streaming loop
  -> WhisperContext.fullTranscribe(...)
  -> raw Whisper text
  -> ATCNormalizer.normalize(...)
  -> TranscriptEntry + highlighted display
  -> SwiftUI transcript, live box, key-instruction chips
```

## Important Files

### `whisper.swiftui.demo/WhisperCppDemoApp.swift`

App entry point. It creates the SwiftUI app and opens `ContentView`.

### `whisper.swiftui.demo/UI/ContentView.swift`

Main UI composition:

- `TopStatusBar`: recording/VAD/model state.
- `StatusLine`: transient status and benchmark messages.
- `KeyInstructionStrip`: latest extracted runway, altitude, heading, frequency, squawk, QNH, and hold-short state.
- `LiveBox`: confirmed plus tentative live text.
- `TranscriptList`: committed transcript history.
- `BottomActionBar`: record, clear, sample, and test-clip actions.
- `ModelsView`: settings screen for model/config actions.

### `whisper.swiftui.demo/Models/WhisperState.swift`

Primary app controller. It owns UI state and coordinates recording, streaming, inference, normalization, speaker tagging, and benchmarks.

Key responsibilities:

- Load the default model from `Resources/models`.
- Refresh and select iOS audio inputs.
- Start/stop mic recording.
- Maintain sliding-window streaming state.
- Run preview inference for live text.
- Commit utterances after silence gating.
- Normalize committed transcript text.
- Update key instruction chips.
- Run bundled test clips through the same streaming path.
- Update or reset `atc_config.json` overrides.

Functions the team will likely touch:

- `loadModel(path:log:)`
- `toggleRecord()`
- `transcribeSample()`
- `transcribeAllClips()`
- `updateATCConfig(from:)`
- `resetATCConfig()`

### `whisper.cpp.swift/LibWhisper.swift`

Swift actor wrapper around the C `whisper.cpp` API.

Key functions:

- `WhisperContext.createContext(path:)`: loads a GGML model file.
- `fullTranscribe(...)`: runs Whisper on a sample buffer.
- `getTranscription()`: returns combined segment text.
- `getSegments()`: returns individual segment strings.
- `benchFull(...)`: low-level benchmark helper.

Notes:

- The actor protects the C context from concurrent access.
- Device builds enable Flash Attention in `whisper_context_default_params`.
- Simulator builds disable GPU use.

### `whisper.swiftui.demo/Utils/Recorder.swift`

Microphone capture utility.

It:

- Configures `AVAudioSession`.
- Selects a preferred input when provided.
- Captures audio through `AVAudioEngine`.
- Converts multi-channel input to mono.
- Resamples to 16 kHz.
- Stores samples behind a lock.
- Exposes `recentRMS(...)` for VAD-style activity checks.

### `whisper.swiftui.demo/Utils/RiffWaveUtils.swift`

Small WAV reader used by sample/test-clip paths. It decodes WAV into float samples.

### `whisper.swiftui.demo/Models/ATCNormalizer.swift`

ATC text cleanup and extraction logic.

Main entry:

- `normalize(_:filterHallucination:)`

Pipeline pieces:

- Join split words, such as `run way` to `runway`.
- Apply spelling and fuzzy corrections.
- Convert airline spoken names to ICAO-style labels when configured.
- Normalize context digits, phonetic letters, runway identifiers, frequencies, headings, and number words.
- Remove filler/noise words.
- Apply configured terminology replacements.
- Filter likely hallucinated repeated text.

Related types:

- `ATCHighlighter`: returns attributed text with safety-category highlighting.
- `ATCKeyState`: extracts latest operational values for UI chips.

### `whisper.swiftui.demo/Models/ATCConfigManager.swift`

Loads ATC configuration from:

1. Local document-directory override.
2. Bundled `Resources/atc_config.json`.

It can also download and cache a remote config. This is useful for updating airline/callsign/correction tables without changing app code.

### `whisper.swiftui.demo/Models/SpeakerID.swift`

Optional mic-session speaker labeling.

- `SpeakerID` loads `speaker-id-v1.mlmodelc` and creates a 256-d embedding from an utterance.
- `SpeakerTracker` compares adjacent embeddings and alternates labels when similarity drops below its threshold.

This does not identify real-world people. It only gives local adjacent-utterance labels such as `ATC` and `PLT`.

## Resources

### `whisper.swiftui.demo/Resources/atc_config.json`

Configurable ATC dictionaries:

- airline entries
- airport and waypoint terms
- custom terminology
- filler/noise words
- correction joins
- correction replacements
- Whisper initial prompt

### `whisper.swiftui.demo/Resources/models/`

Small speaker model is included. Large ASR model assets are expected to be generated or copied in locally.

Expected ASR assets for the default app path:

```text
flightai-asr-v1-q5_1.bin
flightai-asr-v1-encoder.mlmodelc/
ggml-silero-v5.1.2.bin
```

### `whisper.swiftui.demo/Resources/samples/`

Bundled WAV clips and `test_clips.json` for local smoke benchmarking.

## Model Conversion Scripts

### `build_ios_model.sh`

Orchestrates the end-to-end model build:

1. `convert_mlx_to_ggml.py`
2. `whisper-quantize`
3. `convert_atc_to_coreml.py`

### `convert_mlx_to_ggml.py`

Converts MLX Whisper safetensors/config into GGML F16.

### `convert_atc_to_coreml.py`

Converts the Whisper encoder into a Core ML `.mlmodelc` package. The output name must match the GGML decoder name convention expected by `whisper.cpp`.

### `convert_speaker_encoder.py`

Builds the small speaker encoder model package used by `SpeakerID`.

## Development Checklist

Before opening an internal PR:

- Build the Xcode project.
- Confirm the app launches with missing-model behavior if large ASR assets are not present.
- If model assets are present, run a short mic transcription.
- Run bundled test clips from the app settings and record model/device/iOS version with any numbers.
- Update this guide when changing streaming, normalization, model naming, or config shape.

## Known Boundaries

- Do not commit production ASR model binaries unless the team explicitly decides to track them.
- Do not commit private cockpit, tower, operator, or pilot data.
- Do not publish benchmark numbers without device/model/iOS version and sample-set details.
- Do not describe this app as certified avionics, navigation software, or flight-control software.
