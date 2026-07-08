import Foundation
import UIKit
import whisper

enum WhisperError: Error {
    case couldNotInitializeContext
}

// Meet Whisper C++ constraint: Don't access from more than one thread at a time.
actor WhisperContext {
    private var context: OpaquePointer

    init(context: OpaquePointer) {
        self.context = context
    }

    deinit {
        whisper_free(context)
    }

    /// - Parameter preview: live-preview mode — shaves per-pass cost where the
    ///   output is transient: skips timestamp tokens (~74ms), disables the
    ///   temperature-fallback re-decodes (kills multi-second outlier passes),
    ///   and caps tokens per segment so a repetition loop can't stall the
    ///   stream. Commit passes keep full quality.
    func fullTranscribe(
        samples: [Float],
        singleSegment: Bool = false,
        initialPrompt: String? = nil,
        carryInitialPrompt: Bool = false,
        useVAD: Bool = true,
        preview: Bool = false
    ) {
        let maxThreads = max(1, min(8, cpuCount() - 2))
        let trimmedPrompt = initialPrompt?.trimmingCharacters(in: .whitespacesAndNewlines)
        let usePrompt = !(trimmedPrompt?.isEmpty ?? true)

        // Only resolve the VAD model path when it will actually be used.
        let vadModelNSString: NSString? = useVAD
            ? Bundle.main
                .url(forResource: "ggml-silero-v5.1.2", withExtension: "bin", subdirectory: "models")
                .map { NSString(string: $0.path) }
            : nil

        "en".withCString { en in
            var params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY)
            params.print_realtime   = false
            params.print_progress   = false
            params.print_timestamps = !singleSegment
            params.print_special    = false
            params.translate        = false
            params.language         = en
            params.n_threads        = Int32(maxThreads)
            params.offset_ms        = 0
            params.no_context       = true
            params.single_segment   = singleSegment

            // Temperature fallbacks are disabled EVERYWHERE: A/B over 100
            // atco2_test clips measured identical WER (19.8% vs 19.9%) with
            // them on vs off, while "on" produced wild hallucinations
            // ("china southern" → "air berlin") and multi-second retry spikes.
            params.temperature_inc = 0

            if preview {
                params.no_timestamps   = true   // don't decode tokens nobody displays
                params.max_tokens      = 96     // bound a runaway repetition loop
            }

            if useVAD, let cPath = vadModelNSString?.utf8String {
                params.vad            = true
                params.vad_model_path = cPath
            }

            let runInference: () -> Void = { [self] in
                whisper_reset_timings(self.context)
                samples.withUnsafeBufferPointer { samples in
                    if (whisper_full(self.context, params, samples.baseAddress, Int32(samples.count)) != 0) {
                        print("Failed to run the model")
                    } else {
#if DEBUG
                        whisper_print_timings(self.context)
#endif
                    }
                }
            }

            if usePrompt, let p = trimmedPrompt {
                p.withCString { promptC in
                    params.initial_prompt = promptC
                    params.carry_initial_prompt = carryInitialPrompt
                    runInference()
                }
            } else {
                runInference()
            }
        }
    }

    func getTranscription() -> String {
        var transcription = ""
        for i in 0..<whisper_full_n_segments(context) {
            transcription += String.init(cString: whisper_full_get_segment_text(context, i))
        }
        return transcription
    }

    func getSegments() -> [String] {
        (0..<whisper_full_n_segments(context)).map {
            String(cString: whisper_full_get_segment_text(context, $0))
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }.filter { !$0.isEmpty }
    }

    static func benchMemcpy(nThreads: Int32) async -> String {
        return String.init(cString: whisper_bench_memcpy_str(nThreads))
    }

    static func benchGgmlMulMat(nThreads: Int32) async -> String {
        return String.init(cString: whisper_bench_ggml_mul_mat_str(nThreads))
    }

    private func systemInfo() -> String {
        var info = ""
        //if (ggml_cpu_has_neon() != 0) { info += "NEON " }
        return String(info.dropLast())
    }

    func benchFull(modelName: String, nThreads: Int32) async -> String {
        let nMels = whisper_model_n_mels(context)
        if (whisper_set_mel(context, nil, 0, nMels) != 0) {
            return "error: failed to set mel"
        }

        // heat encoder
        if (whisper_encode(context, 0, nThreads) != 0) {
            return "error: failed to encode"
        }

        var tokens = [whisper_token](repeating: 0, count: 512)

        // prompt heat
        if (whisper_decode(context, &tokens, 256, 0, nThreads) != 0) {
            return "error: failed to decode"
        }

        // text-generation heat
        if (whisper_decode(context, &tokens, 1, 256, nThreads) != 0) {
            return "error: failed to decode"
        }

        whisper_reset_timings(context)

        // actual run
        if (whisper_encode(context, 0, nThreads) != 0) {
            return "error: failed to encode"
        }

        // text-generation
        for i in 0..<256 {
            if (whisper_decode(context, &tokens, 1, Int32(i), nThreads) != 0) {
                return "error: failed to decode"
            }
        }

        // batched decoding
        for _ in 0..<64 {
            if (whisper_decode(context, &tokens, 5, 0, nThreads) != 0) {
                return "error: failed to decode"
            }
        }

        // prompt processing
        for _ in 0..<16 {
            if (whisper_decode(context, &tokens, 256, 0, nThreads) != 0) {
                return "error: failed to decode"
            }
        }

        whisper_print_timings(context)

        let deviceModel = await UIDevice.current.model
        let systemName = await UIDevice.current.systemName
        let systemInfo = self.systemInfo()
        let timings: whisper_timings = whisper_get_timings(context).pointee
        let encodeMs = String(format: "%.2f", timings.encode_ms)
        let decodeMs = String(format: "%.2f", timings.decode_ms)
        let batchdMs = String(format: "%.2f", timings.batchd_ms)
        let promptMs = String(format: "%.2f", timings.prompt_ms)
        return "| \(deviceModel) | \(systemName) | \(systemInfo) | \(modelName) | \(nThreads) | 1 | \(encodeMs) | \(decodeMs) | \(batchdMs) | \(promptMs) | <todo> |"
    }

    static func createContext(path: String) throws -> WhisperContext {
        var params = whisper_context_default_params()
#if targetEnvironment(simulator)
        params.use_gpu = false
        print("Running on the simulator, using CPU")
#else
        params.flash_attn = true
#endif
        let context = whisper_init_from_file_with_params(path, params)
        if let context {
            return WhisperContext(context: context)
        } else {
            print("Couldn't load model at \(path)")
            throw WhisperError.couldNotInitializeContext
        }
    }
}

fileprivate func cpuCount() -> Int {
    ProcessInfo.processInfo.processorCount
}
