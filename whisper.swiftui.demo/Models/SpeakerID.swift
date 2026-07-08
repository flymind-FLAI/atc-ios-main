import Foundation
import CoreML
import Accelerate

/// On-device speaker embedding for ATC/PLT labeling, ported from atc-python's
/// SpeakerTracker. The Core ML model (speaker-id-v1.mlmodelc, 5.7 MB) is
/// resemblyzer's GE2E voice encoder with the librosa mel front-end baked in as
/// fixed conv kernels, so it takes raw 16 kHz waveform and returns a 256-d
/// L2-normalized embedding. Verified bit-parity (cosine 1.000000) against
/// resemblyzer on conversion.
final class SpeakerID {
    static let windowSamples = 25_840          // 1.615 s → exactly 160 mel frames
    private static let hopSamples = 12_800     // 0.8 s ≈ resemblyzer's 1.3 partials/s
    private static let embeddingDim = 256

    private let model: MLModel

    init?() {
        guard let url = Bundle.main.url(forResource: "speaker-id-v1",
                                        withExtension: "mlmodelc",
                                        subdirectory: "models"),
              let m = try? MLModel(contentsOf: url) else { return nil }
        self.model = m
    }

    /// Mean-of-partials utterance embedding (resemblyzer embed_utterance):
    /// 1.615 s windows hopped 0.8 s, averaged then re-normalized. Volume is
    /// normalized to -30 dBFS first (preprocess_wav); the VAD trim is skipped
    /// because the app's RMS gate already bounds the utterance to speech.
    func embedUtterance(_ samples: [Float]) -> [Float]? {
        guard !samples.isEmpty else { return nil }
        var wav = samples

        var meanSq: Float = 0
        vDSP_measqv(wav, 1, &meanSq, vDSP_Length(wav.count))
        if meanSq > 0 {
            var gain = powf(10, (-30 - 10 * log10f(meanSq)) / 20)
            vDSP_vsmul(wav, 1, &gain, &wav, 1, vDSP_Length(wav.count))
        }
        if wav.count < Self.windowSamples {
            wav += [Float](repeating: 0, count: Self.windowSamples - wav.count)
        }

        var sum = [Float](repeating: 0, count: Self.embeddingDim)
        var n = 0
        var start = 0
        while start + Self.windowSamples <= wav.count {
            guard let e = embedWindow(Array(wav[start..<start + Self.windowSamples])) else { return nil }
            vDSP_vadd(sum, 1, e, 1, &sum, 1, vDSP_Length(Self.embeddingDim))
            n += 1
            start += Self.hopSamples
        }
        guard n > 0 else { return nil }

        var normSq: Float = 0
        vDSP_svesq(sum, 1, &normSq, vDSP_Length(Self.embeddingDim))
        var inv = 1 / (sqrtf(normSq) + 1e-8)
        vDSP_vsmul(sum, 1, &inv, &sum, 1, vDSP_Length(Self.embeddingDim))
        return sum
    }

    private func embedWindow(_ window: [Float]) -> [Float]? {
        guard let arr = try? MLMultiArray(shape: [1, NSNumber(value: Self.windowSamples)],
                                          dataType: .float32) else { return nil }
        window.withUnsafeBufferPointer { buf in
            arr.dataPointer.assumingMemoryBound(to: Float.self)
                .update(from: buf.baseAddress!, count: window.count)
        }
        guard let input = try? MLDictionaryFeatureProvider(dictionary: ["waveform": arr]),
              let out = try? model.prediction(from: input),
              let emb = out.featureValue(for: "embedding")?.multiArrayValue else { return nil }
        let p = emb.dataPointer.assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: p, count: Self.embeddingDim))
    }
}

/// Adjacent-utterance voiceprint segmentation. We don't maintain speaker
/// identities — only the previous utterance's embedding. Each new utterance is
/// compared to the previous one; if the cosine similarity drops below the
/// threshold the voice changed, so we flip the label. ATC radio is half-duplex
/// (one party at a time, taking turns), so a flip-on-change rule naturally
/// produces an ATC↔PLT alternation without ever building a voiceprint library
/// — which is what stays robust under the heavy distortion of radio audio,
/// where long-lived centroids drift.
final class SpeakerTracker {
    private let threshold: Float
    private let labels = ["ATC", "PLT"]
    private var lastEmbedding: [Float]?
    private var labelIndex = 0

    init(threshold: Float = 0.82) {
        self.threshold = threshold
    }

    func identify(embedding emb: [Float]) -> String {
        defer { lastEmbedding = emb }
        guard let last = lastEmbedding else {
            labelIndex = 0
            return labels[0]
        }
        var dot: Float = 0
        vDSP_dotpr(emb, 1, last, 1, &dot, vDSP_Length(emb.count))
        var ne: Float = 0
        vDSP_svesq(emb, 1, &ne, vDSP_Length(emb.count))
        var nl: Float = 0
        vDSP_svesq(last, 1, &nl, vDSP_Length(last.count))
        let sim = dot / (sqrtf(ne) * sqrtf(nl) + 1e-8)
        if sim < threshold {           // voice changed → next speaker
            labelIndex = (labelIndex + 1) % labels.count
        }
        return labels[labelIndex]
    }

    func reset() {
        lastEmbedding = nil
        labelIndex = 0
    }
}
