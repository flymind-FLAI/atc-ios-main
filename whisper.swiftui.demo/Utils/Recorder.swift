import Foundation
import AVFoundation

final class Recorder: @unchecked Sendable {
    private var audioEngine: AVAudioEngine?
    private let samplesLock = NSLock()
    private var _samples: [Float] = []

    enum RecorderError: Error {
        case couldNotStartRecording
    }

    func startRecording(preferredInputUID: String? = nil) throws {
        samplesLock.withLock { _samples = [] }

#if !os(macOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record,
                                mode: .measurement,
                                options: [.duckOthers, .allowBluetooth])
        try session.setActive(true, options: .notifyOthersOnDeactivation)
        if let uid = preferredInputUID,
           let port = session.availableInputs?.first(where: { $0.uid == uid }) {
            try? session.setPreferredInput(port)
        }
#endif

        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let nativeFormat = inputNode.outputFormat(forBus: 0)
        let nativeSampleRate = nativeFormat.sampleRate
        let channelCount = Int(nativeFormat.channelCount)

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: nativeFormat) { [weak self] buffer, _ in
            guard let self, let channelData = buffer.floatChannelData else { return }
            let frameCount = Int(buffer.frameLength)

            var mono = [Float](repeating: 0, count: frameCount)
            for ch in 0..<channelCount {
                let channel = channelData[ch]
                for i in 0..<frameCount {
                    mono[i] += channel[i]
                }
            }
            if channelCount > 1 {
                let scale = 1.0 / Float(channelCount)
                for i in 0..<frameCount { mono[i] *= scale }
            }

            let ratio = 16000.0 / nativeSampleRate
            let newCount = Int(Double(frameCount) * ratio)
            guard newCount > 0 else { return }

            var resampled = [Float](repeating: 0, count: newCount)
            for i in 0..<newCount {
                let srcIdx = Double(i) / ratio
                let idx0 = Int(srcIdx)
                let idx1 = min(idx0 + 1, frameCount - 1)
                let frac = Float(srcIdx - Double(idx0))
                resampled[i] = mono[idx0] * (1 - frac) + mono[idx1] * frac
            }

            self.samplesLock.withLock {
                self._samples.append(contentsOf: resampled)
            }
        }

        engine.prepare()
        try engine.start()
        audioEngine = engine
    }

    func stopRecording() {
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine?.stop()
        audioEngine = nil
    }

    func getSamples() -> [Float] {
        samplesLock.withLock { _samples }
    }

    /// RMS of the most recent `windowSamples` samples (default 0.3s at 16kHz).
    func recentRMS(windowSamples: Int = 4800) -> Float {
        samplesLock.withLock {
            guard _samples.count >= windowSamples else { return 0 }
            let window = _samples.suffix(windowSamples)
            let sumSq = window.reduce(0.0) { $0 + Double($1 * $1) }
            return Float(sqrt(sumSq / Double(windowSamples)))
        }
    }
}
