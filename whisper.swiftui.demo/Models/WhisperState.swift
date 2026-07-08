import Foundation
import SwiftUI
import AVFoundation
#if canImport(UIKit)
import UIKit
#endif

struct TranscriptEntry: Identifiable {
    let id = UUID()
    let timestamp: Date
    let text: String
    /// Voiceprint label ("ATC" / "PLT" / "?") — mic sessions only; nil for
    /// benchmark clips, whose random airports would confuse the tracker.
    let speaker: String?
    /// Highlight is computed ONCE at commit time — running the regex set on
    /// every SwiftUI render made the whole list re-scan on each live update.
    let display: AttributedString

    init(timestamp: Date, text: String, speaker: String? = nil) {
        self.timestamp = timestamp
        self.text = text
        self.speaker = speaker
        self.display = ATCHighlighter.highlight(text)
    }

    var timeString: String {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss"
        return f.string(from: timestamp)
    }
}

struct AudioInputOption: Identifiable, Hashable {
    let id: String      // portDescription.uid
    let name: String    // portDescription.portName
    let kind: Kind
    enum Kind { case builtIn, bluetooth, lineIn, usb, headset, other
        var label: String {
            switch self {
            case .builtIn: return "Mic"
            case .bluetooth: return "BT"
            case .lineIn: return "Line In"
            case .usb: return "USB"
            case .headset: return "Headset"
            case .other: return "Other"
            }
        }
        var icon: String {
            switch self {
            case .builtIn: return "mic.fill"
            case .bluetooth: return "headphones"
            case .lineIn: return "cable.connector"
            case .usb: return "cable.connector.horizontal"
            case .headset: return "headphones.circle"
            case .other: return "dot.radiowaves.left.and.right"
            }
        }
    }
}

@MainActor
class WhisperState: ObservableObject {
    @Published var entries: [TranscriptEntry] = []
    /// Latest value per safety-critical category (runway, altitude, …) —
    /// the "currently assigned" instruction state shown as chips under the
    /// status row. A new transmission only overwrites the slots it mentions.
    @Published var keyInstructions: [ATCKeyState.Kind: String] = [:]
    @Published var statusText = ""
    @Published var modelName = ""
    @Published var canTranscribe = false
    @Published var isRecording = false
    // Two-tier live display, ported from atc_app.py: the word-level common
    // prefix of consecutive passes is "confirmed" (never regresses), the rest
    // is a gray "tentative" tail. Same inference cadence, but text grows
    // steadily instead of rewriting wholesale — this is what makes the Mac
    // version feel fast.
    @Published var liveConfirmed = ""
    @Published var liveTentative = ""
    private var lastLiveRaw = ""
    @Published var atcConfigVersion = "–"
    @Published var isUpdatingConfig = false
    @Published var vadSpeaking = false
    @Published var silenceCutoffSec: Double = 0.6
    @Published var lastLatencyMs: Double = 0
    @Published var lastRTF: Double = 0
    @Published var availableInputs: [AudioInputOption] = []
    @Published var selectedInputUID: String?

    private var whisperContext: WhisperContext?
    private let recorder = Recorder()
    private var audioPlayer: AVAudioPlayer?
    private var atcConfig: ATCConfig?
    private var transcriptionTask: Task<Void, Never>?
    private var isTranscribing = false

    // Sliding window streaming state.
    // All speech/silence timing is derived from the AUDIO BUFFER (wall truth),
    // never from loop-tick counting: inference passes block the polling loop
    // for ~1.2s, and tick-counted clocks freeze during that await — the audit
    // traced ~0.5s/pass of dead gating and 1-3s of commit lag to exactly that.
    private var windowStartSample = 0
    private let maxWindowSamples = 28 * 16000        // stay under whisper's 30s window: one encoder pass per preview
    private let keepOverlapSamples = 200 * 16        // 200ms pre-speech overlap
    private let silenceRMSThreshold: Float = 0.008   // was 0.012; quiet radio audio was detected late
    private let minPreviewGap: Double = 0.3          // floor between preview STARTS (a large pass takes longer anyway)
    private let minUtteranceSamples = 5120           // 0.32s before first preview/commit (Mac parity)
    /// Window end (sample count) snapshot of the most recent completed preview —
    /// used at commit time to detect whether the hypothesis misses tail words.
    private var lastPreviewWindowEnd = 0

    // Speaker labeling (mic sessions only). Lazy so the 5.7 MB Core ML load
    // doesn't sit on app launch; nil if the model is missing from the bundle.
    private lazy var speakerID: SpeakerID? = SpeakerID()
    private let speakerTracker = SpeakerTracker()

    // Streaming source: nil = live microphone (Recorder); non-nil = a buffer
    // the benchmark grows in real time, so bundled clips run through the EXACT
    // same pipeline the mic uses (sliding-window preview, silence segmentation,
    // commit, speaker tagging). A benchmark that one-shot transcribed each clip
    // would bypass all of that and prove nothing about the live experience.
    private var simulatedSamples: [Float]?
    private func currentSamples() -> [Float] { simulatedSamples ?? recorder.getSamples() }
    /// Drives the transcription loop for BOTH mic recording and benchmark
    /// playback. (isRecording stays mic-only, so the UI chrome is unaffected.)
    private var streamActive = false

    // Benchmark accumulators — only touched while a benchmark is streaming.
    private var benchmarkActive = false
    private var benchRawHyps: [String] = []

    private var builtInModelUrl: URL? {
        // Simulator can't handle the 1.1 GB ATC Q5_1 model (Metal working set is 0,
        // CPU_REPACK OOMs). Skip auto-load there; user picks a small model via Settings.
        #if targetEnvironment(simulator)
        return nil
        #else
        return Bundle.main.url(forResource: "flightai-asr-v1-q5_1", withExtension: "bin", subdirectory: "models")
        #endif
    }

    private var sampleUrl: URL? {
        Bundle.main.url(forResource: "jfk", withExtension: "wav", subdirectory: "samples")
    }

    private var whisperInitialPrompt: String {
        atcConfig?.effectiveWhisperInitialPrompt ?? ATCConfig.defaultWhisperInitialPrompt
    }

    init() {
        loadATCConfig()
        refreshInputs()
        statusText = "Loading model…"
        let url = builtInModelUrl
        Task.detached(priority: .userInitiated) { [weak self] in
            await Self.loadModelOffMain(url: url, state: self)
        }
    }

    nonisolated private static func loadModelOffMain(url: URL?, state: WhisperState?) async {
        guard let state else { return }
        guard let url else {
            await MainActor.run {
                state.modelName = ""
                state.statusText = "Model not found"
            }
            return
        }
        do {
            let ctx = try WhisperContext.createContext(path: url.path())
            let name = url.deletingPathExtension().lastPathComponent
            await MainActor.run {
                state.whisperContext = ctx
                state.modelName = name
                state.statusText = "Ready"
                state.canTranscribe = true
            }
        } catch {
            let msg = error.localizedDescription
            await MainActor.run {
                state.statusText = msg
            }
        }
    }

    func refreshInputs() {
#if !os(macOS)
        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.record,
                                 mode: .measurement,
                                 options: [.duckOthers, .allowBluetooth])
        let ports = session.availableInputs ?? []
        availableInputs = ports.map { p in
            let kind: AudioInputOption.Kind
            switch p.portType {
            case .builtInMic: kind = .builtIn
            case .bluetoothHFP, .bluetoothA2DP, .bluetoothLE: kind = .bluetooth
            case .lineIn: kind = .lineIn
            case .usbAudio: kind = .usb
            case .headsetMic, .headphones: kind = .headset
            default: kind = .other
            }
            return AudioInputOption(id: p.uid, name: p.portName, kind: kind)
        }
        if selectedInputUID == nil || !availableInputs.contains(where: { $0.id == selectedInputUID }) {
            selectedInputUID = availableInputs.first?.id
        }
#endif
    }

    func loadModel(path: URL? = nil, log: Bool = true) {
        do {
            whisperContext = nil
            if log { statusText = "Loading model…" }
            let modelUrl = path ?? builtInModelUrl
            if let modelUrl {
                whisperContext = try WhisperContext.createContext(path: modelUrl.path())
                modelName = modelUrl.deletingPathExtension().lastPathComponent
                if log { statusText = "Ready" }
            } else {
                modelName = ""
                statusText = "Model not found"
            }
            canTranscribe = true
        } catch {
            statusText = error.localizedDescription
        }
    }

    func benchCurrentModel() async {
        guard whisperContext != nil else {
            statusText = "No model loaded"
            return
        }
        statusText = "Benchmarking…"
        let result = await whisperContext?.benchFull(modelName: "<current>", nThreads: Int32(min(4, cpuCount())))
        if let result { statusText = result }
    }

    func bench(models: [Model]) async {
        let nThreads = Int32(min(4, cpuCount()))
        statusText = "Benchmarking…"
        for model in models {
            loadModel(path: model.fileURL, log: false)
            guard whisperContext != nil else { break }
            _ = await whisperContext?.benchFull(modelName: model.name, nThreads: nThreads)
        }
        statusText = "Benchmark done"
    }

    func transcribeSample() async {
        guard let sampleUrl else {
            statusText = "Sample not found"
            return
        }
        guard canTranscribe, let whisperContext else { return }

        do {
            canTranscribe = false
            stopPlayback()
            try startPlayback(sampleUrl)
            let data = try decodeWaveFile(sampleUrl)
            statusText = "Transcribing sample…"
            await whisperContext.fullTranscribe(
                samples: data,
                initialPrompt: whisperInitialPrompt,
                carryInitialPrompt: false
            )
            let text = await whisperContext.getTranscription()
            let normalizer = ATCNormalizer(config: atcConfig)
            let normalized = normalizer.normalize(text)
            if !normalized.isEmpty {
                appendTranscript(normalized)
            }
            statusText = "Sample done"
        } catch {
            statusText = error.localizedDescription
        }
        canTranscribe = true
    }

    /// Append a committed line and fold its key elements into the chip state.
    private func appendTranscript(_ line: String, at time: Date = Date(), speaker: String? = nil) {
        entries.append(TranscriptEntry(timestamp: time, text: line, speaker: speaker))
        for (k, v) in ATCKeyState.extract(from: line) { keyInstructions[k] = v }
    }

    /// Voiceprint label for one committed utterance. The embedding runs off
    /// the main actor (a few LSTM windows, ~10-30 ms) so commits never stall
    /// the UI; the tracker itself is touched only from here (main actor).
    private func identifySpeaker(in samples: [Float]) async -> String? {
        guard samples.count >= minUtteranceSamples, let speakerID else { return nil }
        let emb = await Task.detached(priority: .userInitiated) {
            speakerID.embedUtterance(samples)
        }.value
        guard let emb else { return nil }
        return speakerTracker.identify(embedding: emb)
    }

    func clearTranscript() {
        entries.removeAll()
        keyInstructions.removeAll()
    }

    // MARK: - Bundled Test Clips

    var bundledTestClips: [URL] {
        let urls = Bundle.main.urls(forResourcesWithExtension: "wav", subdirectory: "samples") ?? []
        return urls.sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    private struct TestClipRef: Decodable {
        let text: String
        let duration: Double
    }

    /// filename → reference transcript, from samples/test_clips.json (atco2_test split).
    private lazy var testClipRefs: [String: TestClipRef] = {
        guard let url = Bundle.main.url(forResource: "test_clips", withExtension: "json", subdirectory: "samples"),
              let data = try? Data(contentsOf: url),
              let refs = try? JSONDecoder().decode([String: TestClipRef].self, from: data)
        else { return [:] }
        return refs
    }()

    func referenceText(for url: URL) -> String? {
        testClipRefs[url.lastPathComponent]?.text
    }

    // MARK: - Keep-awake

    /// Keep the screen on while we're actively listening or benchmarking —
    /// a transcription tool that lets the screen sleep mid-session is useless.
    private func setKeepAwake(_ on: Bool) {
#if canImport(UIKit)
        UIApplication.shared.isIdleTimerDisabled = on
#endif
    }

    // MARK: - Live two-tier text (stable prefix + typewriter reveal)

    // Inference lands every ~1.2s and produces several new words at once.
    // Dumping the batch on screen reads as chunky; holding it back at a fixed
    // rate reads as laggy. The adaptive typewriter drains the pending words at
    // a rate sized to finish JUST before the next batch is due, so the live
    // box streams word-by-word continuously with near-zero added latency.
    //
    // Everything here is WORD-based. Whisper's casing/punctuation jitters
    // between passes, so character-offset slicing cuts words in half and
    // produces garbled/duplicated text — the display is rebuilt from word
    // arrays instead, which makes duplication structurally impossible.
    private var targetConfirmedWords: [String] = []
    private var targetTentativeWords: [String] = []
    private var displayedWordCount = 0
    private var typewriterTask: Task<Void, Never>?

    private func resetLive() {
        typewriterTask?.cancel()
        typewriterTask = nil
        targetConfirmedWords = []
        targetTentativeWords = []
        displayedWordCount = 0
        liveConfirmed = ""
        liveTentative = ""
        lastLiveRaw = ""
    }

    private func updateLiveStable(with rawText: String) {
        let curr = rawText.trimmingCharacters(in: .whitespacesAndNewlines)
        let cw = curr.split(separator: " ").map(String.init)
        let pw = lastLiveRaw.split(separator: " ").map(String.init)

        // Stable prefix (atc_app.py): words agreeing between consecutive passes.
        var stableN = 0
        while stableN < min(pw.count, cw.count),
              pw[stableN].lowercased() == cw[stableN].lowercased() { stableN += 1 }

        if stableN > targetConfirmedWords.count {
            targetConfirmedWords = Array(cw[0..<stableN])   // adopt current pass's casing
        }
        let n = min(targetConfirmedWords.count, cw.count)
        targetTentativeWords = cw.count > n ? Array(cw[n...]) : []
        lastLiveRaw = curr

        syncDisplay()
        startTypewriter()
    }

    /// Rebuild the display from the word arrays: the first `displayedWordCount`
    /// words of the full hypothesis are visible; within them, confirmed words
    /// are solid and the rest gray. The live box shows the RAW spoken words
    /// exactly as heard ("alpha charlie", "two seven left"); normalization to
    /// "AC" / "27 left" happens only when the line commits to the transcript.
    private func syncDisplay() {
        let full = targetConfirmedWords + targetTentativeWords
        displayedWordCount = min(displayedWordCount, full.count)
        let visible = full.prefix(displayedWordCount)
        let solidCount = min(displayedWordCount, targetConfirmedWords.count)
        liveConfirmed = visible.prefix(solidCount).joined(separator: " ")
        liveTentative = visible.dropFirst(solidCount).joined(separator: " ")
    }

    /// Fast-drain any words the typewriter hasn't revealed yet (35ms/word,
    /// typically ≤0.2s), then hold a beat so the completed line registers
    /// before it moves down into the transcript.
    private func finishLiveReveal() async {
        typewriterTask?.cancel()
        typewriterTask = nil
        let full = targetConfirmedWords + targetTentativeWords
        while displayedWordCount < full.count {
            displayedWordCount += 1
            syncDisplay()
            try? await Task.sleep(nanoseconds: 35_000_000)
        }
        try? await Task.sleep(nanoseconds: 150_000_000)
    }

    private func startTypewriter() {
        guard typewriterTask == nil else { return }
        typewriterTask = Task { @MainActor in
            defer { typewriterTask = nil }
            while !Task.isCancelled {
                let total = targetConfirmedWords.count + targetTentativeWords.count
                let pending = total - displayedWordCount
                guard pending > 0 else { break }
                displayedWordCount += 1
                syncDisplay()
                // Adaptive rate: drain the backlog over ~1s (the gap until the
                // next batch), clamped to 30–180ms per word. Small batch →
                // relaxed typing; big backlog → fast burst that catches up.
                let interval = min(0.18, max(0.03, 1.0 / Double(pending)))
                try? await Task.sleep(nanoseconds: UInt64(interval * 1_000_000_000))
            }
        }
    }

    private func werTokens(_ s: String) -> [String] {
        s.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
    }

    /// Word-level Levenshtein distance (substitutions + insertions + deletions).
    private func wordErrors(ref: [String], hyp: [String]) -> Int {
        if ref.isEmpty { return hyp.count }
        if hyp.isEmpty { return ref.count }
        var dp = Array(0...hyp.count)
        for i in 1...ref.count {
            var prev = dp[0]
            dp[0] = i
            for j in 1...hyp.count {
                let cur = dp[j]
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ref[i - 1] == hyp[j - 1] ? 0 : 1))
                prev = cur
            }
        }
        return dp[hyp.count]
    }

    /// Run a random 10-clip sample through the REAL streaming pipeline. Each
    /// clip's samples are appended to a simulated buffer in real time (while the
    /// clip plays out loud), so sliding-window preview, silence segmentation,
    /// commit and speaker tagging all run EXACTLY as they do for the live mic —
    /// the whole point: a benchmark that one-shot transcribed each clip would
    /// bypass the streaming behaviour it's meant to measure. A ~1s silence gap
    /// after each clip is the end-of-transmission cue, like releasing PTT.
    func transcribeAllClips() async {
        let clips = Array(bundledTestClips.shuffled().prefix(10))
        guard !clips.isEmpty else { statusText = "No test clips bundled"; return }
        guard canTranscribe, whisperContext != nil else { return }

        setKeepAwake(true)
        defer { setKeepAwake(false) }
        canTranscribe = false

        // Enter simulated-streaming mode and start the real transcription loop.
        speakerTracker.reset()
        windowStartSample = 0
        simulatedSamples = []
        benchmarkActive = true
        benchRawHyps = []
        resetLive()
        streamActive = true
        statusText = "Benchmarking…"
        startTranscriptionLoop()

        var totalAudio = 0.0
        var totalErrors = 0
        var totalRefWords = 0
        var totalLatency = 0.0
        var committedClips = 0
        let gapSec = 1.0   // > silenceCutoffSec so each clip ends its own transmission

        for url in clips {
            guard let samples = try? decodeWaveFile(url) else { continue }
            let audioSec = Double(samples.count) / 16000.0
            totalAudio += audioSec
            let hypStart = benchRawHyps.count

            stopPlayback()
            try? startPlayback(url)
            await feedSamples(samples, over: audioSec)
            let audioEnd = Date()
            await feedSilence(seconds: gapSec)
            await waitForCommit(after: hypStart, timeout: 6.0)

            if benchRawHyps.count > hypStart {
                committedClips += 1
                totalLatency += Date().timeIntervalSince(audioEnd)
                if let ref = referenceText(for: url) {
                    let hyp = benchRawHyps[hypStart...].joined(separator: " ")
                    let refTok = werTokens(ref)
                    totalErrors += wordErrors(ref: refTok, hyp: werTokens(hyp))
                    totalRefWords += refTok.count
                }
            }
        }

        // Leave streaming mode and drain anything still pending.
        streamActive = false
        transcriptionTask?.cancel()
        transcriptionTask = nil
        await finalTranscribe()
        stopPlayback()
        benchmarkActive = false
        simulatedSamples = nil
        resetLive()
        windowStartSample = 0

        var summary = String(format: "All %d clips — %.1fs audio", clips.count, totalAudio)
        if committedClips > 0 {
            summary += String(format: ", end-to-end %.1fs", totalLatency / Double(committedClips))
        }
        if totalRefWords > 0 {
            summary += String(format: ", WER %.0f%%", Double(totalErrors) / Double(totalRefWords) * 100)
        }
        entries.append(TranscriptEntry(timestamp: Date(), text: "== \(summary) =="))
        statusText = "Benchmark done"
        canTranscribe = true
    }

    /// Append `samples` to the simulated buffer at real-time speed, so the
    /// streaming loop sees audio arrive exactly as the mic would deliver it.
    private func feedSamples(_ samples: [Float], over seconds: Double) async {
        let start = Date()
        var fed = 0
        while Date().timeIntervalSince(start) < seconds {
            let upTo = min(samples.count, Int(Date().timeIntervalSince(start) * 16000))
            if upTo > fed { simulatedSamples?.append(contentsOf: samples[fed..<upTo]); fed = upTo }
            try? await Task.sleep(nanoseconds: 40_000_000)
        }
        if fed < samples.count { simulatedSamples?.append(contentsOf: samples[fed...]) }
    }

    /// Feed real-time silence — the end-of-transmission cue that makes the loop
    /// commit the clip and run speaker segmentation, like releasing PTT.
    private func feedSilence(seconds: Double) async {
        let total = Int(seconds * 16000)
        let start = Date()
        var fed = 0
        while Date().timeIntervalSince(start) < seconds {
            let upTo = min(total, Int(Date().timeIntervalSince(start) * 16000))
            if upTo > fed { simulatedSamples?.append(contentsOf: repeatElement(Float(0), count: upTo - fed)); fed = upTo }
            try? await Task.sleep(nanoseconds: 40_000_000)
        }
        if fed < total { simulatedSamples?.append(contentsOf: repeatElement(Float(0), count: total - fed)) }
    }

    /// Block until the loop commits at least one new transmission (or times out).
    private func waitForCommit(after count: Int, timeout: Double) async {
        let start = Date()
        while benchRawHyps.count <= count, Date().timeIntervalSince(start) < timeout {
            try? await Task.sleep(nanoseconds: 80_000_000)
        }
    }

    // MARK: - Streaming Record + Transcribe

    func toggleRecord() {
        if isRecording {
            stopRecordingAndTranscribe()
        } else {
            startRecordingStream()
        }
    }

    private func startRecordingStream() {
        requestRecordPermission { [weak self] granted in
            guard granted, let self else { return }
            Task { @MainActor in
                do {
                    self.stopPlayback()
                    self.windowStartSample = 0
                    // Fresh segmentation per session: first voice heard = ATC,
                    // label flips whenever the next utterance's voice differs.
                    self.speakerTracker.reset()
                    try self.recorder.startRecording(preferredInputUID: self.selectedInputUID)
                    self.isRecording = true
                    self.streamActive = true
                    self.setKeepAwake(true)
                    self.resetLive()
                    self.statusText = "Listening…"
                    self.startTranscriptionLoop()
                } catch {
                    self.statusText = error.localizedDescription
                    self.isRecording = false
                }
            }
        }
    }

    private func stopRecordingAndTranscribe() {
        isRecording = false
        streamActive = false
        setKeepAwake(false)
        canTranscribe = false
        transcriptionTask?.cancel()
        transcriptionTask = nil
        recorder.stopRecording()
        statusText = "Processing…"

        Task {
            await finalTranscribe()
            canTranscribe = true
            statusText = isRecording ? "Listening…" : "Ready"
        }
    }

    private func startTranscriptionLoop() {
        transcriptionTask = Task {
            let tickNanos: UInt64 = 80_000_000
            var utteranceActive = false
            var lastPreviewStart = Date.distantPast

            while streamActive && !Task.isCancelled {
                try? await Task.sleep(nanoseconds: tickNanos)
                guard streamActive, !Task.isCancelled else { break }

                let allSamples = currentSamples()
                let allCount = allSamples.count
                // Silence measured from the buffer itself — accurate even right
                // after a 1.2s inference await froze the loop.
                let silence = Self.silenceDuration(in: allSamples, threshold: silenceRMSThreshold)
                let speaking = silence < 0.12
                vadSpeaking = speaking

                if !utteranceActive {
                    if speaking {
                        utteranceActive = true
                    } else {
                        windowStartSample = max(windowStartSample, allCount - keepOverlapSamples)
                        continue
                    }
                }

                let windowSize = allCount - windowStartSample

                // End of transmission (or window cap): commit NOW.
                if (silence >= silenceCutoffSec && windowSize >= minUtteranceSamples)
                    || windowSize >= maxWindowSamples {
                    await commitUtterance(allSamples: allSamples, silence: silence)
                    utteranceActive = false
                    lastPreviewStart = .distantPast   // next utterance previews immediately
                    continue
                }

                // Live preview, serial and paced from pass START — zero dead time
                // after a pass. Stop launching new passes once the silence tail
                // is under way, so the commit isn't stuck behind a fresh 1.2s pass.
                if silence < 0.3,
                   windowSize > minUtteranceSamples,
                   Date().timeIntervalSince(lastPreviewStart) >= minPreviewGap {
                    lastPreviewStart = Date()
                    await updateLiveText()
                    // Brief breather: back-to-back passes run the ANE/GPU at
                    // 100% duty and thermal-throttle long sessions into the
                    // ground. 150ms costs nothing perceptually (the typewriter
                    // is still revealing) and sheds real heat.
                    try? await Task.sleep(nanoseconds: 150_000_000)
                }
            }
        }
    }

    /// Seconds of silence at the end of the buffer (scans backwards in 80ms
    /// blocks, capped at ~4.8s). Buffer-derived, so it keeps ticking while the
    /// polling loop is blocked on an inference await.
    nonisolated private static func silenceDuration(in samples: [Float], threshold: Float) -> Double {
        let block = 1280                       // 80ms @ 16kHz
        let maxBlocks = 60                     // scan at most ~4.8s back
        var end = samples.count
        var scanned = 0
        while end > 0, scanned < maxBlocks {
            let start = max(0, end - block)
            var sum: Float = 0
            for i in start..<end { sum += samples[i] * samples[i] }
            let rms = sqrtf(sum / Float(end - start))
            if rms > threshold {
                return Double(samples.count - end) / 16000.0
            }
            end = start
            scanned += 1
        }
        return Double(scanned) * 0.08
    }

    private func updateLiveText() async {
        guard let whisperContext else { return }
        let allSamples = currentSamples()
        let windowSamples = Array(allSamples[windowStartSample...])
        guard windowSamples.count > minUtteranceSamples else { return }

        isTranscribing = true
        await whisperContext.fullTranscribe(
            samples: windowSamples,
            singleSegment: true,
            initialPrompt: nil,   // 68-token prompt costs ~140ms/pass; preview text is transient
            useVAD: false,        // app already gates previews via RMS VAD
            preview: true         // no timestamp tokens, no fallback re-decodes, capped tokens
        )
        let text = await whisperContext.getTranscription()
        updateLiveStable(with: text)
        lastPreviewWindowEnd = allSamples.count
        isTranscribing = false
    }

    /// Commit the utterance the way the Mac app does: reuse the last preview
    /// hypothesis with ZERO extra inference. Only when the hypothesis is stale
    /// (speech continued meaningfully past the last preview's window) or absent
    /// (utterance too short to ever preview) do we pay for one final pass.
    private func commitUtterance(allSamples: [Float], silence: Double) async {
        let hypothesisWords = targetConfirmedWords + targetTentativeWords
        let speechEndSample = allSamples.count - Int(silence * 16000)
        let uncoveredTail = max(0, speechEndSample - lastPreviewWindowEnd)

        if !hypothesisWords.isEmpty, uncoveredTail < Int(0.3 * 16000) {
            // Fresh hypothesis covers the utterance — instant commit. Let the
            // live box finish revealing first so the line doesn't get yanked
            // away mid-typing.
            let utterance = Array(allSamples[windowStartSample..<max(windowStartSample, speechEndSample)])
            let speakerLabel = await identifySpeaker(in: utterance)
            await finishLiveReveal()
            let normalizer = ATCNormalizer(config: atcConfig)
            let line = normalizer.normalize(hypothesisWords.joined(separator: " "))
            if !line.isEmpty {
                appendTranscript(line, speaker: speakerLabel)
            }
            if benchmarkActive { benchRawHyps.append(hypothesisWords.joined(separator: " ")) }
            windowStartSample = max(0, allSamples.count - keepOverlapSamples)
            resetLive()
        } else {
            await commitCurrentWindow()
        }
    }

    private func commitCurrentWindow() async {
        guard let whisperContext else { return }
        let allSamples = currentSamples()
        let windowSamples = Array(allSamples[windowStartSample...])
        guard !windowSamples.isEmpty else { return }

        isTranscribing = true
        let t0 = Date()
        await whisperContext.fullTranscribe(
            samples: windowSamples,
            initialPrompt: whisperInitialPrompt,
            carryInitialPrompt: false,
            useVAD: false   // utterance is already RMS-segmented; Silero costs 50-800ms here
        )
        let elapsed = Date().timeIntervalSince(t0)
        let audioSec = Double(windowSamples.count) / 16000.0
        lastLatencyMs = elapsed * 1000
        lastRTF = audioSec > 0 ? elapsed / audioSec : 0

        let segments = await whisperContext.getSegments()
        let normalizer = ATCNormalizer(config: atcConfig)
        let lines = segments.map { normalizer.normalize($0) }.filter { !$0.isEmpty }
        if benchmarkActive { benchRawHyps.append(await whisperContext.getTranscription()) }
        // One window = one PTT transmission = one speaker, even if whisper
        // split it into several segments.
        let speakerLabel = await identifySpeaker(in: windowSamples)
        // Same hand-off as the instant path: let the live box finish typing
        // before the committed line replaces it.
        await finishLiveReveal()
        let now = Date()
        for line in lines { appendTranscript(line, at: now, speaker: speakerLabel) }
        windowStartSample = max(0, allSamples.count - keepOverlapSamples)
        resetLive()
        isTranscribing = false
    }

    private func finalTranscribe() async {
        guard let whisperContext else {
            resetLive()
            return
        }
        let allSamples = currentSamples()
        guard !allSamples.isEmpty else {
            resetLive()
            return
        }

        if windowStartSample < allSamples.count {
            let remaining = Array(allSamples[windowStartSample...])
            await whisperContext.fullTranscribe(
                samples: remaining,
                initialPrompt: whisperInitialPrompt,
                carryInitialPrompt: false
            )
            let segments = await whisperContext.getSegments()
            let normalizer = ATCNormalizer(config: atcConfig)
            let lines = segments
                .map { normalizer.normalize($0) }
                .filter { !$0.isEmpty }
            let speakerLabel = await identifySpeaker(in: remaining)
            await finishLiveReveal()
            let now = Date()
            for line in lines {
                appendTranscript(line, at: now, speaker: speakerLabel)
            }
        }

        resetLive()
        windowStartSample = 0
    }

    // MARK: - Playback

    private func startPlayback(_ url: URL) throws {
#if !os(macOS)
        // The session is set to .record elsewhere; AVAudioPlayer is silent in
        // that category. Switch to playback (speaker) for clip audition.
        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.playback, mode: .default)
        try? session.setActive(true)
#endif
        audioPlayer = try AVAudioPlayer(contentsOf: url)
        audioPlayer?.play()
    }

    private func stopPlayback() {
        audioPlayer?.stop()
        audioPlayer = nil
    }

    // MARK: - Permissions

    private func requestRecordPermission(response: @escaping (Bool) -> Void) {
#if os(macOS)
        response(true)
#else
        AVAudioSession.sharedInstance().requestRecordPermission { granted in
            response(granted)
        }
#endif
    }

    // MARK: - ATC Config

    private func loadATCConfig() {
        Task {
            do {
                atcConfig = try await ATCConfigManager.shared.loadConfig()
                atcConfigVersion = atcConfig?.version ?? "unknown"
            } catch {
                statusText = "Config load failed: \(error.localizedDescription)"
            }
        }
    }

    func updateATCConfig(from urlString: String) async {
        let trimmed = urlString.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let url = URL(string: trimmed) else {
            statusText = "Invalid config URL"
            return
        }
        isUpdatingConfig = true
        do {
            let config = try await ATCConfigManager.shared.downloadConfig(from: url)
            atcConfig = config
            atcConfigVersion = config.version
            statusText = "Config updated to v\(config.version)"
        } catch {
            statusText = "Config download failed: \(error.localizedDescription)"
        }
        isUpdatingConfig = false
    }

    func resetATCConfig() async {
        do {
            try await ATCConfigManager.shared.clearCache()
            atcConfig = try await ATCConfigManager.shared.loadConfig()
            atcConfigVersion = atcConfig?.version ?? "unknown"
            statusText = "Config reset to v\(atcConfigVersion)"
        } catch {
            statusText = "Config reset failed: \(error.localizedDescription)"
        }
    }
}

fileprivate func cpuCount() -> Int {
    ProcessInfo.processInfo.processorCount
}
