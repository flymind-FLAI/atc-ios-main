import SwiftUI
import AVFoundation
import Foundation

struct ContentView: View {
    @StateObject var whisperState = WhisperState()
    // "system" (default, follows the device), "light", or "dark".
    @AppStorage("appearancePref") private var appearancePref = "system"

    var body: some View {
        NavigationStack {
            VStack(spacing: 10) {
                TopStatusBar(whisperState: whisperState)
                StatusLine(whisperState: whisperState)

                KeyInstructionStrip(whisperState: whisperState)

                LiveBox(whisperState: whisperState)

                TranscriptList(whisperState: whisperState)

                BottomActionBar(whisperState: whisperState)
            }
            .padding(.horizontal, 12)
            .navigationTitle("ATC Copilot")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    if !whisperState.entries.isEmpty {
                        Button {
                            UIPasteboard.general.string = whisperState.entries
                                .map { "[\($0.timeString)] \($0.text)" }
                                .joined(separator: "\n")
                        } label: {
                            Image(systemName: "doc.on.doc")
                        }
                        .accessibilityLabel("Copy transcript")
                    }
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    NavigationLink(destination: ModelsView(whisperState: whisperState)) {
                        Image(systemName: "gearshape")
                    }
                    .accessibilityLabel("Settings")
                }
            }
        }
        .preferredColorScheme(colorScheme)
    }

    private var colorScheme: ColorScheme? {
        switch appearancePref {
        case "light": return .light
        case "dark":  return .dark
        default:      return nil   // follow the system
        }
    }
}

// MARK: - Top Status Bar

private struct TopStatusBar: View {
    @ObservedObject var whisperState: WhisperState

    var body: some View {
        HStack(spacing: 8) {
            // Status chip — reads like an instrument annunciator.
            HStack(spacing: 6) {
                Circle()
                    .fill(vadColor)
                    .frame(width: 7, height: 7)
                Text(vadLabel)
                    .font(.caption.weight(.bold))
                    .tracking(0.8)
                    .foregroundColor(vadColor)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(vadColor.opacity(0.12))
            .clipShape(Capsule())

            Spacer()

            Text(modelLabel)
                .font(.system(.caption, design: .monospaced))
                .foregroundColor(modelIsError ? .orange : .secondary)
                .lineLimit(1)
        }
        .padding(.horizontal, 2)
    }

    private var vadColor: Color {
        if !whisperState.isRecording { return .secondary }
        return whisperState.vadSpeaking ? .green : .red
    }
    private var vadLabel: String {
        if !whisperState.isRecording { return "STANDBY" }
        return whisperState.vadSpeaking ? "RX ACTIVE" : "MONITORING"
    }
    /// While the model is still loading, say so instead of a cold "No model".
    private var modelLabel: String {
        if !whisperState.modelName.isEmpty { return whisperState.modelName }
        if !whisperState.canTranscribe && !whisperState.statusText.isEmpty {
            return whisperState.statusText   // "Loading model…"
        }
        return "No model"
    }
    /// Distinguish a real failure from a normal transient state, by color.
    private var modelIsError: Bool {
        guard whisperState.modelName.isEmpty else { return false }
        let s = whisperState.statusText.lowercased()
        return s.contains("not found") || s.contains("fail") || s.contains("error")
    }
}

// MARK: - Key Instruction Strip

/// "Currently assigned" chips — one slot per safety-critical category, each
/// holding the latest value heard (RWY 27L · ↓ 3,000 · HDG 270 · 118.1 …).
/// Hidden until something has been committed.
private struct KeyInstructionStrip: View {
    @ObservedObject var whisperState: WhisperState

    var body: some View {
        if !whisperState.keyInstructions.isEmpty {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(ATCKeyState.Kind.allCases.sorted(), id: \.self) { kind in
                        if let label = whisperState.keyInstructions[kind] {
                            Text(label)
                                .font(.system(.caption, design: .monospaced).weight(.bold))
                                .foregroundColor(kind.color)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 5)
                                .background(kind.color.opacity(0.14))
                                .clipShape(Capsule())
                        }
                    }
                }
                .padding(.horizontal, 2)
            }
        }
    }
}

// MARK: - Status Line

/// Surfaces statusText — "Playing 01_atco2…", per-clip RTF/WER results,
/// benchmark summaries. Previously this was set everywhere but never rendered.
private struct StatusLine: View {
    @ObservedObject var whisperState: WhisperState

    var body: some View {
        // When no model is loaded the top bar already mirrors statusText —
        // don't show the same message twice.
        if !whisperState.modelName.isEmpty,
           !whisperState.statusText.isEmpty, whisperState.statusText != "Ready" {
            Text(whisperState.statusText)
                .font(.footnote)
                .foregroundColor(.secondary)
                .lineLimit(2)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 4)
        }
    }
}


// MARK: - Live Box

private struct LiveBox: View {
    @ObservedObject var whisperState: WhisperState
    @State private var textHeight: CGFloat = 0

    /// One line of the monospaced callout font, padded slightly for ascenders.
    private var lineHeight: CGFloat {
        UIFont.preferredFont(forTextStyle: .callout).lineHeight + 1
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Text("LIVE")
                    .font(.caption2.weight(.bold))
                    .tracking(1.2)
                    .foregroundColor(whisperState.isRecording ? .red : .secondary)
                if whisperState.isRecording {
                    Circle().fill(Color.red).frame(width: 5, height: 5)
                }
                Spacer()
            }
            // Grows 1→3 lines with the text; beyond that it scrolls inside,
            // auto-following the newest words. Keeps the layout below stable.
            ScrollViewReader { proxy in
                ScrollView(.vertical, showsIndicators: false) {
                    liveBody
                        .font(.system(.callout, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(GeometryReader { g in
                            Color.clear.preference(key: LiveTextHeightKey.self, value: g.size.height)
                        })
                        .id("liveEnd")
                }
                .frame(height: min(max(textHeight, lineHeight), lineHeight * 3))
                .onPreferenceChange(LiveTextHeightKey.self) { textHeight = $0 }
                .onChange(of: whisperState.liveTentative) { _ in
                    proxy.scrollTo("liveEnd", anchor: .bottom)
                }
                .onChange(of: whisperState.liveConfirmed) { _ in
                    proxy.scrollTo("liveEnd", anchor: .bottom)
                }
            }
        }
        .padding(12)
        .background(Color(.systemGray6))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(whisperState.isRecording ? Color.red.opacity(0.35)
                                                 : Color.secondary.opacity(0.15),
                        lineWidth: 1)
        )
    }

    /// Two-tier live text (ported from the Mac app): confirmed words in solid
    /// accent — they never regress — followed by the still-changing tentative
    /// tail in gray italic. Same cadence, but reads as steady growth.
    private var liveBody: Text {
        let confirmed = whisperState.liveConfirmed
        let tentative = whisperState.liveTentative
        if confirmed.isEmpty && tentative.isEmpty {
            return Text("—").foregroundColor(.secondary.opacity(0.5))
        }
        var t = Text(confirmed).foregroundColor(.accentColor)
        if !tentative.isEmpty {
            t = t + Text(confirmed.isEmpty ? tentative : " " + tentative)
                .foregroundColor(.secondary)
                .italic()
        }
        return t
    }
}

private struct LiveTextHeightKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}

// MARK: - Transcript List

private struct TranscriptList: View {
    @ObservedObject var whisperState: WhisperState

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 0) {
                    Color.clear.frame(height: 1).id("top")
                    // Newest first: a fresh transmission lands right under the
                    // live box, so the reader's eye never leaves the top.
                    ForEach(whisperState.entries.reversed()) { entry in
                        TranscriptRow(entry: entry).id(entry.id)
                        Divider()
                            .overlay(Color.secondary.opacity(0.1))
                            .padding(.leading, 12)
                    }
                }
                .padding(.vertical, 4)
            }
            // systemGray6 contrasts with systemBackground in BOTH modes
            // (light: #F2F2F7 vs white; dark: #1C1C1E vs black).
            .background(Color(.systemGray6))
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .overlay {
                if whisperState.entries.isEmpty {
                    VStack(spacing: 10) {
                        Image(systemName: "text.bubble")
                            .font(.largeTitle)
                            .foregroundColor(.secondary.opacity(0.35))
                        Text("Committed transmissions will appear here.\nPress Start to begin listening.")
                            .font(.footnote)
                            .multilineTextAlignment(.center)
                            .foregroundColor(.secondary.opacity(0.6))
                    }
                    .allowsHitTesting(false)
                }
            }
            .onChange(of: whisperState.entries.count) { _ in
                withAnimation { proxy.scrollTo("top") }
            }
        }
    }
}

private struct TranscriptRow: View {
    let entry: TranscriptEntry

    var body: some View {
        // Log-style inline prefix: the timestamp doesn't reserve a column, so
        // wrapped lines use the full row width.
        (Text(entry.timeString + "  ")
            .font(.system(.caption2, design: .monospaced))
            .foregroundColor(.secondary)
         + Text(entry.display)
            .font(.system(.callout, design: .monospaced))
            .foregroundColor(.primary))
            .frame(maxWidth: .infinity, alignment: .leading)
            .textSelection(.enabled)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
        .contextMenu {
            Button {
                UIPasteboard.general.string = "[\(entry.timeString)] \(entry.text)"
            } label: {
                Label("Copy line", systemImage: "doc.on.doc")
            }
        }
    }
}

// MARK: - Bottom Action Bar

private struct BottomActionBar: View {
    @ObservedObject var whisperState: WhisperState
    @State private var showClearConfirm = false

    var body: some View {
        HStack(spacing: 14) {
            Button {
                whisperState.toggleRecord()
            } label: {
                HStack(spacing: 10) {
                    Image(systemName: whisperState.isRecording ? "stop.fill" : "play.fill")
                    Text(whisperState.isRecording ? "Stop" : "Start")
                }
                .font(.headline)
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 16)
                .background(startBackground)
                .clipShape(RoundedRectangle(cornerRadius: 12))
            }
            .buttonStyle(.plain)
            .disabled(!whisperState.canTranscribe)
            .accessibilityLabel(whisperState.isRecording ? "Stop listening" : "Start listening")

            Button {
                showClearConfirm = true
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: "trash")
                    Text("Clear")
                }
                .font(.headline)
                .foregroundColor(.primary)
                .frame(width: 112)
                .padding(.vertical, 16)
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.secondary.opacity(0.35), lineWidth: 1)
                )
            }
            .buttonStyle(.plain)
            .disabled(whisperState.entries.isEmpty)
            .opacity(whisperState.entries.isEmpty ? 0.4 : 1.0)
            .confirmationDialog("Clear all transcripts?",
                                isPresented: $showClearConfirm,
                                titleVisibility: .visible) {
                Button("Clear \(whisperState.entries.count) lines", role: .destructive) {
                    withAnimation { whisperState.clearTranscript() }
                }
                Button("Cancel", role: .cancel) {}
            }
        }
        // Top gap only — the bottom rests on the home-indicator safe area,
        // which already provides the visual breathing room.
        .padding(.top, 10)
    }

    /// Gray when the model isn't ready; accent when armed; red while recording.
    private var startBackground: Color {
        if !whisperState.canTranscribe { return Color(.systemGray3) }
        return whisperState.isRecording ? .red : .accentColor
    }
}

// MARK: - Models / Settings View

extension ContentView {
    struct ModelsView: View {
        @ObservedObject var whisperState: WhisperState
        @Environment(\.dismiss) var dismiss
        @State private var configURL = ""
        @FocusState private var isURLFieldFocused: Bool
        @AppStorage("appearancePref") private var appearancePref = "system"

        // The ATC model ships in the bundle and auto-loads on device — no
        // downloads needed there. The simulator can't load the 1.1 GB model
        // (OOM), so it keeps small downloadable models for functional testing.
        private static let models: [Model] = {
#if targetEnvironment(simulator)
            [
                Model(name: "tiny", info: "(simulator testing, 75 MiB)",  url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin", filename: "tiny.bin"),
                Model(name: "base", info: "(simulator testing, 142 MiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin", filename: "base.bin"),
            ]
#else
            []
#endif
        }()

        static func getDownloadedModels() -> [Model] {
            models.filter { FileManager.default.fileExists(atPath: $0.fileURL.path()) }
        }

        func loadModel(model: Model) {
            Task {
                dismiss()
                whisperState.loadModel(path: model.fileURL)
            }
        }

        var body: some View {
            List {
                if !Self.models.isEmpty {
                    Section(header: Text("Models")) {
                        ForEach(Self.models) { model in
                            DownloadButton(model: model)
                                .onLoad(perform: loadModel)
                        }
                    }
                }

                Section(header: Text("Appearance")) {
                    Picker("Theme", selection: $appearancePref) {
                        Text("System").tag("system")
                        Text("Light").tag("light")
                        Text("Dark").tag("dark")
                    }
                    .pickerStyle(.segmented)
                }

                Section(header: Text("Audio Input")) {
                    Picker("Input", selection: $whisperState.selectedInputUID) {
                        ForEach(whisperState.availableInputs) { opt in
                            Label("\(opt.kind.label) · \(opt.name)", systemImage: opt.kind.icon)
                                .tag(Optional(opt.id))
                        }
                    }
                    .disabled(whisperState.isRecording)

                    Button {
                        whisperState.refreshInputs()
                    } label: {
                        Label("Refresh inputs", systemImage: "arrow.clockwise")
                    }
                    .disabled(whisperState.isRecording)
                }
                .onAppear { whisperState.refreshInputs() }

                Section(header: Text("Recording"),
                        footer: Text("How long a pause must last before the current transmission is committed to the transcript. Shorter = snappier commits; longer = fewer split sentences.")) {
                    HStack(spacing: 12) {
                        Text("Silence")
                        Slider(value: $whisperState.silenceCutoffSec, in: 0.2...2.0, step: 0.1)
                        Text(String(format: "%.1fs", whisperState.silenceCutoffSec))
                            .font(.system(.subheadline, design: .monospaced))
                            .foregroundColor(.secondary)
                            .frame(width: 44, alignment: .trailing)
                    }
                }

                Section(header: Text("Benchmark"),
                        footer: Text("Plays 10 random clips from the \(whisperState.bundledTestClips.count) bundled atco2_test tower recordings and scores them against reference transcripts. Reports average WER and real-time factor.")) {
                    Button {
                        dismiss()
                        Task { await whisperState.transcribeAllClips() }
                    } label: {
                        Label("Run benchmark", systemImage: "gauge.with.needle")
                    }
                    .disabled(!whisperState.canTranscribe || whisperState.bundledTestClips.isEmpty)
                }

                Section(header: Text("ATC Config")) {
                    HStack {
                        Text("Version")
                        Spacer()
                        Text(whisperState.atcConfigVersion)
                            .foregroundColor(.secondary)
                    }

                    TextField("Config URL", text: $configURL)
                        .textContentType(.URL)
                        .keyboardType(.URL)
                        .autocapitalization(.none)
                        .disableAutocorrection(true)
                        .focused($isURLFieldFocused)
                        .submitLabel(.done)
                        .onSubmit { isURLFieldFocused = false }

                    HStack(spacing: 12) {
                        let isUpdateDisabled = configURL.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                            || whisperState.isUpdatingConfig

                        Button {
                            isURLFieldFocused = false
                            Task { await whisperState.updateATCConfig(from: configURL) }
                        } label: {
                            HStack(spacing: 6) {
                                if whisperState.isUpdatingConfig { ProgressView().tint(.white) }
                                Text(whisperState.isUpdatingConfig ? "Updating…" : "Update")
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                            .background(isUpdateDisabled ? Color(.systemGray4) : Color.accentColor)
                            .foregroundColor(isUpdateDisabled ? Color(.systemGray) : .white)
                            .cornerRadius(8)
                        }
                        .buttonStyle(.plain)
                        .disabled(isUpdateDisabled)

                        Button {
                            Task { await whisperState.resetATCConfig() }
                        } label: {
                            Text("Reset")
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 10)
                                .background(Color(.systemGray5))
                                .foregroundColor(.red)
                                .cornerRadius(8)
                        }
                        .buttonStyle(.plain)
                    }
                    .listRowInsets(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))
                }
            }
            .listStyle(GroupedListStyle())
            .scrollDismissesKeyboard(.interactively)
            .background(KeyboardDismissView())
            .navigationBarTitle("Settings", displayMode: .inline)
            .toolbar {}
        }
    }
}

// MARK: - Keyboard Dismiss

private struct KeyboardDismissView: UIViewRepresentable {
    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        view.backgroundColor = .clear
        let tap = UITapGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.dismiss))
        tap.cancelsTouchesInView = false
        view.addGestureRecognizer(tap)
        return view
    }
    func updateUIView(_ uiView: UIView, context: Context) {}
    func makeCoordinator() -> Coordinator { Coordinator() }
    class Coordinator {
        @objc func dismiss() {
            UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
        }
    }
}
