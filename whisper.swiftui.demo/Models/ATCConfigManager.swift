//
//  ATCConfigManager.swift
//  whisper.swiftui.demo
//
//  ATC 配置管理器
//  支持从 Bundle / 本地缓存 / 网络远程加载配置
//

import Foundation

// MARK: - Error Types

enum ATCConfigError: Error, LocalizedError {
    case bundleConfigNotFound
    case downloadFailed(statusCode: Int)
    case invalidConfigData(underlying: Error)

    var errorDescription: String? {
        switch self {
        case .bundleConfigNotFound:
            return "ATC config file not found in app bundle"
        case .downloadFailed(let code):
            return "ATC config download failed (HTTP \(code))"
        case .invalidConfigData(let err):
            return "Invalid ATC config: \(err.localizedDescription)"
        }
    }
}

// MARK: - Config Data Model

struct ATCConfig: Codable {
    let version: String
    let updatedAt: String?
    let airlines: [AirlineEntry]
    let airports: [String]?
    let waypoints: [String]?
    let customTerminology: [TermReplacement]?
    let customNoiseWords: [String]?
    let correctionJoins: [TermReplacement]?
    let correctionReplacements: [String: String]?
    /// Whisper `initial_prompt`: bias decoding toward ATC phraseology (English).
    let whisperInitialPrompt: String?

    struct AirlineEntry: Codable {
        let icao: String
        let callsign: String
        let spoken: [String]
    }

    struct TermReplacement: Codable {
        let from: [String]
        let to: String
    }

    enum CodingKeys: String, CodingKey {
        case version
        case updatedAt = "updated_at"
        case airlines
        case airports
        case waypoints
        case customTerminology = "custom_terminology"
        case customNoiseWords = "custom_noise_words"
        case correctionJoins = "correction_joins"
        case correctionReplacements = "correction_replacements"
        case whisperInitialPrompt = "whisper_initial_prompt"
    }
}

extension ATCConfig {
    /// Default when JSON omits `whisper_initial_prompt` or it is blank.
    static let defaultWhisperInitialPrompt = """
    ATC radio ICAO phraseology. numbers to digits. altitude numeric (3000). flight level FLxxx. heading 3 digits. frequency xxx.x. callsign airline + digits. words: cleared, descend, climb, maintain, contact, squawk, hold short, line up and wait.
    """

    var effectiveWhisperInitialPrompt: String {
        if let p = whisperInitialPrompt {
            let t = p.trimmingCharacters(in: .whitespacesAndNewlines)
            if !t.isEmpty { return t }
        }
        return Self.defaultWhisperInitialPrompt
    }
}

// MARK: - Config Manager

actor ATCConfigManager {
    static let shared = ATCConfigManager()

    private var cachedConfig: ATCConfig?

    private var localConfigURL: URL {
        FileManager.default
            .urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("atc_config.json")
    }

    /// 加载配置（优先级：本地缓存 > Bundle 默认）
    func loadConfig() throws -> ATCConfig {
        if let cached = cachedConfig { return cached }

        if FileManager.default.fileExists(atPath: localConfigURL.path) {
            if let local = try? decodeConfig(from: localConfigURL) {
                cachedConfig = local
                return local
            }
        }

        let config = try loadFromBundle()
        cachedConfig = config
        return config
    }

    /// 从远程 URL 下载配置，验证后缓存到本地
    @discardableResult
    func downloadConfig(from url: URL) async throws -> ATCConfig {
        let (data, response) = try await URLSession.shared.data(from: url)
        guard let http = response as? HTTPURLResponse else {
            throw ATCConfigError.downloadFailed(statusCode: -1)
        }
        guard http.statusCode == 200 else {
            throw ATCConfigError.downloadFailed(statusCode: http.statusCode)
        }

        let config: ATCConfig
        do {
            config = try JSONDecoder().decode(ATCConfig.self, from: data)
        } catch {
            throw ATCConfigError.invalidConfigData(underlying: error)
        }

        try data.write(to: localConfigURL, options: .atomic)
        cachedConfig = config
        return config
    }

    /// 获取当前缓存的配置版本号
    func currentVersion() -> String? {
        cachedConfig?.version
    }

    /// 强制从磁盘重新加载
    func reloadConfig() throws -> ATCConfig {
        cachedConfig = nil
        return try loadConfig()
    }

    /// 清除缓存及本地配置文件
    func clearCache() throws {
        cachedConfig = nil
        if FileManager.default.fileExists(atPath: localConfigURL.path) {
            try FileManager.default.removeItem(at: localConfigURL)
        }
    }

    // MARK: - Private

    private func loadFromBundle() throws -> ATCConfig {
        guard let url = Bundle.main.url(forResource: "atc_config", withExtension: "json") else {
            throw ATCConfigError.bundleConfigNotFound
        }
        return try decodeConfig(from: url)
    }

    private func decodeConfig(from url: URL) throws -> ATCConfig {
        let data = try Data(contentsOf: url)
        do {
            return try JSONDecoder().decode(ATCConfig.self, from: data)
        } catch {
            throw ATCConfigError.invalidConfigData(underlying: error)
        }
    }
}
