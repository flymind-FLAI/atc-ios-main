//
//  ATCNormalizer.swift
//  whisper.swiftui.demo
//
//  Ported from atc-python atc_app.py: number-word spans are REPLACED with
//  digits ("two seven left" -> "27 left"), phonetic letters collapse to
//  letter runs ("alfa oscar" -> "AO"), plus ICAO/ATC capitalization. Pre-processing (spell/join/fuzzy corrections,
//  multi-word airline → ICAO, noise removal, terminology replacement) stays.
//

import Foundation
import SwiftUI

final class ATCNormalizer {

    // MARK: - Constants (mirrors atc_app.py)

    private static let digitMap: [String: Int] = [
        "zero": 0, "oh": 0,
        "one": 1, "two": 2,
        "tree": 3, "three": 3,
        "fower": 4, "four": 4,
        "fife": 5, "five": 5,
        "six": 6, "seven": 7,
        "ait": 8, "eight": 8,
        "niner": 9, "nine": 9,
    ]

    static let phoneticAlphabet: [String: String] = [
        "alfa": "A", "alpha": "A", "bravo": "B", "charlie": "C",
        "echo": "E", "foxtrot": "F", "golf": "G", "hotel": "H",
        "india": "I", "juliet": "J", "juliett": "J", "kilo": "K",
        "lima": "L", "mike": "M", "november": "N", "oscar": "O",
        "papa": "P", "quebec": "Q", "romeo": "R", "sierra": "S",
        "tango": "T", "uniform": "U", "victor": "V", "whiskey": "W",
        "xray": "X", "x-ray": "X", "yankee": "Y", "zulu": "Z",
    ]

    /// Title-cased ICAO phonetic words + waypoint-like names.
    private static let icaoTitle: Set<String> = [
        "alpha","bravo","charlie","delta","echo","foxtrot","golf","hotel",
        "india","juliet","kilo","lima","mike","november","oscar","papa",
        "quebec","romeo","sierra","tango","uniform","victor","whiskey",
        "xray","yankee","zulu",
    ]

    /// Uppercased ATC acronyms.
    private static let atcCaps: Set<String> = [
        "ils","rnp","qnh","atc","ifr","vfr","ctaf","atis","fl","luaw",
    ]

    /// Title-cased ATC/airline/place names.
    private static let atcTitle: Set<String> = [
        "ruzyne","praha","radar","tower","approach","baltu","vlasim",
        "lanux","benesov","liege","wien","warsaw","speedbird","ryanair",
        "lufthansa","eurowings","csa","belavia","skytravel","klm",
    ]

    private static let multipliers: [String: Int] = ["hundred": 100, "thousand": 1000]
    private static let decimalWords: Set<String> = ["decimal", "point"]

    /// "ten"…"nineteen" → direct values.
    private static let teensMap: [String: Int] = [
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    ]
    /// "twenty"…"ninety" → direct values.
    private static let tensMap: [String: Int] = [
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
        "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    ]

    // MARK: - Default correction tables

    private static let defaultNoiseWords: Set<String> = ["uh", "um", "ah", "er", "like"]

    private static let defaultTerminologyReplacements: [([String], String)] = [
        (["climb", "to", "and", "maintain"], "climb maintain"),
        (["descend", "and", "maintain"], "descend maintain"),
        (["climb", "and", "maintain"], "climb maintain"),
        (["line", "up", "and", "wait"], "LUAW"),
        (["hold", "short", "of"], "hold short"),
        (["flight", "level"], "FL"),
        (["hold", "short"], "hold short"),
        (["go", "around"], "go-around"),
        (["read", "back"], "readback"),
        (["say", "again"], "say again"),
        (["taxi", "via"], "taxi via"),
        (["taxi", "to"], "taxi to"),
    ]

    private static let defaultWordJoinCorrections: [([String], String)] = [
        (["al", "ti", "meter"], "altimeter"),
        (["all", "to", "meter"], "altimeter"),
        (["all", "timer"], "altimeter"),
        (["old", "timer"], "altimeter"),
        (["run", "way"], "runway"),
        (["taxi", "way"], "taxiway"),
        (["main", "tain"], "maintain"),
        (["main", "tane"], "maintain"),
        (["fox", "trot"], "foxtrot"),
        (["will", "co"], "wilco"),
        (["no", "vember"], "november"),
        (["head", "in"], "heading"),
    ]

    private static let defaultSpellCorrections: [String: String] = [
        "descent": "descend", "dissent": "descend", "descents": "descend",
        "squash": "squawk", "squat": "squawk", "squaw": "squawk",
        "rodger": "roger", "rajah": "roger",
        "runaway": "runway",
        "negatory": "negative", "affirmativ": "affirmative",
        "charley": "charlie", "whisky": "whiskey",
        "hedding": "heading", "juliette": "juliet",
    ]

    private static let contextDigitCorrections: [String: String] = [
        "minor": "niner", "miner": "niner", "liner": "niner",
        "to": "two", "too": "two", "tue": "two",
        "for": "four", "fore": "four",
        "won": "one", "wan": "one",
        "ate": "eight", "aid": "eight",
        "free": "three", "sicks": "six",
    ]

    private static let fuzzyMatchKeywords: Set<String> = [
        "runway", "taxiway", "heading", "squawk", "altimeter",
        "maintain", "descend", "climb", "cleared", "contact",
        "approach", "departure", "tower", "ground", "center",
        "roger", "wilco", "affirm", "affirmative", "negative",
        "thousand", "hundred", "knots", "gusting", "holding",
        "direct", "proceed", "report", "expect", "altitude",
        "ceiling", "visibility", "frequency", "ident",
    ]

    // MARK: - Config-derived tables

    private let airlineSpokenMap: [String: String]        // single-word spoken → ICAO
    private let airlineICAOSet: Set<String>
    private let waypointSet: Set<String>
    private let noiseWords: Set<String>
    private let terminologyReplacements: [([String], String)]
    private let multiWordAirlines: [(words: [String], icao: String)]
    private let wordJoinCorrections: [([String], String)]
    private let spellCorrections: [String: String]

    // MARK: - Init

    init(config: ATCConfig? = nil) {
        var spokenMap: [String: String] = [:]
        var icaoSet: Set<String> = []
        var multiWord: [(words: [String], icao: String)] = []

        if let config = config {
            for airline in config.airlines {
                let icao = airline.icao.uppercased()
                icaoSet.insert(icao)
                for spoken in airline.spoken {
                    let words = spoken.lowercased()
                        .split(separator: " ", omittingEmptySubsequences: true)
                        .map(String.init)
                    if words.count > 1 {
                        multiWord.append((words, icao))
                    } else if let single = words.first {
                        spokenMap[single] = icao
                    }
                }
            }
        }

        self.airlineSpokenMap = spokenMap
        self.airlineICAOSet = icaoSet
        self.multiWordAirlines = multiWord.sorted { $0.words.count > $1.words.count }

        var waypoints: Set<String> = []
        if let config = config {
            for wp in config.airports ?? [] { waypoints.insert(wp.uppercased()) }
            for wp in config.waypoints ?? [] { waypoints.insert(wp.uppercased()) }
        }
        self.waypointSet = waypoints

        var noise = Self.defaultNoiseWords
        if let extra = config?.customNoiseWords {
            for w in extra { noise.insert(w.lowercased()) }
        }
        self.noiseWords = noise

        var terms = Self.defaultTerminologyReplacements
        if let extra = config?.customTerminology {
            for t in extra { terms.append((t.from, t.to)) }
        }
        self.terminologyReplacements = terms.sorted { $0.0.count > $1.0.count }

        var joins: [([String], String)] = Self.defaultWordJoinCorrections
        if let configJoins = config?.correctionJoins {
            for j in configJoins { joins.append((j.from, j.to)) }
        }
        self.wordJoinCorrections = joins.sorted { $0.0.count > $1.0.count }

        var spells = Self.defaultSpellCorrections
        if let configSpells = config?.correctionReplacements {
            for (k, v) in configSpells { spells[k] = v }
        }
        self.spellCorrections = spells
    }

    // MARK: - Main entry

    /// - Parameter filterHallucination: disable for live partial hypotheses —
    ///   a half-sentence can trip the repetition heuristic and blank the display.
    func normalize(_ text: String, filterHallucination: Bool = true) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }
        if filterHallucination, Self.isHallucination(trimmed) { return "" }

        var words = splitWords(trimmed)
        words = applyJoinCorrections(words)
        words = applyPerWordCorrections(words)                       // spell + fuzzy
        words = applyMultiWordAirlineSubstitution(words)              // "singapore airlines" → SIA
        words = applySingleWordAirlineSubstitution(words)             // "speedbird" → BAW
        words = applyContextDigitCorrections(words)                   // "to"→"two" etc.
        words = removeNoiseWords(words)
        words = applyTerminologyReplacements(words)
        words = applyWaypointUppercase(words)
        words = applyPhoneticLetterSubstitution(words)

        let joined = words.joined(separator: " ")
        let cased = normalizeCase(joined)
        return mergeStandIdentifiers(normalizeNumbers(cased))
    }

    // MARK: - Hallucination filter (atc_app._is_hallucination)

    static func isHallucination(_ text: String) -> Bool {
        let words = text.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
        if words.count > 4 {
            let uniq = Set(words.map { $0.lowercased() })
            if Double(uniq.count) / Double(words.count) < 0.35 { return true }
        }
        if words.contains(where: { $0.count > 25 }) { return true }
        for w in words {
            let lw = w.lowercased()
            guard lw.count >= 12 else { continue }
            for k in 2...6 {
                guard lw.count >= k * 4 else { continue }
                let unit = String(lw.prefix(k))
                let times = lw.count / k
                let repeated = String(repeating: unit, count: times)
                if lw.hasPrefix(repeated) { return true }
            }
        }
        return false
    }

    // MARK: - Confidence

    func confidenceScore(original: String, normalized: String) -> Float {
        let o = original.trimmingCharacters(in: .whitespacesAndNewlines)
        let n = normalized.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !o.isEmpty else { return 1.0 }
        let lenDiff = abs(o.count - n.count)
        let maxLen = max(o.count, n.count, 1)
        let lengthPenalty = Float(lenDiff) / Float(maxLen)
        let commonPrefix = zip(o, n).prefix(while: { $0.0 == $0.1 }).count
        let commonSuffix = zip(o.reversed(), n.reversed()).prefix(while: { $0.0 == $0.1 }).count
        let similarity = Float(commonPrefix + commonSuffix) / Float(maxLen * 2)
        let score = 1.0 - (lengthPenalty * 0.5) - ((1.0 - similarity) * 0.5)
        return max(0, min(1, score))
    }

    // MARK: - Utility

    /// Splits on whitespace; preserves trailing punctuation on each token (Python-compatible).
    private func splitWords(_ s: String) -> [String] {
        s.split(whereSeparator: { $0.isWhitespace }).map(String.init)
    }

    private func lowerStripped(_ w: String) -> (core: String, trail: String) {
        let lw = w.lowercased()
        var coreEnd = lw.endIndex
        let punct: Set<Character> = [".", ",", ";", ":"]
        while coreEnd > lw.startIndex {
            let prev = lw.index(before: coreEnd)
            if punct.contains(lw[prev]) { coreEnd = prev } else { break }
        }
        return (String(lw[..<coreEnd]), String(lw[coreEnd...]))
    }

    /// Parking-stand / gate context words after which a number+letter (or
    /// letter+number) pair is one compact identifier, not two tokens.
    static let standContextWords: Set<String> = [
        "stand", "position", "gate", "spot", "bay", "apron", "parking", "ramp"
    ]

    /// Compact alphanumeric stand/gate identifiers that ASR splits apart:
    /// "position 5 F" → "position 5F", "gate 12 A" → "gate 12A", and the
    /// letter-first form "stand A 12" → "stand A12". Only fires within a few
    /// words of a parking-context word, so ordinary phrases like "descend 5
    /// thousand" or "runway 2 7" are left alone.
    func mergeStandIdentifiers(_ text: String) -> String {
        var words = text.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
        var armed = -1                          // words since a context word; -1 = off
        var i = 0
        while i < words.count {
            let core = lowerStripped(words[i]).core
            if Self.standContextWords.contains(core) {
                armed = 0
            } else if armed >= 0 {
                armed += 1
            }
            if armed >= 1, armed <= 3, i + 1 < words.count,
               let merged = mergedStandToken(words[i], words[i + 1]) {
                words[i] = merged
                words.remove(at: i + 1)
                armed = -1
                continue
            }
            i += 1
        }
        return words.joined(separator: " ")
    }

    private func mergedStandToken(_ a: String, _ b: String) -> String? {
        let (ac, at) = lowerStripped(a)
        let (bc, bt) = lowerStripped(b)
        guard at.isEmpty, !ac.isEmpty, !bc.isEmpty else { return nil }
        let aNum = ac.allSatisfy(\.isNumber)
        let bNum = bc.allSatisfy(\.isNumber)
        let aLetter = ac.count == 1 && (ac.first?.isLetter ?? false)
        let bLetter = bc.count == 1 && (bc.first?.isLetter ?? false)
        if aNum, bLetter { return ac + bc.uppercased() + bt }
        if aLetter, bNum { return ac.uppercased() + bc + bt }
        return nil
    }
}

// MARK: - Pre-processing (word list)

private extension ATCNormalizer {

    func applyJoinCorrections(_ words: [String]) -> [String] {
        var result: [String] = []
        var i = 0
        while i < words.count {
            var matched = false
            for (pattern, replacement) in wordJoinCorrections {
                guard i + pattern.count <= words.count else { continue }
                var ok = true
                for (j, p) in pattern.enumerated() where words[i + j].lowercased() != p {
                    ok = false; break
                }
                if ok {
                    result.append(replacement)
                    i += pattern.count
                    matched = true
                    break
                }
            }
            if !matched { result.append(words[i]); i += 1 }
        }
        return result
    }

    func applyPerWordCorrections(_ words: [String]) -> [String] {
        words.map { w in
            let lw = w.lowercased()
            if let c = spellCorrections[lw] { return c }
            if let f = fuzzyCorrectWord(lw) { return f }
            return w
        }
    }

    func fuzzyCorrectWord(_ word: String) -> String? {
        guard word.count >= 5 else { return nil }
        if Self.digitMap[word] != nil { return nil }
        if Self.phoneticAlphabet[word] != nil { return nil }
        if Self.fuzzyMatchKeywords.contains(word) { return nil }

        var best: String?
        var count = 0
        for kw in Self.fuzzyMatchKeywords {
            guard abs(word.count - kw.count) <= 1 else { continue }
            if Self.editDistance(word, kw) == 1 { best = kw; count += 1 }
        }
        return count == 1 ? best : nil
    }

    static func editDistance(_ a: String, _ b: String) -> Int {
        let m = a.count, n = b.count
        if m == 0 { return n }
        if n == 0 { return m }
        let aa = Array(a), bb = Array(b)
        var prev = Array(0...n)
        var curr = Array(repeating: 0, count: n + 1)
        for i in 1...m {
            curr[0] = i
            for j in 1...n {
                let cost = aa[i - 1] == bb[j - 1] ? 0 : 1
                curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
            }
            swap(&prev, &curr)
        }
        return prev[n]
    }

    func applyMultiWordAirlineSubstitution(_ words: [String]) -> [String] {
        var result: [String] = []
        var i = 0
        while i < words.count {
            var matched = false
            for airline in multiWordAirlines {
                let pattern = airline.words
                guard i + pattern.count <= words.count else { continue }
                var ok = true
                for (j, p) in pattern.enumerated() where words[i + j].lowercased() != p {
                    ok = false; break
                }
                if ok {
                    result.append(airline.icao)
                    i += pattern.count
                    matched = true
                    break
                }
            }
            if !matched { result.append(words[i]); i += 1 }
        }
        return result
    }

    func applySingleWordAirlineSubstitution(_ words: [String]) -> [String] {
        words.map { w in airlineSpokenMap[w.lowercased()] ?? w }
    }

    func applyContextDigitCorrections(_ words: [String]) -> [String] {
        var result = words
        let isDigitLike: (String) -> Bool = { w in
            let lw = w.lowercased()
            return Self.digitMap[lw] != nil || Self.multipliers[lw] != nil || Self.decimalWords.contains(lw)
        }
        // Genuine English function words convert only when sandwiched between
        // digit words ("two four TO the right" must stay "to"; "one two for
        // five" was spoken "four"). Pure mishearings (niner/miner/won/ate…)
        // keep the looser one-sided rule.
        let needsBothSides: Set<String> = ["to", "too", "for"]
        for i in 0..<result.count {
            let lw = result[i].lowercased()
            guard let corrected = Self.contextDigitCorrections[lw] else { continue }
            let before = i > 0 && isDigitLike(result[i - 1])
            let after  = i + 1 < result.count && isDigitLike(result[i + 1])
            let convert = needsBothSides.contains(lw) ? (before && after) : (before || after)
            if convert { result[i] = corrected }
        }
        return result
    }

    func removeNoiseWords(_ words: [String]) -> [String] {
        words.filter { !noiseWords.contains($0.lowercased()) }
    }

    func applyTerminologyReplacements(_ words: [String]) -> [String] {
        var result: [String] = []
        var i = 0
        while i < words.count {
            var matched = false
            for (pattern, replacement) in terminologyReplacements {
                guard i + pattern.count <= words.count else { continue }
                var ok = true
                for (j, p) in pattern.enumerated() where words[i + j].lowercased() != p {
                    ok = false; break
                }
                if ok {
                    result.append(replacement)
                    i += pattern.count
                    matched = true
                    break
                }
            }
            if !matched { result.append(words[i]); i += 1 }
        }
        return result
    }

    func applyWaypointUppercase(_ words: [String]) -> [String] {
        words.map { waypointSet.contains($0.uppercased()) ? $0.uppercased() : $0 }
    }

    /// Collapse runs of phonetic words into their letters, joined without spaces.
    /// "alfa oscar" → "AO", "foxtrot alfa oscar" → "FAO", "taxi via alfa charlie" → "taxi via AC".
    /// Trailing punctuation on the last phonetic word is preserved on the letter run.
    func applyPhoneticLetterSubstitution(_ words: [String]) -> [String] {
        var result: [String] = []
        var i = 0
        while i < words.count {
            let (core, _) = lowerStripped(words[i])
            if Self.phoneticAlphabet[core] != nil {
                var letters = ""
                var lastTrail = ""
                while i < words.count {
                    let (c, t) = lowerStripped(words[i])
                    guard let letter = Self.phoneticAlphabet[c] else { break }
                    letters += letter
                    lastTrail = t
                    i += 1
                    if !t.isEmpty { break }
                }
                result.append(letters + lastTrail)
            } else {
                result.append(words[i])
                i += 1
            }
        }
        return result
    }
}

// MARK: - normalize_case (Python parity)

private extension ATCNormalizer {

    func normalizeCase(_ text: String) -> String {
        text.split(separator: " ", omittingEmptySubsequences: true).map { tok -> String in
            let w = String(tok)
            let (core, trail) = lowerStripped(w)
            if Self.icaoTitle.contains(core) || Self.atcTitle.contains(core) {
                return core.prefix(1).uppercased() + core.dropFirst() + trail
            }
            if Self.atcCaps.contains(core) {
                return core.uppercased() + trail
            }
            // preserve pre-uppercased tokens (airline ICAO, terminology like LUAW/FL, waypoints)
            return w
        }.joined(separator: " ")
    }
}

// MARK: - normalize_numbers (Python parity)

private extension ATCNormalizer {

    func normalizeNumbers(_ text: String) -> String {
        let words = text.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
        var result: [String] = []
        var i = 0

        let isNumWord: (String) -> Bool = { c in
            Self.digitMap[c] != nil || Self.teensMap[c] != nil
                || Self.tensMap[c] != nil || Self.multipliers[c] != nil
        }

        while i < words.count {
            let w = words[i]
            let (core, trail) = lowerStripped(w)
            let isNumStart = isNumWord(core) && trail.isEmpty
            let isDecimalStart = Self.decimalWords.contains(core) && trail.isEmpty
                && i + 1 < words.count
                && isNumWord(lowerStripped(words[i + 1]).core)

            if isNumStart || isDecimalStart {
                var spanLows = [core]
                var lastTrail = trail
                i += 1
                while i < words.count && lastTrail.isEmpty {
                    let nw = words[i]
                    let (nc, np) = lowerStripped(nw)
                    if isNumWord(nc) {
                        spanLows.append(nc); lastTrail = np; i += 1
                    } else if Self.decimalWords.contains(nc) && np.isEmpty {
                        if i + 1 < words.count && isNumWord(lowerStripped(words[i + 1]).core) {
                            spanLows.append(nc); i += 1
                        } else { break }
                    } else { break }
                }

                let numstr = spanToNumString(spanLows)
                result.append(numstr + lastTrail)
            } else {
                result.append(w); i += 1
            }
        }
        return result.joined(separator: " ")
    }

    func spanToNumString(_ lows: [String]) -> String {
        if lows.contains(where: { Self.decimalWords.contains($0) }) {
            let decIdx = lows.firstIndex(where: { Self.decimalWords.contains($0) })!
            let intPart = computeNumber(Array(lows[..<decIdx]))
            let fracPart = digitsFromWords(Array(lows[(decIdx + 1)...]))
            return groupThousands(intPart.isEmpty ? "0" : intPart) + "." + fracPart
        }
        let num = computeNumber(lows)
        // VHF frequency heuristic: applies ONLY when the span is pure digit-by-digit
        // words (no teens/tens/hundred/thousand), so altitudes like "three five thousand"
        // (35000) and "one hundred twenty" (120) are never misinterpreted.
        let allDigitWords = !lows.isEmpty && lows.allSatisfy { Self.digitMap[$0] != nil }
        if allDigitWords, let freq = frequencyFromDigits(num) { return freq }
        return groupThousands(num)
    }

    /// If `digits` is a 5- or 6-digit string whose first 3 digits fall in the VHF
    /// aviation band (118–137), insert a decimal point after the 3rd digit.
    /// Returns nil otherwise — caller falls back to plain formatting.
    func frequencyFromDigits(_ digits: String) -> String? {
        guard digits.count == 5 || digits.count == 6,
              digits.allSatisfy({ $0.isNumber }) else { return nil }
        let prefix = Int(digits.prefix(3)) ?? 0
        guard (118...137).contains(prefix) else { return nil }
        let idx = digits.index(digits.startIndex, offsetBy: 3)
        return digits[..<idx] + "." + digits[idx...]
    }

    /// Build the fractional digit string for a decimal span, preserving leading zeros.
    /// Handles digits ("zero zero five" → "005"), teens ("nineteen" → "19"),
    /// and tens+digit ("seventy five" → "75", "twenty" → "20").
    func digitsFromWords(_ lows: [String]) -> String {
        var out = ""
        var i = 0
        while i < lows.count {
            let t = lows[i]
            if let tn = Self.tensMap[t] {
                if i + 1 < lows.count, let d = Self.digitMap[lows[i + 1]] {
                    out += String(tn + d); i += 2
                } else {
                    out += String(tn); i += 1
                }
            } else if let teen = Self.teensMap[t] {
                out += String(teen); i += 1
            } else if let d = Self.digitMap[t] {
                out += String(d); i += 1
            } else { i += 1 }
        }
        return out
    }

    /// Insert commas every 3 digits from the right. "5000" → "5,000", "27" → "27".
    func groupThousands(_ digits: String) -> String {
        guard digits.count > 3, digits.allSatisfy({ $0.isNumber }) else { return digits }
        var out = ""
        for (i, ch) in digits.reversed().enumerated() {
            if i > 0 && i % 3 == 0 { out.append(",") }
            out.append(ch)
        }
        return String(out.reversed())
    }

    /// Evaluate a span of number words into an integer string.
    /// Handles digit concatenation ("two seven" → 27), teens ("nineteen" → 19),
    /// tens+digit ("seventy five" → 75), and multipliers ("five thousand" → 5000).
    func computeNumber(_ lows: [String]) -> String {
        var val = 0
        var cur = 0
        var lastWasTens = false
        var wrote = false
        for t in lows {
            if let d = Self.digitMap[t] {
                if lastWasTens { cur += d } else { cur = cur * 10 + d }
                lastWasTens = false
                wrote = true
            } else if let teen = Self.teensMap[t] {
                cur = cur * 100 + teen
                lastWasTens = false
                wrote = true
            } else if let tn = Self.tensMap[t] {
                cur = cur * 100 + tn
                lastWasTens = true
                wrote = true
            } else if t == "hundred" {
                val += (cur == 0 ? 1 : cur) * 100
                cur = 0; lastWasTens = false; wrote = true
            } else if t == "thousand" {
                val += (cur == 0 ? 1 : cur) * 1000
                cur = 0; lastWasTens = false; wrote = true
            }
        }
        return wrote ? String(val + cur) : ""
    }
}

// MARK: - Safety-critical element highlighting

/// Colors the parts of a normalized transcript line that a pilot must not
/// miss (ICAO mandatory-readback items). Works on ATCNormalizer output, where
/// digit spans are annotated in parentheses: "Runway two seven left (27L)".
enum ATCHighlighter {

    enum Category: Int {
        case runway     // runway refs + takeoff/landing/taxi clearances
        case vertical   // altitude / FL / heading / speed / QNH
        case comms      // frequency changes / squawk

        var color: Color {
            switch self {
            case .runway:   return .orange
            case .vertical: return Color(red: 0.25, green: 0.85, blue: 0.66)   // mint
            case .comms:    return Color(red: 0.38, green: 0.68, blue: 1.00)   // sky
            }
        }
    }

    // Order = priority when matches overlap.
    private static let patterns: [(NSRegularExpression, Category)] = {
        // Normalized text shapes: "runway 27 left", "descend 3,000",
        // "flight level 350", "heading 270", "118.1", "squawk 4,521", "QNH 1,013".
        let NUM = "\\d{1,3}(?:,\\d{3})*"
        let defs: [(String, Category)] = [
            // ── runway-critical ────────────────────────────────────────────
            ("hold short(?: of)?(?: runway)?(?: ?\\d{1,2} ?(?:left|right|center|[LRC])?)?", .runway),
            ("cleared (?:to land|for take ?off|take ?off)(?: runway ?\\d{1,2} ?(?:left|right|center|[LRC])?)?", .runway),
            ("line up and wait|LUAW|go[- ]around", .runway),
            ("runway ?\\d{1,2} ?(?:left|right|center|[LRC])?\\b", .runway),
            // ── vertical / lateral / speed / pressure ──────────────────────
            ("(?:climb(?:ing)?|descend(?:ing)?|maintain(?:ing)?)(?: (?:and|to|at|altitude|maintain)){0,3} ?(?:flight level ?|FL ?)?\(NUM)(?: feet)?", .vertical),
            ("flight level ?\\d{2,3}\\b|FL ?\\d{2,3}\\b", .vertical),
            ("heading ?\\d{1,3}\\b", .vertical),
            ("speed(?: (?:now|below|above|to|at|of)){0,2} ?\(NUM)(?: knots)?|\\b\(NUM) ?knots|mach ?\\.?\\d+", .vertical),
            ("(?:QNH|altimeter)(?: [a-z]+){0,2}? ?\(NUM)(?:\\.\\d+)?", .vertical),
            // ── comms ──────────────────────────────────────────────────────
            ("(?:contact|monitor) [a-z ]{0,15}?\\d{3}\\.\\d{1,4}", .comms),
            ("\\b\\d{3}\\.\\d{1,4}\\b", .comms),
            ("squawk(?: ident)?(?: ?[\\d,]{4,5})?", .comms),
        ]
        return defs.compactMap { p, c in
            (try? NSRegularExpression(pattern: p, options: [.caseInsensitive])).map { ($0, c) }
        }
    }()

    static func highlight(_ text: String) -> AttributedString {
        let ns = text as NSString
        var found: [(NSRange, Category, Int)] = []
        for (idx, (re, cat)) in patterns.enumerated() {
            for m in re.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
                found.append((m.range, cat, idx))
            }
        }
        // earliest start wins; ties → higher-priority pattern; drop overlaps
        found.sort { ($0.0.location, $0.2) < ($1.0.location, $1.2) }
        var kept: [(NSRange, Category)] = []
        var end = 0
        for (r, c, _) in found where r.location >= end {
            kept.append((r, c))
            end = r.location + r.length
        }

        var out = AttributedString()
        var cursor = 0
        for (r, cat) in kept {
            if r.location > cursor {
                out += AttributedString(ns.substring(with: NSRange(location: cursor, length: r.location - cursor)))
            }
            var seg = AttributedString(ns.substring(with: r))
            seg.foregroundColor = cat.color
            seg.font = .system(.callout, design: .monospaced).weight(.bold)
            out += seg
            cursor = r.location + r.length
        }
        if cursor < ns.length {
            out += AttributedString(ns.substring(from: cursor))
        }
        return out
    }
}

// MARK: - Key instruction state ("what's currently assigned to me")

/// Extracts the latest value per safety-critical category from a committed,
/// normalized transcript line. Each category keeps its own "current" value —
/// a new transmission only overwrites the slots it mentions, mirroring the
/// instruction state a pilot maintains mentally.
enum ATCKeyState {

    enum Kind: Int, CaseIterable, Comparable {
        case runway, altitude, heading, speed, frequency, squawk, qnh
        static func < (a: Kind, b: Kind) -> Bool { a.rawValue < b.rawValue }

        var color: Color {
            switch self {
            case .runway:                          return ATCHighlighter.Category.runway.color
            case .altitude, .heading, .speed, .qnh: return ATCHighlighter.Category.vertical.color
            case .frequency, .squawk:              return ATCHighlighter.Category.comms.color
            }
        }
    }

    private static func re(_ p: String) -> NSRegularExpression? {
        try? NSRegularExpression(pattern: p, options: [.caseInsensitive])
    }

    private static let runwayRe   = re("runway ?(\\d{1,2}) ?(left|right|center|[LRC])?\\b")
    private static let altFLRe    = re("(climb(?:ing)?|descend(?:ing)?|maintain(?:ing)?)(?: (?:and|to|at|altitude|maintain)){0,3} ?(?:FL|flight level) ?(\\d{2,3})\\b")
    private static let altFtRe    = re("(climb(?:ing)?|descend(?:ing)?|maintain(?:ing)?)(?: (?:and|to|at|altitude|maintain)){0,3} ?(\\d{1,3}(?:,\\d{3})+|\\d{3,5})(?: feet)?\\b")
    private static let headingRe  = re("heading ?(\\d{1,3})\\b")
    private static let speedRe    = re("speed(?: (?:now|below|above|to|at|of)){0,2} ?(\\d{2,3})\\b|\\b(\\d{2,3}) ?knots\\b")
    private static let freqRe     = re("\\b(\\d{3}\\.\\d{1,4})\\b")
    private static let squawkRe   = re("squawk ?([\\d,]{4,5})\\b")
    private static let qnhRe      = re("(?:QNH|altimeter)(?: [a-z]+){0,2}? ?([\\d,]{3,5}(?:\\.\\d+)?)\\b")

    static func extract(from text: String) -> [Kind: String] {
        let ns = text as NSString
        let all = NSRange(location: 0, length: ns.length)
        func last(_ rx: NSRegularExpression?) -> [String]? {
            guard let rx, let m = rx.matches(in: text, range: all).last else { return nil }
            return (0..<m.numberOfRanges).map {
                m.range(at: $0).location == NSNotFound ? "" : ns.substring(with: m.range(at: $0))
            }
        }

        var out: [Kind: String] = [:]
        if let g = last(runwayRe) {
            let side = g[2].isEmpty ? "" : String(g[2].uppercased().prefix(1))
            out[.runway] = "RWY \(g[1])\(side)"
        }
        if let g = last(altFLRe) {
            out[.altitude] = "\(arrow(g[1])) FL\(g[2])"
        } else if let g = last(altFtRe) {
            out[.altitude] = "\(arrow(g[1])) \(g[2])"
        }
        if let g = last(headingRe)  { out[.heading]   = "HDG \(g[1])" }
        if let g = last(speedRe)    { out[.speed]     = "SPD \(g[1].isEmpty ? g[2] : g[1])" }
        if let g = last(freqRe)     { out[.frequency] = g[1] }
        if let g = last(squawkRe)   { out[.squawk]    = "SQK \(g[1].replacingOccurrences(of: ",", with: ""))" }
        if let g = last(qnhRe)      { out[.qnh]       = "QNH \(g[1].replacingOccurrences(of: ",", with: ""))" }
        return out
    }

    private static func arrow(_ verb: String) -> String {
        let v = verb.lowercased()
        if v.hasPrefix("climb")   { return "↑" }
        if v.hasPrefix("descend") { return "↓" }
        return "＝"
    }
}
