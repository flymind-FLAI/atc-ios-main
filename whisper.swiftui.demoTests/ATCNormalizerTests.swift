//
//  ATCNormalizerTests.swift
//  whisper.swiftui.demoTests
//

import XCTest
@testable import whisper_swiftui_demo

final class ATCNormalizerTests: XCTestCase {

    private var normalizer: ATCNormalizer!

    override func setUp() {
        super.setUp()
        normalizer = ATCNormalizer()
    }

    override func tearDown() {
        normalizer = nil
        super.tearDown()
    }

    // MARK: - 示例用例

    func testExampleFromSpec() {
        let input = "delta two five six descend to flight level three five zero"
        let output = normalizer.normalize(input)
        XCTAssertEqual(output, "DAL256 descend to FL350")
    }

    // MARK: - Callsign 识别

    func testCallsignAAL() {
        let input = "AAL one two three"
        let output = normalizer.normalize(input)
        XCTAssertEqual(output, "AAL123")
    }

    func testCallsignDeltaSpoken() {
        let input = "delta two five six"
        let output = normalizer.normalize(input)
        XCTAssertEqual(output, "DAL256")
    }

    func testCallsignCaseInsensitive() {
        let input = "aal one two three"
        let output = normalizer.normalize(input)
        XCTAssertEqual(output, "AAL123")
    }

    // MARK: - 数字标准化

    func testNumberContinuous() {
        let input = "three five zero"
        let output = normalizer.normalize(input)
        XCTAssertEqual(output, "350")
    }

    func testNumberWithThousand() {
        let input = "one two thousand"
        let output = normalizer.normalize(input)
        XCTAssertEqual(output, "12000")
    }

    func testAviationSpokenVariants() {
        XCTAssertEqual(normalizer.normalize("tree five"), "35")
        XCTAssertEqual(normalizer.normalize("fower six"), "46")
        XCTAssertEqual(normalizer.normalize("fife seven"), "57")
        XCTAssertEqual(normalizer.normalize("niner zero"), "90")
    }

    // MARK: - Flight Level

    func testFlightLevel() {
        let input = "flight level three five zero"
        let output = normalizer.normalize(input)
        XCTAssertEqual(output, "FL350")
    }

    func testFlightLevelTwoZeroZero() {
        let input = "flight level two zero zero"
        let output = normalizer.normalize(input)
        XCTAssertEqual(output, "FL200")
    }

    // MARK: - 高度解析

    func testDescendToAltitude() {
        let input = "descend to one zero thousand"
        let output = normalizer.normalize(input)
        XCTAssertEqual(output, "descend to 10000")
    }

    func testClimbMaintainAltitude() {
        let input = "climb maintain five thousand"
        let output = normalizer.normalize(input)
        XCTAssertEqual(output, "climb maintain 5000")
    }

    // MARK: - 术语规范化

    func testTerminologyFlightLevel() {
        let input = "flight level one five zero"
        let output = normalizer.normalize(input)
        XCTAssertEqual(output, "FL150")
    }

    func testTerminologyDescendAndMaintain() {
        let input = "descend and maintain one zero thousand"
        let output = normalizer.normalize(input)
        XCTAssertEqual(output, "descend maintain 10000")
    }

    // MARK: - 噪音词清理

    func testNoiseWordRemoval() {
        let input = "AAL uh one two three um"
        let output = normalizer.normalize(input)
        XCTAssertEqual(output, "AAL123")
    }

    // MARK: - 置信度评分

    func testConfidenceScoreUnchanged() {
        let text = "hello world"
        let normalized = normalizer.normalize(text)
        let score = normalizer.confidenceScore(original: text, normalized: normalized)
        XCTAssertGreaterThan(score, 0.9)
    }

    func testConfidenceScoreHeavyCorrection() {
        let original = "delta two five six descend to flight level three five zero"
        let normalized = normalizer.normalize(original)
        let score = normalizer.confidenceScore(original: original, normalized: normalized)
        XCTAssertGreaterThan(score, 0)
        XCTAssertLessThanOrEqual(score, 1)
    }

    // MARK: - 性能测试

    func testPerformanceNormalize() {
        let input = "delta two five six descend to flight level three five zero"
        measure {
            for _ in 0..<100 {
                _ = normalizer.normalize(input)
            }
        }
    }
}
