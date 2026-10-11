import XCTest
import Foundation
import Hummingbird
import HTTPTypes
import MLXLMCommon
import NIOCore
@testable import SwiftLM

// MARK: - `finish_reason` on the non-streaming /v1/completions path
//
// The handler used to hardcode "stop": it read `info.generationTokenCount` and ignored
// `info.stopReason`, so a completion cut off by `max_tokens` was reported as a natural
// stop. The chat handler and the *streaming* text handler both already mapped `.length`.
// These drive the handler with a synthetic generation stream, no model needed.
final class TextCompletionFinishReasonTests: XCTestCase {

    private final class BodySink: @unchecked Sendable {
        var data = Data()
    }

    private struct CollectingWriter: ResponseBodyWriter {
        let sink: BodySink
        mutating func write(_ buffer: ByteBuffer) async throws {
            sink.data.append(contentsOf: buffer.readableBytesView)
        }
        consuming func finish(_ trailingHeaders: HTTPFields?) async throws {}
    }

    /// Runs the non-streaming text handler over `events` and decodes the JSON it answers with.
    private func respond(to events: [Generation], stop: [String] = []) async throws -> [String: Any] {
        let (stream, continuation) = AsyncStream<Generation>.makeStream()
        for event in events { continuation.yield(event) }
        continuation.finish()

        let semaphore = AsyncSemaphore(limit: 1)
        await semaphore.wait()
        let slot = GenerationSlot(semaphore: semaphore)
        let response = try await handleTextNonStreaming(
            stream: stream, modelId: "test-model", stopSequences: stop,
            promptTokenCount: 3, slot: slot, stats: ServerStats(), genStart: Date())

        let sink = BodySink()
        try await response.body.write(CollectingWriter(sink: sink))
        return try XCTUnwrap(JSONSerialization.jsonObject(with: sink.data) as? [String: Any])
    }

    private func info(stopReason: GenerateStopReason, tokens: Int) -> Generation {
        .info(GenerateCompletionInfo(
            promptTokenCount: 3, generationTokenCount: tokens,
            promptTime: 0, generationTime: 0, stopReason: stopReason))
    }

    private func finishReason(_ body: [String: Any]) throws -> String {
        let choices = try XCTUnwrap(body["choices"] as? [[String: Any]])
        return try XCTUnwrap(choices.first?["finish_reason"] as? String)
    }

    func testMaxTokensReportsLength() async throws {
        let body = try await respond(to: [
            .chunk("one", tokenId: 1), .chunk(" two", tokenId: 2),
            info(stopReason: .length, tokens: 2),
        ])
        XCTAssertEqual(try finishReason(body), "length")
        let choices = try XCTUnwrap(body["choices"] as? [[String: Any]])
        XCTAssertEqual(choices.first?["text"] as? String, "one two", "the text itself is untouched")
    }

    func testEndOfSequenceReportsStop() async throws {
        let body = try await respond(to: [
            .chunk("done", tokenId: 1), info(stopReason: .stop, tokens: 1),
        ])
        XCTAssertEqual(try finishReason(body), "stop")
    }

    /// A client stop sequence is a stop even when the generator ran into its token limit
    /// on the same chunk: the text was cut at the stop, so that is what happened.
    func testClientStopSequenceReportsStop() async throws {
        let body = try await respond(to: [
            .chunk("alpha", tokenId: 1), .chunk("<|end|>beta", tokenId: 2),
            info(stopReason: .length, tokens: 2),
        ], stop: ["<|end|>"])
        XCTAssertEqual(try finishReason(body), "stop")
        let choices = try XCTUnwrap(body["choices"] as? [[String: Any]])
        XCTAssertEqual(choices.first?["text"] as? String, "alpha")
    }

    /// The `.info` event carries the real token count (the chunk count undercounts when
    /// the decoder buffers), and that must survive the finish_reason change.
    func testUsageStillComesFromTheInfoEvent() async throws {
        let body = try await respond(to: [
            .chunk("x", tokenId: 1), info(stopReason: .length, tokens: 7),
        ])
        let usage = try XCTUnwrap(body["usage"] as? [String: Any])
        XCTAssertEqual(usage["completion_tokens"] as? Int, 7)
        XCTAssertEqual(usage["prompt_tokens"] as? Int, 3)
        XCTAssertEqual(usage["total_tokens"] as? Int, 10)
    }
}
