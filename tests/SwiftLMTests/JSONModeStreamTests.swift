import XCTest
import Foundation
import Hummingbird
import HTTPTypes
import MLXLMCommon
import NIOCore
@testable import SwiftLM

// MARK: - JSON mode on the streaming chat path (#233)
//
// JSON-mode output is buffered so a hallucinated prefix object can be dropped before
// anything reaches the client. The buffer used to be flushed only from inside a chunk,
// once 32 chunks had arrived or a second `{` showed up, and never when the stream ended.
// A reply that starts with `{` is held by the generator as a possible bare tool call and
// arrives as one chunk, so in practice a JSON-mode stream sent nothing. These drive the
// handler with a synthetic generation stream, no model needed.
final class JSONModeStreamTests: XCTestCase {

    // MARK: The prefix rule

    func testFlatObjectIsUntouched() {
        XCTAssertEqual(jsonModeFlushText("{\"ok\": true}"), "{\"ok\": true}")
    }

    /// The old rule cut at the second `{` wherever it was, so a nested object lost its
    /// outer key and kept an unbalanced brace.
    func testNestedObjectIsNotCutAtItsInnerBrace() {
        let nested = "{\"user\": {\"name\": \"Bob\"}}"
        XCTAssertEqual(jsonModeFlushText(nested), nested)
        let list = "{\"items\": [{\"id\": 1}, {\"id\": 2}]}"
        XCTAssertEqual(jsonModeFlushText(list), list)
    }

    /// The reason the buffer exists: a complete object before the reply is dropped.
    func testClosedPrefixObjectIsDropped() {
        XCTAssertEqual(jsonModeFlushText("{\"thought\": \"ok\"}\n{\"answer\": 1}"), "{\"answer\": 1}")
        XCTAssertEqual(jsonModeFlushText("{}{\"answer\": 1}"), "{\"answer\": 1}")
    }

    func testOpenObjectAndNonObjectTextPassThroughTrimmed() {
        XCTAssertEqual(jsonModeFlushText("  {\"a\": {\"b\": "), "{\"a\": {\"b\":")
        XCTAssertEqual(jsonModeFlushText("```json\n{\"a\": 1}\n"), "```json\n{\"a\": 1}")
    }

    // MARK: The handler

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

    private func info(stopReason: GenerateStopReason, tokens: Int) -> Generation {
        .info(GenerateCompletionInfo(
            promptTokenCount: 3, generationTokenCount: tokens,
            promptTime: 0, generationTime: 0, stopReason: stopReason))
    }

    private func chunks(_ pieces: [String]) -> [Generation] {
        pieces.enumerated().map { .chunk($1, tokenId: $0 + 1) }
    }

    /// Streams `events` through the chat handler and returns the content deltas it put
    /// on the wire, in order.
    private func contentDeltas(for events: [Generation], jsonMode: Bool = true) async throws -> [String] {
        let (stream, continuation) = AsyncStream<Generation>.makeStream()
        for event in events { continuation.yield(event) }
        continuation.finish()

        let semaphore = AsyncSemaphore(limit: 1)
        await semaphore.wait()
        let slot = GenerationSlot(semaphore: semaphore)
        let response = handleChatStreaming(
            startGeneration: { (stream, nil) }, modelId: "test-model", stopSequences: [],
            includeUsage: false, promptTokenCount: 3, jsonMode: jsonMode, tokenEcho: false,
            slot: slot, stats: ServerStats(), genStart: Date(), prefillStart: Date(),
            emitPrefillProgress: false)

        let sink = BodySink()
        try await response.body.write(CollectingWriter(sink: sink))
        let body = String(decoding: sink.data, as: UTF8.self)
        var deltas: [String] = []
        for raw in body.components(separatedBy: "\n") {
            let line = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard line.hasPrefix("data: {") else { continue }
            let event = try XCTUnwrap(
                JSONSerialization.jsonObject(with: Data(line.dropFirst(6).utf8)) as? [String: Any])
            let choices = try XCTUnwrap(event["choices"] as? [[String: Any]])
            if let delta = choices.first?["delta"] as? [String: Any],
               let text = delta["content"] as? String {
                deltas.append(text)
            }
        }
        return deltas
    }

    /// The bug itself: a reply shorter than the flush threshold, delivered as one chunk,
    /// never reached the client.
    func testReplyShorterThanTheFlushThresholdIsSentWhenTheStreamEnds() async throws {
        let deltas = try await contentDeltas(for: [
            .chunk("{\"ok\": true}", tokenId: 1), info(stopReason: .stop, tokens: 5),
        ])
        XCTAssertEqual(deltas.joined(), "{\"ok\": true}")
    }

    /// A stream that ends without `.info` (an error, a cancelled decode) keeps what it had.
    func testBufferIsSentWhenTheStreamEndsWithoutInfo() async throws {
        let deltas = try await contentDeltas(for: [.chunk("{\"ok\": true}", tokenId: 1)])
        XCTAssertEqual(deltas.joined(), "{\"ok\": true}")
    }

    /// With per-token delivery the second `{` of a nested object used to trigger the
    /// prefix cut, so the client saw `{"name": "Bob"}}`.
    func testNestedObjectArrivingPerTokenIsForwardedWhole() async throws {
        let pieces = ["{\"", "user", "\": {\"", "name", "\": \"", "Bob", "\"}}"]
        let deltas = try await contentDeltas(for: chunks(pieces) + [info(stopReason: .stop, tokens: pieces.count)])
        XCTAssertEqual(deltas.joined(), "{\"user\": {\"name\": \"Bob\"}}")
    }

    /// What the buffer is for, unchanged: a closed object ahead of the reply is dropped.
    func testHallucinatedPrefixObjectIsStillDropped() async throws {
        let pieces = ["{\"", "note", "\": 1}", "{\"", "answer", "\": 2}"]
        let deltas = try await contentDeltas(for: chunks(pieces) + [info(stopReason: .stop, tokens: pieces.count)])
        XCTAssertEqual(deltas.joined(), "{\"answer\": 2}")
    }

    func testOutsideJSONModeTheChunkIsForwardedAsIs() async throws {
        let deltas = try await contentDeltas(
            for: [.chunk("{\"ok\": true}", tokenId: 1), info(stopReason: .stop, tokens: 5)],
            jsonMode: false)
        XCTAssertEqual(deltas, ["{\"ok\": true}"])
    }
}
