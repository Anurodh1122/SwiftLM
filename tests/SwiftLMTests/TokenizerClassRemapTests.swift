import XCTest
import Foundation
@testable import SwiftLM

// MARK: - tokenizer_class values swift-transformers does not know
//
// OLMoE ships `tokenizer_class: GPTNeoXTokenizer`. swift-transformers' table has no such
// entry and throws before the server starts. The checkpoint's tokenizer.json is plain
// byte-level BPE, so SwiftLM loads it from a scratch copy that says GPT2Tokenizer.
final class TokenizerClassRemapTests: XCTestCase {

    private var dirs: [URL] = []

    override func tearDown() {
        for d in dirs { try? FileManager.default.removeItem(at: d) }
        dirs = []
        super.tearDown()
    }

    private func makeModelDir(tokenizerClass: String?, modelType: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tok-remap-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        dirs.append(dir)
        var config: [String: Any] = ["model_max_length": 1024]
        if let tokenizerClass { config["tokenizer_class"] = tokenizerClass }
        try JSONSerialization.data(withJSONObject: config).write(to: dir.appendingPathComponent("tokenizer_config.json"))
        try JSONSerialization.data(withJSONObject: ["model": ["type": modelType]])
            .write(to: dir.appendingPathComponent("tokenizer.json"))
        try Data("weights".utf8).write(to: dir.appendingPathComponent("model.safetensors"))
        return dir
    }

    private func tokenizerClass(in dir: URL) throws -> String? {
        let data = try Data(contentsOf: dir.appendingPathComponent("tokenizer_config.json"))
        return (try JSONSerialization.jsonObject(with: data) as? [String: Any])?["tokenizer_class"] as? String
    }

    func testGPTNeoXBPEIsRemappedToGPT2() throws {
        let dir = try makeModelDir(tokenizerClass: "GPTNeoXTokenizer", modelType: "BPE")
        let out = remappingBPECompatibleTokenizerClassIfNeeded(in: dir, modelId: "test")
        dirs.append(out)
        XCTAssertNotEqual(out, dir)
        XCTAssertEqual(try tokenizerClass(in: out), "GPT2Tokenizer")
        XCTAssertEqual(try tokenizerClass(in: dir), "GPTNeoXTokenizer", "the original checkpoint is not modified")
        XCTAssertEqual(try Data(contentsOf: out.appendingPathComponent("model.safetensors")), Data("weights".utf8),
                       "other files are reachable through the scratch copy")
    }

    func testFastSuffixIsStripped() throws {
        let dir = try makeModelDir(tokenizerClass: "GPTNeoXTokenizerFast", modelType: "BPE")
        let out = remappingBPECompatibleTokenizerClassIfNeeded(in: dir, modelId: "test")
        dirs.append(out)
        XCTAssertEqual(try tokenizerClass(in: out), "GPT2Tokenizer")
    }

    func testKnownClassIsLeftAlone() throws {
        let dir = try makeModelDir(tokenizerClass: "Qwen2Tokenizer", modelType: "BPE")
        XCTAssertEqual(remappingBPECompatibleTokenizerClassIfNeeded(in: dir, modelId: "test"), dir)
    }

    /// A GPTNeoX-labelled checkpoint whose tokenizer.json is not BPE must not be forced
    /// into the BPE implementation: the loader should report its own error instead.
    func testNonBPEModelIsLeftAlone() throws {
        let dir = try makeModelDir(tokenizerClass: "GPTNeoXTokenizer", modelType: "Unigram")
        XCTAssertEqual(remappingBPECompatibleTokenizerClassIfNeeded(in: dir, modelId: "test"), dir)
    }

    func testMissingTokenizerClassIsLeftAlone() throws {
        let dir = try makeModelDir(tokenizerClass: nil, modelType: "BPE")
        XCTAssertEqual(remappingBPECompatibleTokenizerClassIfNeeded(in: dir, modelId: "test"), dir)
    }
}
