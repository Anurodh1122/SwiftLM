// APIReplyFilter.swift — turns engine tokens into the parts of an OpenAI reply
import Foundation
#if canImport(MLXInferenceCore)
import MLXInferenceCore
#endif

/// Shared by streaming and non-streaming replies: reasoning, answer text cut at the
/// request's `stop` sequences, tool calls (including ones the library rejected) with
/// arguments in their declared types, and the finish reason.
struct APIReplyFilter {
    enum Event {
        case content(String)
        case reasoning(String)
        case toolCall(ChatMessage.ToolCallRecord)
        case finish(reason: String, usage: (prompt: Int, completion: Int))
    }

    /// Declared tools: name -> parameter name -> JSON Schema type.
    let toolSchemas: [String: [String: String]]
    let stops: [String]
    let maxTokens: Int

    /// Set once a stop sequence appears; the caller then stops generation.
    private(set) var stopped = false
    private var heldCall = ""   // from a `<tool_call>` the library did not take as a call
    private var heldText = ""   // a tail that may be the start of a stop sequence
    private var toolCallCount = 0

    init(toolSchemas: [String: [String: String]], stops: [String], maxTokens: Int) {
        self.toolSchemas = toolSchemas
        self.stops = stops
        self.maxTokens = maxTokens
    }

    mutating func feed(_ token: GenerationToken) -> [Event] {
        if let usage = token.usage { return finish(usage) }
        if stopped { return [] }
        if token.isThinking { return token.text.isEmpty ? [] : [.reasoning(token.text)] }
        if let call = token.toolCall {
            toolCallCount += 1
            return releaseHeldText() + [.toolCall(normalized(call))]
        }
        if !toolSchemas.isEmpty {
            if heldCall.isEmpty, let start = token.text.range(of: "<tool_call>") {
                let events = text(String(token.text[..<start.lowerBound]))
                heldCall = String(token.text[start.lowerBound...])
                return events
            } else if !heldCall.isEmpty {
                heldCall += token.text
                return []
            }
        }
        return text(token.text)
    }

    /// The end of the reply. `usage` is nil when generation failed before its counts.
    mutating func finish(_ usage: (prompt: Int, completion: Int)?) -> [Event] {
        var events: [Event] = []
        if !heldCall.isEmpty {
            let held = heldCall
            heldCall = ""
            if let recovered = ServerManager.recoverToolCalls(from: held, toolNames: toolSchemas) {
                events += releaseHeldText()
                for call in recovered.calls {
                    toolCallCount += 1
                    events.append(.toolCall(normalized(call)))
                }
            } else {
                events += text(held)
            }
        }
        events += releaseHeldText()
        guard let usage else { return events }
        let reason = toolCallCount > 0 ? "tool_calls"
            : (!stopped && usage.completion >= maxTokens ? "length" : "stop")
        events.append(.finish(reason: reason, usage: usage))
        return events
    }

    private mutating func text(_ s: String) -> [Event] {
        guard !s.isEmpty, !stopped else { return [] }
        guard !stops.isEmpty else { return [.content(s)] }
        heldText += s
        if let hit = stops.compactMap({ heldText.range(of: $0) }).min(by: { $0.lowerBound < $1.lowerBound }) {
            let before = String(heldText[..<hit.lowerBound])
            heldText = ""
            stopped = true
            return before.isEmpty ? [] : [.content(before)]
        }
        // Hold back the longest tail that could still begin a stop sequence.
        var keep = min(heldText.count, (stops.map(\.count).max() ?? 1) - 1)
        while keep > 0, !stops.contains(where: { $0.hasPrefix(heldText.suffix(keep)) }) { keep -= 1 }
        let out = String(heldText.dropLast(keep))
        heldText = String(heldText.suffix(keep))
        return out.isEmpty ? [] : [.content(out)]
    }

    private mutating func releaseHeldText() -> [Event] {
        guard !heldText.isEmpty, !stopped else { return [] }
        defer { heldText = "" }
        return [.content(heldText)]
    }

    /// Arguments in their declared types: Qwen writes every parameter as text, so an
    /// integer came out as "45", a boolean as "True" and an array as its JSON text.
    private func normalized(_ call: ChatMessage.ToolCallRecord) -> ChatMessage.ToolCallRecord {
        guard let types = toolSchemas[call.name], !types.isEmpty,
              let data = call.arguments.data(using: .utf8),
              var arguments = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else { return call }
        var changed = false
        for (key, type) in types {
            guard let raw = arguments[key] as? String else { continue }
            let value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            var typed: Any?
            switch type {
            case "integer": typed = Int(value)
            case "number": typed = Double(value)
            case "boolean":
                if value.lowercased() == "true" { typed = true } else if value.lowercased() == "false" { typed = false }
            case "array", "object":
                let parsed = value.data(using: .utf8).flatMap { try? JSONSerialization.jsonObject(with: $0) }
                if type == "array" ? parsed is [Any] : parsed is [String: Any] { typed = parsed }
            default: break
            }
            if let typed {
                arguments[key] = typed
                changed = true
            }
        }
        guard changed, let out = try? JSONSerialization.data(withJSONObject: arguments) else { return call }
        var result = call
        result.arguments = String(decoding: out, as: UTF8.self)
        return result
    }
}
