// ChatMessage.swift — SwiftLM shared message model
// Used by both MLXInferenceCore and SwiftLM Chat UI

import Foundation

/// Represents a single turn in a chat conversation.
public struct ChatMessage: Identifiable, Codable, Equatable, Sendable {
    public let id: UUID
    public let role: Role
    public var content: String
    public var thinkingContent: String?
    public let timestamp: Date
    /// How a reply was generated, shown under it. Only set on replies from this session.
    public var stats: GenerationStats?
    /// The tools an assistant message called (OpenAI `tool_calls`), for API clients
    /// that send their history back.
    public var toolCalls: [ToolCallRecord]?
    /// On a tool result: the call it answers (OpenAI `tool_call_id`).
    public var toolCallID: String?

    /// One tool call, with its arguments as a JSON object string, as OpenAI sends it.
    public struct ToolCallRecord: Codable, Equatable, Sendable {
        public var id: String
        public var name: String
        public var arguments: String

        public init(id: String, name: String, arguments: String) {
            self.id = id
            self.name = name
            self.arguments = arguments
        }
    }

    public struct GenerationStats: Codable, Equatable, Sendable {
        public var tokens: Int
        public var tokensPerSecond: Double
        public var timeToFirstToken: Double

        public init(tokens: Int, tokensPerSecond: Double, timeToFirstToken: Double) {
            self.tokens = tokens
            self.tokensPerSecond = tokensPerSecond
            self.timeToFirstToken = timeToFirstToken
        }
    }

    public enum Role: String, Codable, Sendable {
        case system
        case user
        case assistant
        case tool
    }

    public init(role: Role, content: String, thinkingContent: String? = nil, id: UUID = UUID(), timestamp: Date = Date()) {
        self.id = id
        self.role = role
        self.content = content
        self.thinkingContent = thinkingContent
        self.timestamp = timestamp
    }

    // Convenience constructors
    public static func system(_ content: String) -> ChatMessage {
        ChatMessage(role: .system, content: content)
    }
    public static func user(_ content: String) -> ChatMessage {
        ChatMessage(role: .user, content: content)
    }
    public static func assistant(_ content: String, thinkingContent: String? = nil) -> ChatMessage {
        ChatMessage(role: .assistant, content: content, thinkingContent: thinkingContent)
    }
    public static func tool(_ content: String) -> ChatMessage {
        ChatMessage(role: .tool, content: content)
    }
}
