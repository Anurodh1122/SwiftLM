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
