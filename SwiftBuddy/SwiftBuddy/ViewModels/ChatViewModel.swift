// ChatViewModel.swift — Bridges InferenceEngine actor to SwiftUI
import SwiftUI
import Combine
import SwiftData
#if canImport(MLXInferenceCore)
import MLXInferenceCore
#endif

@MainActor
final class ChatViewModel: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var streamingText: String = ""
    @Published var thinkingText: String? = nil
    @Published var isGenerating: Bool = false
    @Published var config: GenerationConfig = .load() {
        didSet { scheduleConfigSave() }
    }
    @Published var systemPrompt: String = {
        UserDefaults.standard.string(forKey: "swiftlm.systemPrompt") ?? ""
    }() {
        didSet { scheduleSystemPromptSave() }
    }
    // Published so the title, the persona badge and the menu's checkmark follow it; as a
    // plain property, picking a persona in the chat's menu changed nothing on screen.
    @Published public var currentWing: String? = nil
    weak var engine: InferenceEngine?
    var modelContext: ModelContext?
    private var generationTask: Task<Void, Never>?
    private var activeSession: ChatSession?

    // MARK: — Debounced persistence
    // Saves are debounced at 0.5 s so rapid slider drags or keystrokes
    // don't saturate UserDefaults with synchronous writes and cause UI jank.
    private var configSaveWork: DispatchWorkItem?
    private var systemPromptSaveWork: DispatchWorkItem?

    private func scheduleConfigSave() {
        configSaveWork?.cancel()
        let work = DispatchWorkItem { [weak self] in self?.config.save() }
        configSaveWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5, execute: work)
    }

    private func scheduleSystemPromptSave() {
        systemPromptSaveWork?.cancel()
        let snapshot = systemPrompt
        let work = DispatchWorkItem {
            UserDefaults.standard.set(snapshot, forKey: "swiftlm.systemPrompt")
        }
        systemPromptSaveWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5, execute: work)
    }

    // MARK: — Send

    func send(_ userText: String) async {
        guard let engine, !isGenerating else { return }

        let userMessage = ChatMessage.user(userText)
        messages.append(userMessage)

        if let context = modelContext, let session = activeSession {
            let turn = ChatTurn(id: userMessage.id, roleRaw: "user", content: userMessage.content, timestamp: userMessage.timestamp, session: session)
            context.insert(turn)
            try? context.save()
        }

        isGenerating = true
        streamingText = ""
        thinkingText = nil
        
        var fullMessages = messages
        
        // 1. Prepend System Persona dynamically for the MLX Engine context (Stateless & Cache-Perfect)
        let identityPayload = await buildIdentityPayload(userText: userText)
        if !identityPayload.isEmpty {
            // Remove any existing system roles to prevent duplication
            fullMessages.removeAll { $0.role == .system }
            fullMessages.insert(ChatMessage.system(identityPayload), at: 0)
        }
        
        // Squash consecutive roles to prevent Jinja alternation crashes on strict models (e.g., Gemma)
        var collapsedMessages: [ChatMessage] = []
        for msg in fullMessages {
            if let last = collapsedMessages.last, last.role == msg.role {
                collapsedMessages[collapsedMessages.count - 1].content += "\n\n" + msg.content
            } else {
                collapsedMessages.append(msg)
            }
        }
        fullMessages = collapsedMessages

        generationTask = Task {
            var latestMessages = fullMessages
            var shouldGenerateAgain = true
            var depth = 0
            
            while shouldGenerateAgain && depth < 3 {
                shouldGenerateAgain = false
                depth += 1
                print("[Chat] round \(depth): \(latestMessages.count) messages, thinking \(config.enableThinking ? "on" : "off")")

                var response = ""
                var thinking = ""
                var hasRawThinkTags = false
                // What the bubble should show, published at most ~12 times a second:
                // re-rendering the whole Markdown reply on every token made long replies
                // stutter.
                var liveText = ""
                var liveThinking: String? = nil
                var lastPublish = Date.distantPast

                for await token in engine.generate(messages: latestMessages, config: config) {
                    guard !Task.isCancelled else { break }
                    defer {
                        let now = Date()
                        if now.timeIntervalSince(lastPublish) >= 0.08 {
                            lastPublish = now
                            streamingText = Self.hidingToolCall(liveText)
                            thinkingText = liveThinking
                        }
                    }

                    if token.isThinking {
                        thinking += token.text
                        liveThinking = thinking
                    } else {
                        response += token.text
                        
                        // Fallback cleanup if the model outputs literal <think>...</think> tags
                        // and the tokenizer isn't setting the isThinking flag correctly.
                        if response.contains("<think>") {
                            hasRawThinkTags = true
                            
                            // Try to safely extract thinking content between the tags
                            if let startRange = response.range(of: "<think>"),
                               let endRange = response.range(of: "</think>") {
                                // Extract thinking
                                let rawThinking = String(response[startRange.upperBound..<endRange.lowerBound])
                                liveThinking = rawThinking
                                
                                // Remove the entire block from the visible response
                                let before = String(response[..<startRange.lowerBound])
                                let after = String(response[endRange.upperBound...])
                                liveText = before + after
                            } else if let startRange = response.range(of: "<think>") {
                                // We have a start tag but no end tag yet, it's currently generating the thought
                                let rawThinking = String(response[startRange.upperBound...])
                                liveThinking = rawThinking
                                
                                // Only update streaming text with what came before
                                liveText = String(response[..<startRange.lowerBound])
                            }
                        } else if let endRange = response.range(of: "</think>") {
                            // The template pre-filled `<think>`, so the reply began inside the
                            // thought and only the closing tag arrives: move what came before
                            // it out of the answer.
                            hasRawThinkTags = true
                            liveThinking = String(response[..<endRange.lowerBound])
                                .trimmingCharacters(in: .whitespacesAndNewlines)
                            liveText = String(response[endRange.upperBound...])
                        } else if !hasRawThinkTags {
                            // Standard flow: no raw tags seen yet, just stream normally
                            liveText = response 
                        }
                    }
                }

                // Stop (or switching chats) cancels this task. `stopGeneration()` has already
                // saved the partial reply and reset the UI, so committing `response` again here
                // duplicated it, could land it in the next conversation, and could even run a
                // tool the user had just stopped.
                if Task.isCancelled {
                    print("[Chat] round \(depth) cancelled after \(response.count) chars")
                    return
                }
                print("[Chat] round \(depth) reply: \(response.count) chars (visible \(liveText.count), thinking \((liveThinking ?? thinking).count))")
                // Publish the last tokens the throttle held back; `thinkingText` is also
                // what the committed message records below.
                streamingText = Self.hidingToolCall(liveText)
                thinkingText = liveThinking

                // First, check if there's a tool call in the complete response
                if let toolCall = ExtractionService.extractToolCall(from: response) {
                    print("[Chat] round \(depth): tool call \(toolCall.name)")
                    // Extract text BEFORE the tool call to save as assistant message
                    if let startRange = response.range(of: "<tool_call>") {
                        let textBeforeTool = String(response[..<startRange.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
                        if !textBeforeTool.isEmpty {
                            let msg = ChatMessage.assistant(textBeforeTool, thinkingContent: thinkingText)
                            messages.append(msg)
                            latestMessages.append(msg)
                            if let context = modelContext, let session = activeSession {
                                let turn = ChatTurn(id: msg.id, roleRaw: "assistant", content: msg.content, thinkingContent: thinkingText, timestamp: msg.timestamp, session: session)
                                context.insert(turn)
                                try? context.save()
                            }
                        }
                    }
                    
                    // Execute tool natively!
                    do {
                        // Tools act on the open persona whatever wing the model names: a
                        // misspelt name ("Albert Einstein" for "Einstein") created a new persona.
                        var arguments = toolCall.parameters ?? [:]
                        if let wing = currentWing { arguments["wing"] = wing }
                        let toolResult = try await MemoryPalaceTools.handleToolCall(name: toolCall.name, arguments: arguments)
                        let msg = ChatMessage.tool(toolResult)
                        messages.append(msg)
                        latestMessages.append(msg)
                        // Trigger generation loop again!
                        shouldGenerateAgain = true
                        continue
                    } catch {
                        let errorMsg = ChatMessage.tool("Error executing tool: \(error.localizedDescription)")
                        messages.append(errorMsg)
                        latestMessages.append(errorMsg)
                        shouldGenerateAgain = true
                        continue
                    }
                }

                // If no tool call, commit the standard completed message
                if !response.isEmpty {
                    // Do a final cleanup just in case
                    var finalVisible = response
                    if let startRange = response.range(of: "<think>"),
                       let endRange = response.range(of: "</think>") {
                        let before = String(response[..<startRange.lowerBound])
                        let after = String(response[endRange.upperBound...])
                        finalVisible = before + after
                    } else if let startRange = response.range(of: "<think>") {
                         finalVisible = String(response[..<startRange.lowerBound])
                    } else if let endRange = response.range(of: "</think>") {
                        finalVisible = String(response[endRange.upperBound...])
                    }
                    
                    // Trim leading newlines that often follow thought blocks
                    finalVisible = finalVisible.trimmingCharacters(in: .whitespacesAndNewlines)
                    
                    print("[Chat] round \(depth): saving \(finalVisible.count) visible chars")
                    if !finalVisible.isEmpty {
                        let msg = ChatMessage.assistant(finalVisible, thinkingContent: thinkingText)
                        messages.append(msg)
                        if let context = modelContext, let session = activeSession {
                            let turn = ChatTurn(id: msg.id, roleRaw: "assistant", content: msg.content, thinkingContent: thinkingText, timestamp: msg.timestamp, session: session)
                            context.insert(turn)
                            try? context.save()
                        }
                    }
                }

                streamingText = ""
                thinkingText = nil
                isGenerating = false
            } // end while

            // The loop can end on its depth limit straight after a tool call, which skips the
            // reset at the end of an iteration and left `isGenerating` stuck on, locking the chat.
            streamingText = ""
            thinkingText = nil
            isGenerating = false
        }

        await generationTask?.value
    }

    /// The part of a streamed reply to show. A tool call is for the app to run, not for
    /// the reader, so everything from `<tool_call>` on is held back, including a tag that
    /// is still arriving.
    static func hidingToolCall(_ text: String) -> String {
        let tag = "<tool_call>"
        if let range = text.range(of: tag) {
            return String(text[..<range.lowerBound])
        }
        if let open = text.lastIndex(of: "<"), tag.hasPrefix(String(text[open...])) {
            return String(text[..<open])
        }
        return text
    }

    // MARK: — Edit & resend

    /// Removes `message` and everything after it from the open chat, saved turns included.
    private func truncate(from message: ChatMessage) {
        guard let index = messages.firstIndex(where: { $0.id == message.id }) else { return }
        let removed = Set(messages[index...].map(\.id))
        messages.removeSubrange(index...)
        if let context = modelContext, let session = activeSession {
            for turn in session.turns where removed.contains(turn.id) {
                context.delete(turn)
            }
            try? context.save()
        }
    }

    /// Takes a sent message back out of the chat, with the replies after it, so it can be
    /// edited and sent again. Returns its text for the input field.
    func editMessage(_ message: ChatMessage) -> String {
        stopGeneration()
        truncate(from: message)
        return message.content
    }

    /// Sends a message again from that point, dropping the replies that followed it.
    func resend(_ message: ChatMessage) async {
        stopGeneration()
        truncate(from: message)
        await send(message.content)
    }

    /// Asks again for a reply: resends the user message this reply answered.
    func regenerate(_ reply: ChatMessage) async {
        guard let index = messages.firstIndex(where: { $0.id == reply.id }),
              let prompt = messages[..<index].last(where: { $0.role == .user }) else { return }
        await resend(prompt)
    }

    // MARK: — Controls

    func stopGeneration() {
        generationTask?.cancel()
        if !streamingText.isEmpty {
            let msg = ChatMessage.assistant(streamingText, thinkingContent: thinkingText)
            messages.append(msg)
            if let context = modelContext, let session = activeSession {
                let turn = ChatTurn(id: msg.id, roleRaw: "assistant", content: msg.content, thinkingContent: thinkingText, timestamp: msg.timestamp, session: session)
                context.insert(turn)
                try? context.save()
            }
        }
        streamingText = ""
        thinkingText = nil
        isGenerating = false
    }

    /// Opens the conversation for `wing` (nil = the core chat) unless it is already open.
    /// The chat screen calls this from `onAppear`, which also fires on every return to
    /// it (switching tabs); reopening then stopped a reply in progress.
    func openConversation(wing: String?) {
        if activeSession != nil, currentWing == wing { return }
        currentWing = wing
        newConversation()
    }

    /// The chat currently open, so the history list can mark it.
    var activeSessionID: UUID? { activeSession?.id }

    /// Starts a fresh chat for the current wing; earlier chats stay in the history list.
    /// An empty chat is reused rather than piling up blank ones.
    func startNewChat() {
        stopGeneration()
        guard let context = modelContext else {
            messages = []
            return
        }
        if let session = activeSession, session.turns.isEmpty {
            messages = []
            return
        }
        let newSession = ChatSession(wingName: currentWing)
        context.insert(newSession)
        try? context.save()
        activeSession = newSession
        messages = []
    }

    /// Deletes the open chat and opens the most recent remaining one (or a fresh chat).
    func deleteActiveChat() {
        stopGeneration()
        guard let context = modelContext, let session = activeSession else {
            messages = []
            return
        }
        context.delete(session)  // its turns go with it (cascade)
        try? context.save()
        activeSession = nil
        newConversation()
    }

    /// Deletes every chat of the current persona (or of the core chat), then starts fresh.
    func deleteAllChats() {
        stopGeneration()
        guard let context = modelContext else {
            messages = []
            return
        }
        let wing = currentWing
        for session in (try? context.fetch(FetchDescriptor<ChatSession>())) ?? [] where session.wingName == wing {
            context.delete(session)
        }
        try? context.save()
        activeSession = nil
        newConversation()
    }

    /// Deletes a persona with its memories, chats and graph. If its chat is open, the core
    /// chat takes its place rather than keeping a deleted session.
    func deletePersona(_ name: String) {
        if currentWing == name {
            stopGeneration()
            activeSession = nil
            messages = []
        }
        do {
            try MemoryPalaceService.shared.deletePersona(name)
        } catch {
            print("[Chat] couldn't delete persona \(name): \(error)")
        }
        if currentWing == name {
            currentWing = nil
            newConversation()
        }
    }

    /// Reopens an earlier chat from the history list.
    func openSession(_ session: ChatSession) {
        guard session.id != activeSession?.id else { return }
        stopGeneration()
        currentWing = session.wingName
        activeSession = session
        loadMessages(from: session)
    }

    /// Restores a chat's turns in order.
    private func loadMessages(from session: ChatSession) {
        let sortedTurns = session.turns.sorted { $0.timestamp < $1.timestamp }
        messages = sortedTurns.map { turn in
            let role: ChatMessage.Role = turn.roleRaw == "assistant" ? .assistant : (turn.roleRaw == "system" ? .system : .user)
            return ChatMessage(role: role, content: turn.content, thinkingContent: turn.thinkingContent, id: turn.id, timestamp: turn.timestamp)
        }
    }

    func newConversation() {
        stopGeneration()

        let targetWing = currentWing ?? "CORE_SYSTEM"
        
        if let context = modelContext {
            let fetchDesc = FetchDescriptor<ChatSession>()
            let allSessions = try? context.fetch(fetchDesc)
            
            // The most recent chat for this wing (fetch order is arbitrary, so with
            // several chats per wing an old one could come back).
            let session = allSessions?
                .filter {
                    if targetWing == "CORE_SYSTEM" { return $0.wingName == nil }
                    return $0.wingName == targetWing
                }
                .max { $0.createdAt < $1.createdAt }

            if let existing = session {
                activeSession = existing
                loadMessages(from: existing)
            } else {
                // Creates the fresh session!
                let wingParam = targetWing == "CORE_SYSTEM" ? nil : targetWing
                let newSession = ChatSession(wingName: wingParam)
                context.insert(newSession)
                try? context.save()
                activeSession = newSession
                messages = []
            }
        } else {
            messages = []
        }
    }
    
    func clearHistory() {
        stopGeneration()
        messages.removeAll()
        if let context = modelContext, let session = activeSession {
            // Delete all stored turns for this session
            for turn in session.turns {
                context.delete(turn)
            }
            try? context.save()
            // Reset the loaded session state so we start fresh
            session.turns.removeAll()
        }
    }
    
    // MARK: - Tool Calling & Context Injection
    public func buildIdentityPayload(userText: String = "") async -> String {
        var wakeUpText = ""
        var activeRagDirective = ""
        
        if let wing = currentWing {
            do {
                // L0: identity and L1: critical facts, always loaded. Rooms are matched
                // loosely (see `identityContext`); exact names matched no persona.
                let persona = try MemoryPalaceService.shared.identityContext(wingName: wing)
                var l0Identity: [String] = []
                var l0Length = 0
                for line in persona.identity where l0Length + line.count <= 1500 {
                    l0Identity.append(line)
                    l0Length += line.count
                }
                let l1Facts = persona.facts

                // Truncate L1 payload to ~120 tokens (approx 500 chars) to prevent prompt bloat 
                // matching the AAAK MemPalace layers.py spec
                var l1Text = ""
                for fact in l1Facts {
                    if l1Text.count + fact.count > 600 {
                        l1Text += "\n  ... (more in L3 search)"
                        break
                    }
                    l1Text += "\n - \(fact)"
                }
                
                wakeUpText = "[SYSTEM OVERRIDE]: YOU ARE \(wing.uppercased()). YOU ARE NOT AN AI OPEN WEIGHTS MODEL. DO NOT INTRODUCE YOURSELF AS GEMINA OR LLAMA. ALWAYS RESPOND ENTIRELY IN CHARACTER.\n\n"
                
                if !l0Identity.isEmpty {
                    wakeUpText += "[L0 IDENTITY]:\n" + l0Identity.joined(separator: "\n") + "\n\n"
                }
                
                if !l1Text.isEmpty {
                    wakeUpText += "[L1 CRITICAL FACTS & PREFERENCES]:" + l1Text + "\n\n"
                }
                
                // ACTIVE RAG HOOK
                if !userText.isEmpty {
                    let facts = try MemoryPalaceService.shared.searchMemories(query: userText, wingName: wing)
                    if !facts.isEmpty {
                        let factList = facts.map { "- [\($0.hallType)] \($0.text)" }.joined(separator: "\n")
                        activeRagDirective = "\n\n[RELEVANT MEMORY CONTEXT FOR THIS TURN]:\n\(factList)\n\nYou must strictly incorporate these facts if they are relevant here."
                    }
                }
            } catch {
                print("RAG Pre-Fetch Failed: \(error.localizedDescription)")
            }
        }
        
        // Memory tools only for personas: their memories live in a wing. In the plain chat
        // small models called them for any question ("what can you do") and then asked the
        // user to set up the palace instead of answering.
        guard currentWing != nil else { return systemPrompt }
        let toolInjection = MemoryPalaceTools.schemaManifestString

        return wakeUpText + systemPrompt + activeRagDirective + "\n\n" + toolInjection
    }
}
