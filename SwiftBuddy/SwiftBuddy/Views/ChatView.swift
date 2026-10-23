// ChatView.swift — Premium chat interface (iOS + macOS)
import SwiftUI
import SwiftData
#if canImport(MLXInferenceCore)
import MLXInferenceCore
#endif

struct ChatView: View {
    @ObservedObject var viewModel: ChatViewModel
    @EnvironmentObject private var engine: InferenceEngine
    @Query(sort: \PalaceWing.createdDate) var wings: [PalaceWing]
    @Query(sort: \ChatSession.createdAt, order: .reverse) private var sessions: [ChatSession]

    // macOS-only sheet control (iOS: these are tabs)
    var showSettings: Binding<Bool>? = nil
    @State private var showCorruptionRecoveryAlert = false
    @State private var showErrorDetail = false
    @State private var confirmDeleteChat = false
    @State private var confirmDeleteAllChats = false
    @State private var inputText = ""
    @FocusState private var inputFocused: Bool

    var body: some View {
        ZStack {
            // ── Deep canvas background ───────────────────────────────────────
            SwiftBuddyTheme.background.ignoresSafeArea()

            VStack(spacing: 0) {
                // ── Message list ─────────────────────────────────────────────
                messageList

                // ── Engine state banner ──────────────────────────────────────
                engineBanner
                
                // ── Memory Active Badge ──────────────────────────────────────
                if let wing = viewModel.currentWing {
                    HStack {
                        Image(systemName: "brain.head.profile")
                        Text("\(wing)'s Memory Active")
                    }
                    .font(.caption2.bold())
                    .foregroundStyle(SwiftBuddyTheme.accent)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(SwiftBuddyTheme.accent.opacity(0.18))
                    .clipShape(Capsule())
                    .padding(.vertical, 6)
                }

                // ── Input bar ────────────────────────────────────────────────
                inputBar
            }
        }
        .navigationTitle(viewModel.currentWing != nil ? "Chatting with \(viewModel.currentWing!)" : "SwiftBuddy Chat")
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar { iOSToolbar }
        .toolbarBackground(SwiftBuddyTheme.background.opacity(0.90), for: .navigationBar)
        .toolbarBackground(.visible, for: .navigationBar)
        #else
        .toolbar { macOSToolbar }
        #endif
        .alert(
            "Model Download Problem",
            isPresented: $showCorruptionRecoveryAlert,
            actions: {
                Button("Delete & Re-download", role: .destructive) {
                    engine.deleteCorruptedAndRedownload()
                }
                Button("Choose Another Model") {
                    engine.clearCorruptionRecovery()
                    NotificationCenter.default.post(name: .showModelManagement, object: nil)
                }
                Button("Cancel", role: .cancel) {
                }
            },
            message: {
                Text("The downloaded weights for this model are corrupted or incomplete. Delete the cached files and start a fresh download, or choose a different model.")
            }
        )
        .onChange(of: engine.corruptedModelId) { _, modelId in
            if modelId != nil {
                showCorruptionRecoveryAlert = true
            }
        }
        .modifier(ChatDeletionDialogs(
            confirmDeleteChat: $confirmDeleteChat,
            confirmDeleteAllChats: $confirmDeleteAllChats,
            deleteChat: { withAnimation { viewModel.deleteActiveChat() } },
            deleteAllChats: { withAnimation { viewModel.deleteAllChats() } }
        ))
    }

    // MARK: — Message List

    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                if viewModel.messages.isEmpty && viewModel.streamingText.isEmpty {
                    emptyStateView
                        .frame(maxWidth: .infinity)
                        .padding(.top, 60)
                } else {
                    LazyVStack(alignment: .leading, spacing: 14) {
                        // Tool results are for the model; showing them put raw memory
                        // lookups ("The Palace is empty") in the chat.
                        ForEach(viewModel.messages.filter { $0.role == .user || $0.role == .assistant }) { message in
                            MessageBubble(
                                message: message,
                                isRPGMode: viewModel.currentWing != nil,
                                personaName: viewModel.currentWing
                            )
                            .id(message.id)
                            .environmentObject(engine)
                            .contextMenu { messageActions(for: message) }
                        }
                        if !viewModel.streamingText.isEmpty || viewModel.thinkingText != nil {
                            StreamingBubble(
                                text: viewModel.streamingText,
                                thinkingText: viewModel.thinkingText,
                                isRPGMode: viewModel.currentWing != nil,
                                personaName: viewModel.currentWing
                            )
                            .id("generating")
                            .environmentObject(engine)
                        }
                        Color.clear.frame(height: 1).id("bottom")
                    }
                    .padding(.horizontal, 14)
                    .padding(.top, 12)
                    .padding(.bottom, 8)
                }
            }
            .scrollDismissesKeyboard(.interactively)
            .onTapGesture { inputFocused = false }
            .onChange(of: viewModel.streamingText) { _, _ in
                withAnimation(.easeOut(duration: 0.1)) {
                    proxy.scrollTo("bottom")
                }
            }
        }
    }

    // MARK: — Empty State

    @ViewBuilder
    private var emptyStateView: some View {
        switch engine.state {

        case .downloading(let progress, let speed):
            VStack(spacing: 16) {
                DownloadAnimationView(progress: progress, speed: speed)
                Button("Cancel Download", role: .destructive) {
                    engine.downloadManager.cancelAllDownloads()
                }
                .buttonStyle(.bordered)
            }

        case .loading(let progress, let stage):
            VStack(spacing: 16) {
                ZStack {
                    Circle()
                        .stroke(SwiftBuddyTheme.accent.opacity(0.15), lineWidth: 3)
                        .frame(width: 64, height: 64)
                    ProgressView(value: progress)
                        .controlSize(.large)
                        .tint(SwiftBuddyTheme.accent)
                        .frame(width: 64)
                }
                VStack(spacing: 4) {
                    Text("Loading model into Metal GPU…")
                        .font(.subheadline)
                        .foregroundStyle(SwiftBuddyTheme.textSecondary)
                    Text(stage)
                        .font(.caption)
                        .foregroundStyle(SwiftBuddyTheme.textTertiary)
                    Text("\(Int(progress * 100))%")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(SwiftBuddyTheme.textTertiary)
                }
            }

        case .idle:
            idleEmptyState

        case .error(let msg):
            VStack(spacing: 14) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 44))
                    .foregroundStyle(SwiftBuddyTheme.error)
                Text(msg)
                    .font(.headline)
                    .foregroundStyle(SwiftBuddyTheme.textPrimary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 40)
                if let detail = engine.errorDetail {
                    if showErrorDetail {
                        Text(detail)
                            .font(.caption)
                            .foregroundStyle(SwiftBuddyTheme.textSecondary)
                            .multilineTextAlignment(.center)
                            .textSelection(.enabled)
                            .padding(.horizontal, 40)
                    }
                    Button(showErrorDetail ? "Hide details" : "Details") {
                        showErrorDetail.toggle()
                    }
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(SwiftBuddyTheme.accent)
                    .buttonStyle(.plain)
                }

                if engine.corruptedModelId != nil {
                    Button(role: .destructive) {
                        engine.deleteCorruptedAndRedownload()
                    } label: {
                        Label("Delete Corrupted Model & Re-download", systemImage: "arrow.down.app.fill")
                            .font(.subheadline.weight(.semibold))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(SwiftBuddyTheme.error.opacity(0.15))
                    .foregroundStyle(SwiftBuddyTheme.error)
                    .padding(.top, 10)
                }
            }

        case .ready, .generating:
            VStack(spacing: 14) {
                // Brand mark
                brandMark
                Text("Start a conversation")
                    .font(.headline)
                    .foregroundStyle(SwiftBuddyTheme.textPrimary)
                Text("Type a message below to begin.")
                    .font(.subheadline)
                    .foregroundStyle(SwiftBuddyTheme.textSecondary)
            }
        }
    }

    // Brand mark — animated bolt in gradient ring
    private var brandMark: some View {
        ZStack {
            Circle()
                .fill(SwiftBuddyTheme.heroGradient)
                .frame(width: 80, height: 80)
                .shadow(color: SwiftBuddyTheme.accent.opacity(0.35), radius: 18)

            Image(systemName: "bolt.fill")
                .font(.system(size: 34, weight: .semibold))
                .foregroundStyle(
                    LinearGradient(colors: [.white, SwiftBuddyTheme.cyan],
                                   startPoint: .top, endPoint: .bottom)
                )
        }
    }

    // Idle empty state — brand mark + tagline
    private var idleEmptyState: some View {
        VStack(spacing: 20) {
            brandMark

            VStack(spacing: 6) {
                Text(viewModel.currentWing != nil ? "System Linked to \(viewModel.currentWing!)" : "SwiftBuddy Chat")
                    .font(.title2.weight(.bold))
                    .foregroundStyle(SwiftBuddyTheme.textPrimary)

                Text("Run any model. Locally. Instantly.")
                    .font(.subheadline)
                    .foregroundStyle(
                        LinearGradient(
                            colors: [SwiftBuddyTheme.accent, SwiftBuddyTheme.cyan],
                            startPoint: .leading, endPoint: .trailing
                        )
                    )
            }

            Text("Go to the **Models** tab to download\na model and start chatting.")
                .font(.caption)
                .foregroundStyle(SwiftBuddyTheme.textTertiary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
        }
    }


    // MARK: — Engine Banner (slim status strip above input)

    @ViewBuilder
    private var engineBanner: some View {
        switch engine.state {
        case .idle:
            bannerRow(icon: "cpu", text: "No model loaded", color: SwiftBuddyTheme.textTertiary)
        case .loading(let progress, let stage):
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(stage)
                        .font(.caption.weight(.medium))
                        .foregroundStyle(SwiftBuddyTheme.textSecondary)
                    Spacer()
                    Text("\(Int(progress * 100))%")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(SwiftBuddyTheme.textTertiary)
                }
                ProgressView(value: progress).tint(SwiftBuddyTheme.accent)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(SwiftBuddyTheme.surface.opacity(0.90))
        case .downloading(let p, let speed):
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Downloading…")
                        .font(.caption.weight(.medium))
                        .foregroundStyle(SwiftBuddyTheme.textSecondary)
                    Spacer()
                    Text("\(Int(p * 100))% · \(speed)")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(SwiftBuddyTheme.textTertiary)
                    Button("Cancel") { engine.downloadManager.cancelAllDownloads() }
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(SwiftBuddyTheme.error)
                }
                ProgressView(value: p).tint(SwiftBuddyTheme.accent)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(SwiftBuddyTheme.surface.opacity(0.90))
        case .error(let msg):
            if engine.corruptedModelId != nil {
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(SwiftBuddyTheme.error)
                        Text(msg)
                            .font(.caption)
                            .foregroundStyle(SwiftBuddyTheme.textPrimary)
                            .lineLimit(1)
                    }
                    Button(role: .destructive) {
                        engine.deleteCorruptedAndRedownload()
                    } label: {
                        Label("Delete Corrupted Model & Re-download", systemImage: "arrow.down.app.fill")
                            .font(.caption.weight(.semibold))
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(SwiftBuddyTheme.error.opacity(0.2))
                    .foregroundStyle(SwiftBuddyTheme.error)
                    .controlSize(.small)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(SwiftBuddyTheme.error.opacity(0.1))
            } else {
                bannerRow(icon: "exclamationmark.triangle.fill", text: msg, color: SwiftBuddyTheme.error)
            }
        case .generating where engine.generationOrigin == .api && !viewModel.isGenerating:
            // Sending is disabled until it finishes; say why.
            bannerRow(icon: "network", text: "Answering an API request…", color: SwiftBuddyTheme.accent)
        case .ready, .generating:
            if engine.maxContextWindow > 0 && !viewModel.messages.isEmpty {
                HStack {
                    Spacer()
                    let percent = Double(engine.activeContextTokens) / Double(engine.maxContextWindow)
                    let displayColor: Color = percent > 0.85 ? SwiftBuddyTheme.error : (percent > 0.6 ? SwiftBuddyTheme.warning : SwiftBuddyTheme.textTertiary)
                    Text("Context: \(engine.activeContextTokens) / \(engine.maxContextWindow)")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(displayColor)
                }
                .padding(.horizontal, 16)
                .padding(.top, 2)
            } else {
                EmptyView()
            }
        }
    }

    private func bannerRow(icon: String, text: String, color: Color) -> some View {
        HStack(spacing: 8) {
            Image(systemName: icon).foregroundStyle(color)
            Text(text)
                .font(.caption)
                .foregroundStyle(color)
                .lineLimit(1)
            Spacer()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(SwiftBuddyTheme.surface.opacity(0.90))
    }

    // MARK: — Input Bar

    private var inputBar: some View {
        HStack(alignment: .bottom, spacing: 10) {
            
            // Text field with frosted glass pill
            HStack(alignment: .bottom) {
                TextField(viewModel.currentWing != nil ? "Message \(viewModel.currentWing!)..." : "Message", text: $inputText, axis: .vertical)
                    .textFieldStyle(.plain)
                    .font(.system(.body))
                    .foregroundStyle(SwiftBuddyTheme.textPrimary)
                    .lineLimit(1...8)
                    .focused($inputFocused)
                    .onSubmit {
                        #if os(macOS)
                        sendMessage()
                        #endif
                    }
                    .disabled(!engine.state.canSend)
                    .accentColor(SwiftBuddyTheme.accent)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(.ultraThinMaterial)
            .background(SwiftBuddyTheme.surface.opacity(0.70))
            .clipShape(RoundedRectangle(cornerRadius: SwiftBuddyTheme.radiusXL))
            .overlay(
                RoundedRectangle(cornerRadius: SwiftBuddyTheme.radiusXL)
                    .strokeBorder(
                        inputFocused
                            ? SwiftBuddyTheme.accent.opacity(0.55)
                            : SwiftBuddyTheme.subtleFill,
                        lineWidth: inputFocused ? 1.5 : 1
                    )
                    .animation(SwiftBuddyTheme.quickSpring, value: inputFocused)
            )
            .glowRing(active: inputFocused)

            // Send / Stop button
            if viewModel.isGenerating {
                Button(action: viewModel.stopGeneration) {
                    ZStack {
                        Circle()
                            .fill(SwiftBuddyTheme.error.opacity(0.18))
                            .frame(width: 40, height: 40)
                        Image(systemName: "stop.fill")
                            .font(.system(size: 14, weight: .bold))
                            .foregroundStyle(SwiftBuddyTheme.error)
                    }
                }
                .buttonStyle(.plain)
            } else {
                Button(action: sendMessage) {
                    ZStack {
                        Circle()
                            .fill(canSend ? AnyShapeStyle(SwiftBuddyTheme.userBubbleGradient) : AnyShapeStyle(SwiftBuddyTheme.subtleFill))
                            .frame(width: 40, height: 40)
                        Image(systemName: "arrow.up")
                            .font(.system(size: 15, weight: .bold))
                            .foregroundStyle(canSend ? .white : SwiftBuddyTheme.textTertiary)
                    }
                }
                .buttonStyle(.plain)
                .disabled(!canSend)
                .keyboardShortcut(.return, modifiers: .command)
                .animation(SwiftBuddyTheme.quickSpring, value: canSend)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(SwiftBuddyTheme.background.opacity(0.95))
    }

    private var canSend: Bool {
        !inputText.trimmingCharacters(in: .whitespaces).isEmpty && engine.state.canSend
    }

    private func sendMessage() {
        let text = inputText.trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty, !viewModel.isGenerating else { return }
        inputText = ""
        Task { await viewModel.send(text) }
    }

    /// Long-press menu on a message: copy it, edit and resend your own messages, or get
    /// a new reply. Editing and resending drop everything after that message.
    @ViewBuilder
    private func messageActions(for message: ChatMessage) -> some View {
        Button {
            #if os(iOS)
            UIPasteboard.general.string = message.content
            #else
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(message.content, forType: .string)
            #endif
        } label: {
            Label("Copy", systemImage: "doc.on.doc")
        }
        if message.role == .user {
            Button {
                inputText = viewModel.editMessage(message)
                inputFocused = true
            } label: {
                Label("Edit", systemImage: "pencil")
            }
            Button {
                Task { await viewModel.resend(message) }
            } label: {
                Label("Resend", systemImage: "arrow.clockwise")
            }
            .disabled(!engine.state.canSend)
        } else {
            Button {
                Task { await viewModel.regenerate(message) }
            } label: {
                Label("Regenerate", systemImage: "arrow.clockwise")
            }
            .disabled(!engine.state.canSend)
        }
    }

    // MARK: — Toolbars

    #if os(iOS)
    @ToolbarContentBuilder
    private var iOSToolbar: some ToolbarContent {
        // Animated status pill (center)
        ToolbarItem(placement: .principal) {
            modelStatusPill
        }
        // Keyboard dismiss
        ToolbarItem(placement: .topBarLeading) {
            if inputFocused {
                Button { inputFocused = false } label: {
                    Image(systemName: "keyboard.chevron.compact.down")
                        .foregroundStyle(SwiftBuddyTheme.textSecondary)
                }
                .transition(.opacity)
            }
        }
        // Persona map selector
        ToolbarItem(placement: .topBarTrailing) {
            Menu {
                // Open that persona's own chat. Setting only the persona kept the
                // current chat's messages on screen under a different prompt.
                Button("No Persona") { viewModel.openConversation(wing: nil) }
                Divider()
                ForEach(wings) { wing in
                    Button(wing.name) { viewModel.openConversation(wing: wing.name) }
                }
            } label: {
                Image(systemName: viewModel.currentWing == nil ? "brain" : "brain.head.profile")
                    .foregroundStyle(viewModel.currentWing == nil ? SwiftBuddyTheme.textSecondary : .orange)
            }
        }
        
        // New chat and earlier chats. There used to be one chat per persona for good;
        // the trash button below still clears the open one.
        ToolbarItem(placement: .topBarTrailing) {
            Menu {
                Button { viewModel.startNewChat() } label: {
                    Label("New Chat", systemImage: "square.and.pencil")
                }
                if !chatHistory.isEmpty {
                    Section("Recent Chats") {
                        ForEach(chatHistory) { session in
                            Button { viewModel.openSession(session) } label: {
                                if session.id == viewModel.activeSessionID {
                                    Label(chatTitle(session), systemImage: "checkmark")
                                } else {
                                    Text(chatTitle(session))
                                }
                            }
                        }
                    }
                    Section {
                        Button(role: .destructive) { confirmDeleteAllChats = true } label: {
                            Label("Delete All Chats…", systemImage: "trash.slash")
                        }
                    }
                }
            } label: {
                Image(systemName: "square.and.pencil")
                    .foregroundStyle(SwiftBuddyTheme.textSecondary)
            }
        }

        // Delete this chat (asks first; it used to clear the chat with no confirmation)
        ToolbarItem(placement: .topBarTrailing) {
            Button { confirmDeleteChat = true } label: {
                Image(systemName: "trash")
                    .foregroundStyle(SwiftBuddyTheme.textSecondary)
            }
        }
    }

    private var modelStatusPill: some View {
        HStack(spacing: 5) {
            if case .generating = engine.state {
                GeneratingDots()
            } else {
                Circle()
                    .fill(engine.state.statusColor)
                    .frame(width: 7, height: 7)
            }
            Text(statusPillLabel)
                .font(.caption.weight(.medium))
                .foregroundStyle(SwiftBuddyTheme.textPrimary)
                .lineLimit(1)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 5)
        .background(.ultraThinMaterial)
        .background(SwiftBuddyTheme.surface.opacity(0.70))
        .clipShape(Capsule())
        .overlay(Capsule().strokeBorder(SwiftBuddyTheme.cardStroke, lineWidth: 1))
    }
    #endif

    /// This persona's (or the core chat's) earlier chats with something in them, newest first.
    private var chatHistory: [ChatSession] {
        Array(sessions.lazy
            .filter { $0.wingName == viewModel.currentWing && !$0.turns.isEmpty }
            .prefix(20))
    }

    /// The chat's first message, shortened, and when it was started.
    private func chatTitle(_ session: ChatSession) -> String {
        let first = session.turns
            .filter { $0.roleRaw == "user" }
            .min { $0.timestamp < $1.timestamp }?
            .content.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let short = first.isEmpty ? "Chat" : (first.count > 40 ? String(first.prefix(40)) + "…" : first)
        return "\(short) · \(session.createdAt.formatted(date: .abbreviated, time: .shortened))"
    }

    /// While generating, keep the model's name in view and say when the request came
    /// from the local API server rather than this chat.
    private var statusPillLabel: String {
        guard case .generating = engine.state else { return engine.state.shortLabel }
        let name = engine.loadedModelId.map { $0.components(separatedBy: "/").last ?? $0 } ?? "Model"
        return engine.generationOrigin == .api ? "\(name) · API request" : "\(name) · Generating"
    }

    #if os(macOS)
    @ToolbarContentBuilder
    private var macOSToolbar: some ToolbarContent {
        ToolbarItem {
            Button { withAnimation { viewModel.clearHistory() } } label: {
                Label("Clear Chat", systemImage: "trash")
            }
        }
        ToolbarItem {
            Button { showSettings?.wrappedValue = true } label: {
                Label("Settings", systemImage: "slider.horizontal.3")
            }
        }
        
    }
    #endif
}

/// The two delete confirmations, kept out of `ChatView.body`'s modifier chain so the
/// compiler can still type-check it in reasonable time.
private struct ChatDeletionDialogs: ViewModifier {
    @Binding var confirmDeleteChat: Bool
    @Binding var confirmDeleteAllChats: Bool
    let deleteChat: () -> Void
    let deleteAllChats: () -> Void

    func body(content: Content) -> some View {
        content
            .confirmationDialog("Delete this chat?", isPresented: $confirmDeleteChat, titleVisibility: .visible) {
                Button("Delete Chat", role: .destructive, action: deleteChat)
            } message: {
                Text("Its messages are removed for good.")
            }
            .confirmationDialog("Delete all chats?", isPresented: $confirmDeleteAllChats, titleVisibility: .visible) {
                Button("Delete All Chats", role: .destructive, action: deleteAllChats)
            } message: {
                Text("Every chat with this persona is removed for good.")
            }
    }
}

// MARK: — ModelState Extensions

extension ModelState {
    var canSend: Bool {
        if case .ready = self { return true }
        return false
    }

    var statusColor: Color {
        switch self {
        case .idle:                       return SwiftBuddyTheme.textTertiary
        case .loading, .downloading:      return SwiftBuddyTheme.warning
        case .ready:                      return SwiftBuddyTheme.success
        case .generating:                 return SwiftBuddyTheme.accent
        case .error:                      return SwiftBuddyTheme.error
        }
    }

    var shortLabel: String {
        switch self {
        case .idle:                        return "No model"
        case .loading(let progress, _):    return "\(Int(progress * 100))% loading"
        case .downloading(let p, _):       return "\(Int(p * 100))% downloading"
        case .ready(let modelId):          return modelId.components(separatedBy: "/").last ?? modelId
        case .generating:                  return "Generating"
        case .error:                       return "Error"
        }
    }
}
import SwiftUI

struct DownloadAnimationView: View {
    let progress: Double
    let speed: String
    
    @State private var isAnimating = false
    @State private var textFlicker = false
    
    var body: some View {
        VStack(spacing: 30) {
            ZStack {
                // Background Ambient Glow
                Circle()
                    .fill(SwiftBuddyTheme.accent.opacity(0.1))
                    .frame(width: 140, height: 140)
                    .blur(radius: isAnimating ? 20 : 10)
                
                // Outer Runic Circle (Dashed, Rotating Clockwise)
                Circle()
                    .stroke(style: StrokeStyle(lineWidth: 1, dash: [4, 8, 2, 8]))
                    .foregroundStyle(SwiftBuddyTheme.accent.opacity(0.4))
                    .frame(width: 130, height: 130)
                    .rotationEffect(.degrees(isAnimating ? 360 : 0))
                    .animation(
                        .linear(duration: 20).repeatForever(autoreverses: false),
                        value: isAnimating
                    )
                
                // Middle Ritual Circle (Thick Dashed, Rotating Counter-Clockwise)
                Circle()
                    .stroke(style: StrokeStyle(lineWidth: 2, dash: [10, 5, 2, 5]))
                    .foregroundStyle(SwiftBuddyTheme.accent.opacity(0.6))
                    .frame(width: 100, height: 100)
                    .rotationEffect(.degrees(isAnimating ? -360 : 0))
                    .animation(
                        .linear(duration: 15).repeatForever(autoreverses: false),
                        value: isAnimating
                    )
                
                // Dynamic Completion Progress Arc (Liquid Arc filling up)
                Circle()
                    .trim(from: 0, to: progress)
                    .stroke(
                        SwiftBuddyTheme.avatarGradient,
                        style: StrokeStyle(lineWidth: 4, lineCap: .round)
                    )
                    .frame(width: 115, height: 115)
                    .rotationEffect(.degrees(-90))
                    .animation(.spring(response: 0.5, dampingFraction: 0.8), value: progress)
                    .shadow(color: SwiftBuddyTheme.accent, radius: progress > 0 ? 5 : 0)
                
                // Core "Persona Soul" Crystal
                Image(systemName: "diamond.inset.filled")
                    .resizable()
                    .scaledToFit()
                    .frame(width: 36, height: 36)
                    .foregroundStyle(SwiftBuddyTheme.cyan)
                    .symbolEffect(.pulse, options: .repeating)
                    .shadow(color: SwiftBuddyTheme.cyan, radius: isAnimating ? 15 : 5)
                    .scaleEffect(isAnimating ? 1.1 : 0.9)
                    .animation(
                        .easeInOut(duration: 1.5).repeatForever(autoreverses: true),
                        value: isAnimating
                    )
            }
            .frame(width: 150, height: 150)
            .padding(.top, 20)
            
            // Decrypting Text Area
            VStack(spacing: 8) {
                Text("SUMMONING PERSONA")
                    .font(.system(.subheadline, design: .monospaced, weight: .bold))
                    .tracking(4)
                    .foregroundStyle(SwiftBuddyTheme.cyan)
                    .shadow(color: SwiftBuddyTheme.cyan.opacity(0.5), radius: 2)
                    .opacity(textFlicker ? 0.8 : 1.0)
                    .animation(.randomFlicker, value: textFlicker)
                
                HStack(alignment: .lastTextBaseline, spacing: 4) {
                    Text("\(Int(progress * 100))")
                        .font(.system(size: 32, design: .monospaced))
                        .fontWeight(.heavy)
                        .foregroundStyle(SwiftBuddyTheme.textPrimary)
                    Text("%")
                        .font(.system(size: 20, design: .monospaced))
                        .foregroundStyle(SwiftBuddyTheme.textSecondary)
                }
                
                Text(speed.isEmpty ? "Casting initial logic runes..." : speed)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(SwiftBuddyTheme.accent)
                    .opacity(0.8)
            }
        }
        .onAppear {
            isAnimating = true
            textFlicker = true
        }
    }
}

// Helper for "cryptographic" flickering effect on the Summon banner
extension Animation {
    static var randomFlicker: Animation {
        .easeInOut(duration: 0.1).repeatForever(autoreverses: true).delay(Double.random(in: 0...0.5))
    }
}
