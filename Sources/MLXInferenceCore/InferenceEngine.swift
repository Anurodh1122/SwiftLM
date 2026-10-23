// InferenceEngine.swift — Core MLX inference engine for SwiftLM Chat
// Handles: model load/unload, token streaming, memory/thermal pressure response.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM
import Hub
import Tokenizers
#if canImport(UIKit)
import UIKit
#endif

// MARK: — Hub Downloader bridge (Downloader protocol conformance over HubApi)

private struct HubDownloader: Downloader, Sendable {
    let hub: HubApi
    func download(
        id: String, revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        try await hub.snapshot(
            from: id,
            matching: patterns,
            progressHandler: progressHandler)
    }
}

// MARK: — swift-transformers TokenizerLoader bridge

private struct TransformersTokenizerLoader: TokenizerLoader, Sendable {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let upstream = try await AutoTokenizer.from(modelFolder: directory)
        return TransformersTokenizerBridge(upstream)
    }
}

private struct TransformersTokenizerBridge: MLXLMCommon.Tokenizer, Sendable {
    let upstream: any Tokenizers.Tokenizer
    init(_ upstream: any Tokenizers.Tokenizer) { self.upstream = upstream }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }
    func convertTokenToId(_ token: String) -> Int? { upstream.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { upstream.convertIdToToken(id) }
    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        do {
            return try upstream.applyChatTemplate(
                messages: messages, tools: tools, additionalContext: additionalContext)
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            throw MLXLMCommon.TokenizerError.missingChatTemplate
        }
    }
}


// MARK: — Model State

public enum ModelState: Equatable, Sendable {
    case idle
    case downloading(progress: Double, speed: String)
    case loading(progress: Double, stage: String)
    case ready(modelId: String)
    case generating
    case error(String)
}

/// Where a generation request came from.
public enum GenerationOrigin: Sendable {
    case app
    case api
}

// MARK: — Thermal State

public enum ThermalLevel: Sendable {
    case nominal, fair, serious, critical
    public var displayString: String {
        switch self {
        case .nominal: return "Normal"
        case .fair:    return "Warm"
        case .serious: return "Hot — generation may be slow"
        case .critical: return "Critical — generation paused"
        }
    }
    public var isThrottled: Bool { self == .serious || self == .critical }
}

// MARK: — Generation Token

public struct GenerationToken: Sendable {
    public let text: String
    public let isThinking: Bool

    public init(text: String, isThinking: Bool = false) {
        self.text = text
        self.isThinking = isThinking
    }
}

// MARK: — Inference Metrics

/// Live performance counters updated at the end of each generation pass.
public struct InferenceMetrics: Sendable {
    /// Time from first-token request to first decoded token (seconds).
    public var ttft: Double
    /// Prompt / prefill throughput (tokens per second).
    public var prefillToksPerSec: Double
    /// Decode throughput — tokens generated per second after the first token.
    public var decodeToksPerSec: Double
    /// Draft token acceptance rate (if speculative decoding is active, 0.0-1.0).
    public var draftAcceptanceRate: Double?

    public init(ttft: Double, prefillToksPerSec: Double, decodeToksPerSec: Double, draftAcceptanceRate: Double? = nil) {
        self.ttft = ttft
        self.prefillToksPerSec = prefillToksPerSec
        self.decodeToksPerSec = decodeToksPerSec
        self.draftAcceptanceRate = draftAcceptanceRate
    }

    public static let zero = InferenceMetrics(ttft: 0, prefillToksPerSec: 0, decodeToksPerSec: 0, draftAcceptanceRate: nil)
}

// MARK: — InferenceEngine

@MainActor
public final class InferenceEngine: ObservableObject {
    @Published public private(set) var state: ModelState = .idle {
        didSet {
            if case .error = state {} else { errorDetail = nil }
        }
    }
    /// The full reason behind the current `.error` state. The state itself carries only a
    /// one-line summary, so status bars stay short; screens offer this on request.
    @Published public private(set) var errorDetail: String?
    @Published public private(set) var thermalLevel: ThermalLevel = .nominal
    @Published public private(set) var activeContextTokens: Int = 0
    @Published public private(set) var maxContextWindow: Int = 0
    /// Performance counters from the most recent completed generation.
    @Published public private(set) var lastMetrics: InferenceMetrics = .zero

    /// Set when a corrupted/truncated model is detected during inference.
    /// The UI should observe this and offer to delete & re-download.
    @Published public var corruptedModelId: String? = nil

    /// Whether to automatically unload the model when the app backgrounds
    /// and reload it when returning to foreground.
    /// Defaults to true on iOS (prevents jetsam), false on macOS.
    public var autoOffloadOnBackground: Bool = {
        #if os(iOS)
        return true
        #else
        return false
        #endif
    }()

    /// Shared download + storage manager.
    public let downloadManager = ModelDownloadManager()

    private var container: ModelContainer?
    private var currentModelId: String?
    /// The ID of the last model that was successfully loaded. Remains set during .generating.
    public var loadedModelId: String? { currentModelId }
    /// True while a `generate` call is running. Generations take turns on the one loaded
    /// model; the chat, the local API server and tool-call follow-ups all share it.
    private var isGenerationActive = false
    /// Set by `stopGeneration()`/unloading to end the generation that holds the turn.
    /// A flag rather than a task handle, so a request still waiting for its turn can't
    /// take the place of the one actually running.
    private var stopRequested = false
    /// Bumped by every load and unload. A load whose token is no longer current was
    /// superseded (unloaded mid-load, e.g. on a memory warning) and must not publish.
    private var loadToken: UInt64 = 0
    /// True from the start of `load(modelId:)` until it returns. Two loads at once would
    /// share the global SSD-streaming configuration and double peak memory.
    private var isLoadInFlight = false
    /// Who started the generation that currently holds the turn, so the UI can say when
    /// the model is busy answering the local API server rather than the chat.
    @Published public private(set) var generationOrigin: GenerationOrigin = .app
    /// Prompt chunk size while SSD expert streaming is active on iOS; nil otherwise.
    ///
    /// mlx-swift-lm's streaming SwitchGLU keeps a small set of reused buffers for up to
    /// 32 routed experts per layer (its single-token path). A larger batch — any prompt
    /// chunk over `32 / experts_per_token` tokens — takes its prompt path instead, which
    /// holds buffers for every routed expert of every layer until the pass ends. On an
    /// 8 GB iPad (5 GB app ceiling) that killed the app on the first message to
    /// Qwen3.6-35B-A3B. Prefilling in chunks that stay on the bounded path keeps memory
    /// flat, at the cost of a slower prompt.
    private var streamingPrefillStepSize: Int?

    // All NotificationCenter observers collected for clean deregistration
    // nonisolated(unsafe): populated exclusively from MainActor init, read only in deinit
    // after all strong references have dropped — no concurrent access possible.
    // Declared nonisolated(unsafe) to satisfy Swift 6 deinit isolation rules.
    nonisolated(unsafe) private var observers: [NSObjectProtocol] = []

    // Tracks the model ID active before app backgrounding so we can restore it on foreground.
    private var backgroundedModelId: String?
    /// Timestamp of when the app entered background. Used to implement a
    /// grace-period: short background sessions (<30 s) skip the unload cycle.
    private var backgroundedAt: Date?
    private static let backgroundGracePeriod: TimeInterval = 30

    public init() {
        setupPressureHandlers()
    }

    deinit {
        observers.forEach { NotificationCenter.default.removeObserver($0) }
    }

    // MARK: — Pressure Handlers

    private func setupPressureHandlers() {
        #if canImport(UIKit)
        // ── REACTIVE: Memory warning (last resort) ────────────────────────────
        // OS sends this *after* pressure builds. We still handle it as a fallback
        // in case the proactive unload wasn't triggered (e.g. app was already
        // under pressure from another process).
        observers.append(
            NotificationCenter.default.addObserver(
                forName: UIApplication.didReceiveMemoryWarningNotification,
                object: nil, queue: .main
            ) { [weak self] _ in
                Task { @MainActor [weak self] in
                    // Only a loaded, idle model is unloaded. Not mid-stream; not mid-load
                    // (loading a large model raises warnings routinely, and unloading then
                    // tore down SSD streaming and left a half-loaded model); and not when
                    // nothing is loaded — iOS keeps sending warnings after a manual unload,
                    // which used to replace "No model loaded" with an error.
                    guard let self, case .ready = self.state else { return }
                    self.unload()
                    self.fail(
                        "Model unloaded to free memory.",
                        detail: "iOS was running low on memory. Load the model again from the Models tab."
                    )
                }
            }
        )

        // ── PROACTIVE: App entered background ───────────────────────────────
        // didEnterBackground fires ONLY when the user truly leaves the app
        // (home gesture / Lock button). willResignActive fires too broadly:
        // notification banners, screenshots, system alerts — all trigger it.
        observers.append(
            NotificationCenter.default.addObserver(
                forName: UIApplication.didEnterBackgroundNotification,
                object: nil, queue: .main
            ) { [weak self] _ in
                Task { @MainActor [weak self] in
                    guard let self, self.autoOffloadOnBackground else { return }

                    self.backgroundedAt = Date()

                    // Only remember model ID when it was actually loaded.
                    switch self.state {
                    case .ready(let id):  self.backgroundedModelId = id
                    case .generating:     self.backgroundedModelId = self.currentModelId
                    default:              self.backgroundedModelId = nil
                    }

                    self.stopGeneration()
                    self.unload()
                    self.state = .idle
                }
            }
        )

        // ── PROACTIVE: App returning to foreground ───────────────────────────
        // willEnterForeground fires before the app is fully active, giving us
        // time to start reloading before the UI appears.
        observers.append(
            NotificationCenter.default.addObserver(
                forName: UIApplication.willEnterForegroundNotification,
                object: nil, queue: .main
            ) { [weak self] _ in
                Task { @MainActor [weak self] in
                    guard let self, self.autoOffloadOnBackground else { return }

                    // Grace period: if the user was gone for less than 30 seconds
                    // (e.g. a brief app-switcher peek), don't burn time reloading.
                    let elapsed = self.backgroundedAt.map { Date().timeIntervalSince($0) } ?? 999
                    self.backgroundedAt = nil

                    guard elapsed >= Self.backgroundGracePeriod else {
                        // Short absence — stay idle, let the user decide what to do.
                        self.backgroundedModelId = nil
                        return
                    }

                    // Only what was loaded when the app left; nothing loaded means nothing
                    // to restore.
                    let modelToReload = self.backgroundedModelId
                    self.backgroundedModelId = nil
                    if let modelId = modelToReload {
                        await self.load(modelId: modelId)
                    }
                }
            }
        )
        #endif

        // ── Thermal state monitoring (all platforms) ──────────────────────────
        observers.append(
            NotificationCenter.default.addObserver(
                forName: ProcessInfo.thermalStateDidChangeNotification,
                object: nil, queue: .main
            ) { [weak self] _ in
                Task { @MainActor [weak self] in
                    self?.updateThermalLevel()
                }
            }
        )
        updateThermalLevel()
    }

    private func updateThermalLevel() {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal:  thermalLevel = .nominal
        case .fair:     thermalLevel = .fair
        case .serious:  thermalLevel = .serious
        case .critical:
            thermalLevel = .critical
            // Critical: stop any generation immediately
            stopGeneration()
        @unknown default: thermalLevel = .nominal
        }
    }

    // MARK: — Model Loading

    /// Load a model by HuggingFace ID. Downloads if not cached.
    /// Uses ModelStorage.cacheRoot as the HubApi download base.
    /// For MoE models, activates expert streaming via ExpertStreamingConfig so
    /// only active expert weights are resident in RAM during inference.
    public func load(modelId: String) async {
        guard state != .ready(modelId: modelId) else { return }
        // Only critical blocks a load: iPads reach "serious" just from charging while
        // downloading, which made models refuse to load for no visible reason.
        guard thermalLevel != .critical else {
            fail("Device is too hot. Let it cool before loading a model.")
            return
        }
        guard !isLoadInFlight else {
            print("[InferenceEngine] Ignoring load of \(modelId): another load is in progress.")
            return
        }
        isLoadInFlight = true
        defer { isLoadInFlight = false }
        corruptedModelId = nil

        // A path the user pointed the app at directly (e.g. via "Add Local
        // Model…") — never downloaded, never in the HF cache, so the usual
        // verify-or-download flow doesn't apply; go straight to loading it.
        // Resolved once here (not re-checked inside loadVerifiedModel) so the
        // answer can't change between this check and that one.
        if ModelStorage.isLocalDirectoryPath(modelId) {
            await loadVerifiedModel(modelId: modelId, localDirectory: URL(filePath: modelId))
            return
        }

        // A path-shaped modelId (starts with "/") that ISN'T currently a valid
        // directory — most likely a local model whose directory the app last saw
        // is now inaccessible (external drive unplugged, folder moved/renamed).
        // A real HuggingFace repo id is always "org/name" with no leading slash,
        // so this can't misfire on one. Without this check, `lastLoadedModelId`
        // persisting a local path and auto-resuming on launch (SwiftBuddyApp)
        // would fall through to verifyModelIntegrity/downloadThenLoad below and
        // try to download the raw filesystem path as if it were a repo id.
        if modelId.hasPrefix("/") {
            fail(
                "Can't find \"\(URL(filePath: modelId).lastPathComponent)\".",
                detail: "The folder may have moved or its drive isn't connected."
            )
            return
        }

        guard ModelStorage.verifyModelIntegrity(for: modelId) else {
            await downloadThenLoad(modelId: modelId)
            return
        }

        await loadVerifiedModel(modelId: modelId)
    }

    private func downloadThenLoad(modelId: String) async {
        print("[InferenceEngine] Model \(modelId) is missing or incomplete. Starting download before load.")
        releaseLoadedModelResources()
        state = .downloading(progress: 0.0, speed: "Preparing...")

        let task = downloadManager.startDownload(modelId: modelId)

        // The download manager tracks the real progress; mirror it into `state`, which
        // the Chat banner and the Models card show. Without this they sat at 0%
        // "Preparing..." for the whole download.
        let progressMirror = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                if let self, case .downloading = self.state,
                    let progress = self.downloadManager.activeDownloads[modelId]
                {
                    self.state = .downloading(
                        progress: progress.fractionCompleted,
                        speed: progress.speedString.isEmpty ? "Downloading..." : progress.speedString
                    )
                }
                try? await Task.sleep(nanoseconds: 250_000_000)
            }
        }
        defer { progressMirror.cancel() }

        do {
            try await task.value
            progressMirror.cancel()
            state = .downloading(progress: 1.0, speed: "Verifying...")

            guard ModelStorage.verifyModelIntegrity(for: modelId) else {
                markModelCorrupted(
                    modelId: modelId,
                    message: "Model files are incomplete after download. Choose a recovery option."
                )
                return
            }

            await loadVerifiedModel(modelId: modelId)
        } catch is CancellationError {
            state = .idle
        } catch let urlError as URLError where urlError.code == .cancelled {
            // URLSession reports a cancelled transfer this way instead of CancellationError.
            state = .idle
        } catch {
            fail(
                "Couldn't download \(modelId.components(separatedBy: "/").last ?? modelId).",
                detail: error.localizedDescription
            )
        }
    }

    /// - Parameter localDirectory: pre-resolved by `load()` when `modelId` is
    ///   itself a directory path (e.g. picked via "Add Local Model…", possibly on
    ///   an external drive entirely outside `cacheRoot`) — `nil` for every normal
    ///   HuggingFace-id model. Passed in rather than re-derived here so the two
    ///   checks can't disagree if the path's existence changes in between (the
    ///   directory is deleted/unmounted between `load()`'s check and this call).
    ///
    ///   Distinct from `ModelStorage.localLoadDirectory(for:)` below: that's for
    ///   an HF-style "org/name" id whose files were copied by hand into a
    ///   recognised `cacheRoot`-relative layout (issue #110) rather than
    ///   downloaded, resolved through the existing `ModelStorage` id-based
    ///   helpers, which still work for that case unchanged.
    private func loadVerifiedModel(modelId: String, localDirectory: URL? = nil) async {
        // Free the model being replaced before reading the new one. The old container
        // used to stay resident until the new load finished, so the two added up: going
        // from a 4B (2.4 GB) to Qwen3.6-35B-A3B reached 5.1 GB mid-load on an 8 GB iPad
        // and iOS killed the app.
        if container != nil {
            releaseLoadedModelResources()
            MLX.Memory.clearCache()
        }
        loadToken &+= 1
        let token = loadToken
        setLoadingState(progress: 0.05, stage: "Preparing model configuration")
        currentModelId = modelId
        let explicitLocalDirectory = localDirectory

        // If the process is killed mid-load (usually out of memory) this stays set, and
        // the next launch sees it and skips the automatic reload instead of crash-looping.
        UserDefaults.standard.set(modelId, forKey: Self.interruptedLoadKey)
        defer { UserDefaults.standard.removeObject(forKey: Self.interruptedLoadKey) }

        do {
            let hub = HubApi(downloadBase: ModelStorage.cacheRoot)

            // For MoE models, enable expert streaming before loading so
            // loadWeights() initialises ExpertStreamerManager correctly.
            // lazyLoad=true means weights are mmap'd and not paged into RAM
            // at load time — only active expert pages touch RAM during inference.
            // A model the user copied in by hand does not live at the materialized
            // `<cacheRoot>/models/<id>` path that ModelConfiguration(id:) resolves
            // through HubApi. Recognising those layouts in the scan (issue #110) without
            // pointing the loader at them would list the model and then re-download it —
            // several GB for a model already on disk. Load such models by directory.
            var config: ModelConfiguration
            if let explicitLocalDirectory {
                config = ModelConfiguration(directory: explicitLocalDirectory)
            } else if let localDirectory = ModelStorage.localLoadDirectory(for: modelId) {
                config = ModelConfiguration(directory: localDirectory)
            } else {
                config = ModelConfiguration(id: modelId)
            }
            // A local directory never matches a catalog id (the catalog only lists
            // HuggingFace-style ids) — fall back to inspecting its own config.json
            // rather than silently defaulting to "not MoE" and disabling SSD expert
            // streaming for exactly the large-MoE-on-external-drive case this local-
            // directory support exists for.
            let isMoE: Bool
            if let catalogIsMoE = ModelCatalog.all.first(where: { $0.id == modelId })?.isMoE {
                isMoE = catalogIsMoE
            } else if let explicitLocalDirectory,
                let localConfig = ModelStorage.readModelConfig(inDirectory: explicitLocalDirectory)
            {
                isMoE = ModelStorage.configIndicatesMoE(localConfig)
            } else if let storedConfig = ModelStorage.readModelConfig(for: modelId) {
                // Downloaded through HuggingFace search, so not in the catalog either.
                isMoE = ModelStorage.configIndicatesMoE(storedConfig)
            } else {
                isMoE = false
            }
            let generationConfig = GenerationConfig.load()
            if generationConfig.enableMTP {
                setenv("SWIFTLM_MTP_ENABLE", "1", 1)
            } else {
                unsetenv("SWIFTLM_MTP_ENABLE")
            }
            // SSD expert streaming defaults ON for MoE until the user saves a preference.
            // Once persisted, the saved toggle becomes authoritative for all models.
            // Streaming pages MoE expert weights in from storage; a dense model has none to
            // stream. Turning it on for one only switches the library into its streaming mode
            // (lazy weights, per-layer evals skipped) with nothing to stream, and such models
            // then fail to load — so streaming applies to MoE models only.
            let shouldStream = isMoE && generationConfig.effectiveStreamExperts(defaultingTo: isMoE)
            streamingPrefillStepSize = nil
            if shouldStream {
                config.lazyLoad = true
                let modelDir = explicitLocalDirectory ?? ModelStorage.snapshotDirectory(for: modelId)
                ExpertStreamingConfig.shared.activate(
                    modelDirectory: modelDir,
                    useDirectIO: {
                        #if os(macOS)
                        return true
                        #else
                        return false
                        #endif
                    }()
                )
                #if os(iOS)
                let modelConfig = explicitLocalDirectory.flatMap(ModelStorage.readModelConfig(inDirectory:))
                    ?? ModelStorage.readModelConfig(for: modelId)
                let expertsPerToken = modelConfig.flatMap(Self.expertsPerToken(in:)) ?? 8
                streamingPrefillStepSize = max(1, 32 / max(expertsPerToken, 1))
                #endif
                print("[InferenceEngine] SSD expert streaming: enabled (isMoE=\(isMoE), persisted=\(GenerationConfig.hasPersistedConfig), setting=\(generationConfig.streamExperts), prefill chunk=\(streamingPrefillStepSize.map { "\($0)" } ?? "default"))")
            } else {
                print("[InferenceEngine] SSD expert streaming: disabled")
            }

            setLoadingState(progress: 0.15, stage: "Inspecting model architecture")
            let downloader = HubDownloader(hub: hub)
            let architecture = try await ModelArchitectureProbe.inspect(
                configuration: config,
                downloader: downloader
            )
            guard token == loadToken else { return }

            let loadingStage = architecture.supportsVision
                ? "Loading multimodal model"
                : "Loading language model"

            setLoadingState(progress: 0.22, stage: loadingStage)

            let loadedContainer: ModelContainer
            if architecture.supportsVision {
                loadedContainer = try await VLMModelFactory.shared.loadContainer(
                    from: downloader,
                    using: TransformersTokenizerLoader(),
                    configuration: config
                ) { [weak self] progress in
                    Task { @MainActor in
                        guard let self, token == self.loadToken else { return }
                        let pct = progress.fractionCompleted
                        self.setLoadingState(progress: 0.22 + (pct * 0.68), stage: loadingStage)
                    }
                }
            } else {
                loadedContainer = try await LLMModelFactory.shared.loadContainer(
                    from: downloader,
                    using: TransformersTokenizerLoader(),
                    configuration: config
                ) { [weak self] progress in
                    Task { @MainActor in
                        guard let self, token == self.loadToken else { return }
                        let pct = progress.fractionCompleted
                        self.setLoadingState(progress: 0.22 + (pct * 0.68), stage: loadingStage)
                    }
                }
            }
            // Unloaded while loading: drop the result instead of publishing a model the
            // rest of the engine already considers gone.
            guard token == loadToken else { return }
            container = loadedContainer

            downloadManager.clearProgress(modelId: modelId)
            downloadManager.lastLoadedModelId = modelId
            downloadManager.refresh()

            // Verify integrity to catch incomplete downloads before marking as ready.
            // A local directory was already validated once before load() was ever
            // called (see ModelManagementView's "Add Local Model…" flow) — no
            // "delete and re-download" recovery makes sense for a folder outside our
            // cache, so this re-check exists to catch the same class of problem
            // (missing/truncated weights) with a message that doesn't imply that.
            setLoadingState(progress: 0.94, stage: "Verifying model files")
            let integrityOK = explicitLocalDirectory.map(ModelStorage.validateLocalModelDirectory)
                ?? ModelStorage.verifyModelIntegrity(for: modelId)
            guard integrityOK else {
                let message = explicitLocalDirectory != nil
                    ? "Model safetensors files are missing or incomplete in this folder."
                    : "Model safetensors files are incomplete. Please delete and re-download."
                throw NSError(domain: "InferenceEngine", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
            }

            // Read the model's actual max context length from config.json
            setLoadingState(progress: 0.98, stage: "Reading model limits")
            let ctxLen = explicitLocalDirectory.map(ModelStorage.readMaxContextLength(inDirectory:))
                ?? ModelStorage.readMaxContextLength(for: modelId)
            if let ctxLen {
                self.maxContextWindow = ctxLen
                print("[InferenceEngine] Model context window: \(ctxLen) tokens")
            } else {
                self.maxContextWindow = 8192  // conservative fallback for models without explicit limits
                print("[InferenceEngine] No explicit context limit found in config.json, defaulting to 8192")
            }

            #if os(iOS)
            // Cap MLX's reusable-buffer cache. Uncapped, it grows with every generation
            // until iOS kills the app for its memory footprint.
            MLX.Memory.cacheLimit = 256 * 1024 * 1024
            #endif
            state = .ready(modelId: modelId)

        } catch {
            // A superseded load's failure is not the current state's business.
            guard token == loadToken else { return }
            ExpertStreamingConfig.shared.deactivate()
            downloadManager.clearProgress(modelId: modelId)
            // A local directory's modelId is a full absolute path — show just the
            // folder name in the error text, matching how "Downloaded" rows
            // elsewhere in the app already display HF ids trimmed to their last
            // component, rather than a raw POSIX path that wraps awkwardly in a
            // narrow error banner.
            let displayName = explicitLocalDirectory?.lastPathComponent
                ?? (modelId.components(separatedBy: "/").last ?? modelId)
            fail("Couldn't load \(displayName).", detail: error.localizedDescription)

            // If the model is incomplete/corrupted, flag it so the UI shows the "Delete
            // & Re-download" button. markModelCorrupted itself no-ops corruptedModelId
            // for a local directory (see its doc comment) — no guard needed here.
            let nsError = error as NSError
            if nsError.domain == "InferenceEngine" && nsError.code == 1 || Self.isModelCorruptionError(error) {
                markModelCorrupted(
                    modelId: modelId,
                    message: "Model weights are corrupted or incomplete. Choose a recovery option."
                )
                return
            }

            container = nil
            self.maxContextWindow = 0
            self.activeContextTokens = 0
        }
    }

    private static let interruptedLoadKey = "swiftlm.interruptedLoadModelId"

    /// Call once at launch before reloading the last model. Returns true when the previous
    /// run died while loading a model: the model is then not reloaded automatically (that
    /// would crash again and trap the user in a loop) and an explanation is shown instead.
    @discardableResult
    public func skipAutoLoadAfterInterruptedLaunch() -> Bool {
        let defaults = UserDefaults.standard
        guard let modelId = defaults.string(forKey: Self.interruptedLoadKey) else { return false }
        defaults.removeObject(forKey: Self.interruptedLoadKey)
        downloadManager.lastLoadedModelId = nil
        let name = ModelStorage.isLocalDirectoryPath(modelId)
            ? URL(filePath: modelId).lastPathComponent
            : (modelId.components(separatedBy: "/").last ?? modelId)
        fail(
            "Didn't reload \(name) automatically.",
            detail: "The app stopped while loading it last time, most likely because it ran out of "
                + "memory. Try a smaller model, or turn on SSD Streaming for MoE models."
        )
        return true
    }

    /// Enters `.error` with a one-line `summary` for status bars and keeps the full
    /// reason in `errorDetail`.
    private func fail(_ summary: String, detail: String? = nil) {
        state = .error(summary)
        errorDetail = detail
        print("[InferenceEngine] \(summary)" + (detail.map { " \($0)" } ?? ""))
    }

    /// Unload the current model and free all GPU memory.
    public func unload() {
        releaseLoadedModelResources()
        corruptedModelId = nil
        state = .idle
    }

    /// UserDefaults key for the context limit in tokens; 0 or unset means no limit.
    /// Kept out of `GenerationConfig`: a new field there would make previously saved
    /// configs fail to decode and silently reset every setting to its default.
    public static let contextLimitKey = "swiftlm.contextLimit"

    /// Longest prompt, in tokens, before the oldest messages are dropped (0 = no limit).
    fileprivate static var contextLimit: Int {
        max(0, UserDefaults.standard.integer(forKey: contextLimitKey))
    }

    /// A parsed tool call as the `<tool_call>` text block the model wrote, in the
    /// `{"name": …, "parameters": …}` shape the tool instructions ask for.
    fileprivate static func toolCallText(_ call: MLXLMCommon.ToolCall) -> String {
        struct Payload: Encodable {
            let name: String
            let parameters: [String: MLXLMCommon.JSONValue]
        }
        let payload = Payload(name: call.function.name, parameters: call.function.arguments)
        let json = (try? JSONEncoder().encode(payload)).flatMap { String(data: $0, encoding: .utf8) }
            ?? #"{"name": "\#(call.function.name)"}"#
        return "<tool_call>\n\(json)\n</tool_call>"
    }

    /// `num_experts_per_tok` from a model config, top level or inside `text_config`.
    private static func expertsPerToken(in config: [String: Any]) -> Int? {
        if let value = config["num_experts_per_tok"] as? Int { return value }
        return (config["text_config"] as? [String: Any])?["num_experts_per_tok"] as? Int
    }

    private func releaseLoadedModelResources() {
        if isGenerationActive { stopRequested = true }
        loadToken &+= 1
        streamingPrefillStepSize = nil
        container = nil
        currentModelId = nil
        maxContextWindow = 0
        activeContextTokens = 0
        ExpertStreamingConfig.shared.deactivate()
        MLX.Memory.cacheLimit = 0
    }

    private func setLoadingState(progress: Double, stage: String) {
        state = .loading(progress: min(max(progress, 0), 1), stage: stage)
    }

    /// Flags a model as corrupted so the UI offers "Delete & Re-download" — except
    /// for a local directory, where that recovery makes no sense: there's no repo
    /// to re-download from, and delete would silently no-op anyway
    /// (`ModelStorage.delete` only ever resolves paths under `cacheRoot`, so an
    /// external-drive path never matches anything it would try to remove). This
    /// guard lives here, not at each call site, so every current and future
    /// caller gets it — a per-call-site guard is easy to add to one call and
    /// forget on the others, which is exactly what happened before this was
    /// centralized (three generation-time call sites had no guard while the
    /// load-time one did).
    private func markModelCorrupted(modelId: String?, message: String) {
        let failedModelId = modelId ?? currentModelId
        releaseLoadedModelResources()
        fail(message)
        if let failedModelId, ModelStorage.isLocalDirectoryPath(failedModelId) {
            corruptedModelId = nil
        } else {
            corruptedModelId = failedModelId
        }
    }

    private static func isModelCorruptionError(_ error: Error) -> Bool {
        let description = error.localizedDescription.lowercased()
        return description.contains("ssd streaming")
            || description.contains("pread")
            || description.contains("safetensors")
            || description.contains("corrupt")
            || description.contains("incomplete")
    }

    public func clearCorruptionRecovery() {
        corruptedModelId = nil
        if case .error = state {
            state = .idle
        }
    }

    // MARK: — Generation
}

// MARK: — Helpers

/// Removes all `<think>…</think>` spans from `text`, including the closing tag's
/// trailing newline when present.  Used to sanitise assistant history messages
/// before they are re-submitted to the Jinja chat-template renderer on subsequent
/// turns — Qwen3 (and similar "thinking" models) raise TemplateException error 1
/// when prior assistant turns contain raw thinking tags.
///
/// Trimming is applied only when at least one tag span was actually removed so
/// that assistant messages without thinking content are returned byte-for-byte
/// (preserving leading spaces, code-block indentation, etc.).
func stripThinkingTags(from text: String) -> String {
    var result = text
    var stripped = false
    // Templates that pre-fill `<think>` in the prompt make the model's output start
    // inside the thought, so the reply holds a closing tag with no opening one.
    if let closeRange = result.range(of: "</think>"),
        result.range(of: "<think>", range: result.startIndex..<closeRange.lowerBound) == nil
    {
        stripped = true
        result.removeSubrange(result.startIndex..<closeRange.upperBound)
    }
    while let openRange = result.range(of: "<think>") {
        stripped = true
        if let closeRange = result.range(of: "</think>", range: openRange.lowerBound..<result.endIndex) {
            // Include the optional newline that immediately follows </think>
            var endIdx = closeRange.upperBound
            if endIdx < result.endIndex && result[endIdx] == "\n" {
                endIdx = result.index(after: endIdx)
            }
            result.removeSubrange(openRange.lowerBound..<endIdx)
        } else {
            // Unclosed <think> — strip from opening tag to end of string
            result.removeSubrange(openRange.lowerBound...)
            break
        }
    }
    // Only trim surrounding whitespace that was introduced by stripping;
    // leave untouched messages that contained no think tags.
    return stripped ? result.trimmingCharacters(in: .whitespacesAndNewlines) : result
}

extension InferenceEngine {
    // MARK: — Generation (continued)

    public nonisolated func generate(
        messages: [ChatMessage],
        config: GenerationConfig = .default,
        origin: GenerationOrigin = .app
    ) -> AsyncStream<GenerationToken> {
        AsyncStream { continuation in
            let task: Task<Void, Never> = Task { @MainActor in
                // One generation at a time: wait for any earlier one (including one a client
                // just abandoned and that is still winding down) instead of running two
                // forward passes on the same model at once.
                while self.isGenerationActive {
                    if Task.isCancelled { continuation.finish(); return }
                    try? await Task.sleep(nanoseconds: 50_000_000)
                }
                if Task.isCancelled { continuation.finish(); return }
                self.isGenerationActive = true
                self.stopRequested = false
                self.generationOrigin = origin
                defer { self.isGenerationActive = false }

                guard let container = self.container else {
                    print("[InferenceEngine] generate (\(origin)): no model loaded, nothing generated")
                    continuation.finish(); return
                }

                // Don't generate when throttled
                if self.thermalLevel == .critical {
                    continuation.yield(GenerationToken(text: "\n\n[Generation paused: device temperature critical]"))
                    continuation.finish(); return
                }

                self.state = .generating

                do {
                    var finalMessages: [[String: String]] = []
                    var pendingSystemContext = ""
                    
                    for msg in messages {
                        if msg.role == .system {
                            pendingSystemContext += msg.content + "\n\n"
                        } else {
                            // Use the canonical role name — Qwen3 (and most models) use
                            // "assistant", not "model". The "model" alias is Gemma-specific
                            // and breaks Qwen3's Jinja chat template on multi-turn history.
                            let roleRaw = msg.role.rawValue  // "user" | "assistant" | "tool"
                            var content = msg.content
                            
                            // Strip <think>…</think> blocks from prior assistant turns.
                            // If the model generated thinking content on a previous turn and
                            // it was not already split into thinkingContent, the raw tags will
                            // be present in `content`. Feeding them back into the Jinja template
                            // on the next request causes TemplateException error 1 on Qwen3.
                            if msg.role == .assistant {
                                content = stripThinkingTags(from: content)
                            }
                            
                            if roleRaw == "user" && !pendingSystemContext.isEmpty {
                                content = "[SYSTEM CONTEXT / PERSONA DATA]\n" + pendingSystemContext + "\n[END CONTEXT]\n\n" + content
                                pendingSystemContext = "" // Clear after injecting
                            }
                            finalMessages.append(["role": roleRaw, "content": content])
                        }
                    }
                    
                    let mlxMessages = finalMessages
                    var params = GenerateParameters(
                        maxTokens: config.maxTokens,
                        kvBits: config.kvBits,
                        kvGroupSize: config.kvGroupSize,
                        temperature: config.temperature,
                        topP: config.topP,
                        topK: config.topK,
                        minP: config.minP,
                        repetitionPenalty: config.repetitionPenalty,
                        prefillStepSize: self.streamingPrefillStepSize.map { min($0, config.prefillSize) }
                            ?? config.prefillSize
                    )
                    params.repetitionContextSize = 20

                    var thinkingActive = false
                    var outputText = ""
                    var tokenCount = 0

                    // ── Metrics timing ──────────────────────────────────────
                    let generationStart = Date()
                    var firstTokenDate: Date? = nil

                    // Set RNG seed for reproducible output when requested.
                    if let seed = config.seed {
                        MLX.seed(seed)
                    }

                    // Pass enable_thinking to the Jinja chat template so the model
                    // actually generates <think> blocks when thinking mode is ON.
                    // Without this kwarg, Qwen3's template defaults to thinking=false
                    // regardless of what the UI toggle shows.
                    let additionalContext: [String: any Sendable]? = config.enableThinking
                        ? ["enable_thinking": true]
                        : ["enable_thinking": false]
                    print("[InferenceEngine] generate (\(origin)): \(messages.count) messages, prefill chunk \(params.prefillStepSize.map { "\($0)" } ?? "default")")
                    var promptMessages = mlxMessages
                    var lmInput = try await container.prepare(
                        input: UserInput(messages: promptMessages, additionalContext: additionalContext))

                    // Context limit (Settings): drop the oldest exchanges until the prompt
                    // fits. The first message is kept, since it carries the system context
                    // and persona, and turns go in pairs so user/assistant still alternate
                    // (strict templates such as Gemma's reject anything else).
                    let contextLimit = Self.contextLimit
                    if contextLimit > 0 {
                        var dropped = 0
                        while lmInput.text.tokens.size > contextLimit, promptMessages.count >= 4 {
                            promptMessages.removeSubrange(1 ... 2)
                            dropped += 2
                            lmInput = try await container.prepare(
                                input: UserInput(messages: promptMessages, additionalContext: additionalContext))
                        }
                        if dropped > 0 {
                            print("[InferenceEngine] context limit \(contextLimit): dropped \(dropped) oldest messages")
                        }
                    }

                    // Use the real token count from the prepared LMInput rather than
                    // a character-length heuristic (which was consistently off by 2–3×
                    // for CJK and code content).
                    let baseTokens = lmInput.text.tokens.size
                    self.activeContextTokens = baseTokens
                    print("[InferenceEngine] prompt: \(baseTokens) tokens")
                    
                    // maxContextWindow is already set during loadModel() from config.json

                    // TurboKV: enable 3-bit PolarQuant+QJL on every KVCacheSimple cache layer.
                    // KVCacheSimple is a cache object (not a neural-network Module), so we
                    // iterate the cache array — mirroring the pattern in Server.swift.
                    let cache = try await container.perform { ctx in try ctx.model.newCache(parameters: params) }
                    if config.turboKV {
                        for layer in cache {
                            if let simple = layer as? KVCacheSimple {
                                simple.turboQuantEnabled = true
                            }
                        }
                        print("[InferenceEngine] TurboKV enabled for this request")
                    }

                    let stream: AsyncStream<Generation> = try await container.perform { ctx in
                        // MTP speculative decoding path: use MTPTokenIterator when
                        //   1. The config requests MTP (enableMTP=true)
                        //   2. The loaded model conforms to MTPLanguageModel
                        if config.enableMTP, ctx.model is (any MTPLanguageModel) {
                            return try MLXLMCommon.generateMTP(
                                input: lmInput,
                                cache: cache,
                                parameters: params,
                                context: ctx,
                                numMTPTokens: config.numMTPTokens
                            )
                        } else {
                            return try MLXLMCommon.generate(
                                input: lmInput,
                                cache: cache,
                                parameters: params,
                                context: ctx
                            )
                        }
                    }

                    var mtpAcceptanceRate: Double? = nil

                    for await generation in stream {
                        guard !Task.isCancelled, !self.stopRequested else { break }

                        if case .chunk(let text, tokenId: _) = generation {
                            // Record time-to-first-token on the very first chunk
                            if firstTokenDate == nil {
                                firstTokenDate = Date()
                                print("[InferenceEngine] first token after \(String(format: "%.1f", Date().timeIntervalSince(generationStart))) s")
                            }

                            outputText += text
                            tokenCount += 1

                            // Update the UI token counter periodically to save CPU
                            if tokenCount % 10 == 0 {
                                self.activeContextTokens = baseTokens + tokenCount
                            }

                            if tokenCount >= config.maxTokens { break }
                            
                            // Hard-stop constraint for Gemma 2/3 and DeepSeek MoE bounds since MLX fails to parse multi-array JSON eos_token_id manifests.
                            if outputText.contains("<end_of_turn>") || outputText.contains("<|im_end|>") || outputText.contains("<|eot_id|>") {
                                let clamped = text.replacingOccurrences(of: "<end_of_turn>", with: "")
                                                  .replacingOccurrences(of: "<|im_end|>", with: "")
                                                  .replacingOccurrences(of: "<|eot_id|>", with: "")
                                continuation.yield(GenerationToken(text: clamped, isThinking: thinkingActive))
                                break
                            }

                            if config.enableThinking {
                                if outputText.contains("<think>") && !outputText.contains("</think>") {
                                    thinkingActive = true
                                } else if outputText.contains("</think>") {
                                    thinkingActive = false
                                }
                            }

                            continuation.yield(GenerationToken(text: text, isThinking: thinkingActive))
                        } else if case .toolCall(let call) = generation {
                            // mlx-swift-lm recognises the model's tool call and emits it as
                            // an event instead of text. The app runs tools from
                            // `<tool_call>` text (ChatViewModel, prompt-based API clients),
                            // and this loop used to read only `.chunk`: the call was dropped
                            // and the reply came out empty. Hand it back as that text.
                            let text = Self.toolCallText(call)
                            print("[InferenceEngine] tool call: \(call.function.name)")
                            if firstTokenDate == nil { firstTokenDate = Date() }
                            outputText += text
                            tokenCount += 1
                            continuation.yield(GenerationToken(text: text))
                        } else if case .rejectedToolCall(let rejected) = generation {
                            // Tool-call-shaped output the library could not parse: pass on
                            // what the model wrote rather than dropping it.
                            print("[InferenceEngine] rejected tool call (\(rejected.reason.rawValue)): \(rejected.detail ?? "")")
                            if !rejected.rawTextPreview.isEmpty {
                                outputText += rejected.rawTextPreview
                                tokenCount += 1
                                continuation.yield(GenerationToken(text: rejected.rawTextPreview))
                            }
                        } else if case .info(let info) = generation {
                            // `proposedDraftTokens`/`acceptedDraftTokens` are `Int?` as of
                            // mlx-swift-lm 348ff97 (nil for non-MTP iterators, renamed from
                            // the previously non-optional `totalDraftTokens`/`acceptedDraftTokens`).
                            if let proposed = info.proposedDraftTokens, proposed > 0,
                                let accepted = info.acceptedDraftTokens
                            {
                                mtpAcceptanceRate = Double(accepted) / Double(proposed)
                            }
                        }
                    }

                    // ── Publish metrics for the completed turn ───────────────
                    let totalElapsed = Date().timeIntervalSince(generationStart)
                    let ttft = firstTokenDate.map { $0.timeIntervalSince(generationStart) } ?? 0
                    // Prefill throughput: prompt tokens / time-to-first-token
                    let prefillTps = (ttft > 0 && baseTokens > 0)
                        ? Double(baseTokens) / ttft
                        : 0
                    // Decode throughput: generated tokens / time spent decoding
                    let decodeElapsed = totalElapsed - ttft
                    let decodeTps = (decodeElapsed > 0 && tokenCount > 1)
                        ? Double(tokenCount - 1) / decodeElapsed
                        : 0
                    print("[InferenceEngine] done: \(tokenCount) tokens, \(outputText.count) chars, \(String(format: "%.1f", decodeTps)) tok/s, cancelled=\(Task.isCancelled), stopped=\(self.stopRequested)")
                    self.lastMetrics = InferenceMetrics(
                        ttft: ttft,
                        prefillToksPerSec: prefillTps,
                        decodeToksPerSec: decodeTps,
                        draftAcceptanceRate: mtpAcceptanceRate
                    )
                } catch let ssdError as SSDStreamingError {
                    // Corrupted/truncated safetensors — surface a clear, actionable error
                    let msg = "Model weights are corrupted or incomplete. Please re-download the model."
                    print("[InferenceEngine] SSD Streaming Error: \(ssdError.localizedDescription)")
                    continuation.yield(GenerationToken(text: "\n\n[Error: \(msg)]"))
                    self.markModelCorrupted(modelId: self.currentModelId, message: msg)
                } catch {
                    print("[InferenceEngine] generation failed: \(error.localizedDescription)")
                    // Check if the generic error is also an SSD streaming issue
                    if Self.isModelCorruptionError(error) {
                        let msg = "Model weights are corrupted or incomplete. Please re-download the model."
                        self.markModelCorrupted(modelId: self.currentModelId, message: msg)
                    }
                    continuation.yield(GenerationToken(text: "\n\n[Error: \(error.localizedDescription)]"))
                }

                if let latchedError = SSDStreamingErrorLatch.shared.consume() {
                    let msg = "Model weights are corrupted or incomplete. Please re-download the model."
                    print("[InferenceEngine] Latched SSD error after generation: \(latchedError.localizedDescription)")
                    self.markModelCorrupted(modelId: self.currentModelId, message: msg)
                } else if case .error = self.state {
                    // Already in error state from catch block above
                } else {
                    self.state = self.currentModelId.map { .ready(modelId: $0) } ?? .idle
                }
                continuation.finish()
            }

            // When the consumer goes away (Stop pressed, chat switched, API client dropped)
            // stop producing; before this the model kept generating to the token limit.
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    public func stopGeneration() {
        if isGenerationActive { stopRequested = true }
        // Only a generation hands the state back to ready; called mid-load (critical heat,
        // backgrounding) this used to report a model as ready before it had loaded.
        if case .generating = state, let id = currentModelId { state = .ready(modelId: id) }
    }

    /// Delete corrupted model files and start a fresh download.
    /// Called from the UI when the user confirms re-download after corruption is detected.
    public func deleteCorruptedAndRedownload() {
        guard let modelId = corruptedModelId else { return }

        releaseLoadedModelResources()
        state = .downloading(progress: 0.0, speed: "Deleting corrupted files...")

        do {
            try ModelStorage.delete(modelId)
            print("[InferenceEngine] Successfully deleted corrupted cache directory for \(modelId).")
        } catch {
            print("[InferenceEngine] FAILED to delete corrupted cache: \(error.localizedDescription)")
            fail("Couldn't delete the corrupted model.", detail: error.localizedDescription)
            return
        }
        downloadManager.refresh()
        corruptedModelId = nil

        print("[InferenceEngine] Deleted corrupted files for \(modelId), starting fresh download")
        Task { @MainActor in
            await downloadThenLoad(modelId: modelId)
        }
    }
}
