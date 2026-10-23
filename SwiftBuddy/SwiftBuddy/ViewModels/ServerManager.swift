import Foundation
import HTTPTypes
import Hummingbird
import NIOCore
#if canImport(MLXInferenceCore)
import MLXInferenceCore
#endif

struct ServerStartupConfiguration: Codable, Equatable, Sendable {
    var autoStart: Bool = true
    var host: String = "127.0.0.1"
    var port: Int = 5413
    var parallelSlots: Int = 1
    var corsOrigin: String = ""
    var apiKey: String = ""

    private static let storageKey = "swiftlm.server.startupConfiguration"

    var normalized: ServerStartupConfiguration {
        var copy = self
        copy.host = copy.host.trimmingCharacters(in: .whitespacesAndNewlines)
        if copy.host.isEmpty { copy.host = "127.0.0.1" }
        copy.port = min(max(copy.port, 1), 65_535)
        copy.parallelSlots = max(copy.parallelSlots, 1)
        copy.corsOrigin = copy.corsOrigin.trimmingCharacters(in: .whitespacesAndNewlines)
        copy.apiKey = copy.apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        return copy
    }

    static func load() -> ServerStartupConfiguration {
        guard let data = UserDefaults.standard.data(forKey: storageKey),
              let decoded = try? JSONDecoder().decode(ServerStartupConfiguration.self, from: data) else {
            return ServerStartupConfiguration()
        }
        return decoded.normalized
    }

    func save() {
        guard let data = try? JSONEncoder().encode(normalized) else { return }
        UserDefaults.standard.set(data, forKey: Self.storageKey)
    }
}

private var swiftBuddyJSONHeaders: HTTPFields {
    HTTPFields([HTTPField(name: .contentType, value: "application/json")])
}

private func swiftBuddyJSONString(_ value: String) -> String {
    guard let data = try? JSONEncoder().encode(value),
          let string = String(data: data, encoding: .utf8) else {
        return "\"\""
    }
    return string
}

private struct SwiftBuddyCORSMiddleware<Context: RequestContext>: RouterMiddleware {
    let allowedOrigin: String

    func handle(_ request: Request, context: Context, next: (Request, Context) async throws -> Response) async throws -> Response {
        if request.method == .options {
            return Response(status: .noContent, headers: corsHeaders(for: request))
        }

        var response = try await next(request, context)
        for field in corsHeaders(for: request) {
            response.headers.append(field)
        }
        return response
    }

    private func corsHeaders(for request: Request) -> HTTPFields {
        var fields: [HTTPField] = []
        if allowedOrigin == "*" {
            fields.append(HTTPField(name: HTTPField.Name("Access-Control-Allow-Origin")!, value: "*"))
        } else {
            let requestOrigin = request.headers[values: HTTPField.Name("Origin")!].first ?? ""
            if requestOrigin == allowedOrigin {
                fields.append(HTTPField(name: HTTPField.Name("Access-Control-Allow-Origin")!, value: allowedOrigin))
                fields.append(HTTPField(name: HTTPField.Name("Vary")!, value: "Origin"))
            }
        }
        fields.append(HTTPField(name: HTTPField.Name("Access-Control-Allow-Methods")!, value: "GET, POST, OPTIONS"))
        fields.append(HTTPField(name: HTTPField.Name("Access-Control-Allow-Headers")!, value: "Content-Type, Authorization, X-SwiftLM-Prefill-Progress"))
        return HTTPFields(fields)
    }
}

private struct SwiftBuddyAPIKeyMiddleware<Context: RequestContext>: RouterMiddleware {
    let apiKey: String

    func handle(_ request: Request, context: Context, next: (Request, Context) async throws -> Response) async throws -> Response {
        let path = request.uri.path
        if path == "/health" || path == "/metrics" {
            return try await next(request, context)
        }

        let authHeader = request.headers[values: .authorization].first ?? ""
        if authHeader == "Bearer \(apiKey)" || authHeader == apiKey {
            return try await next(request, context)
        }

        return Response(
            status: .unauthorized,
            headers: swiftBuddyJSONHeaders,
            body: .init(byteBuffer: ByteBuffer(string: #"{"error":{"message":"Invalid API key","type":"invalid_request_error","code":"invalid_api_key"}}"#))
        )
    }
}

@MainActor
final class ServerManager: ObservableObject {
    @Published var isOnline = false
    @Published var host: String = "127.0.0.1"
    @Published var port: Int = 5413
    @Published private(set) var startupConfiguration: ServerStartupConfiguration
    @Published private(set) var runningConfiguration: ServerStartupConfiguration?
    @Published private(set) var restartRequired = false
    
    // In a real implementation this would hold the Hummingbird App and tie into `engine`
    private var task: Task<Void, Never>?

    init() {
        let configuration = ServerStartupConfiguration.load()
        self.startupConfiguration = configuration
        self.host = configuration.host
        self.port = configuration.port
    }

    func start(engine: InferenceEngine) {
        guard !isOnline else { return }
        let configuration = startupConfiguration.normalized

        task = Task.detached { [weak self] in
            guard let self = self else { return }
            do {
                let router = Router()

                if !configuration.corsOrigin.isEmpty {
                    router.add(middleware: SwiftBuddyCORSMiddleware(allowedOrigin: configuration.corsOrigin))
                }

                if !configuration.apiKey.isEmpty {
                    router.add(middleware: SwiftBuddyAPIKeyMiddleware(apiKey: configuration.apiKey))
                }

                router.get("/health") { _, _ -> Response in
                    let body = """
                    {"status":"ok","message":"SwiftBuddy Local Server","host":\(swiftBuddyJSONString(configuration.host)),"port":\(configuration.port),"parallel":\(configuration.parallelSlots),"cors":\(swiftBuddyJSONString(configuration.corsOrigin.isEmpty ? "disabled" : configuration.corsOrigin)),"auth":"\(configuration.apiKey.isEmpty ? "disabled" : "enabled")"}
                    """
                    let buffer = ByteBuffer(string: body)
                    return Response(status: .ok, headers: swiftBuddyJSONHeaders, body: .init(byteBuffer: buffer))
                }

                // ── /v1/models ─────────────────────────────────────────
                router.get("/v1/models") { _, _ -> Response in
                    let modelId: String
                    switch await engine.state {
                    case .ready(let id): modelId = id
                    default: modelId = "none"
                    }
                    // Use swiftBuddyJSONString to safely escape the model ID —
                    // model IDs with slashes (e.g. "mlx-community/Qwen3") are safe,
                    // but quotes or control chars would break the JSON structure.
                    let body = "{\"object\":\"list\",\"data\":[{\"id\":\(swiftBuddyJSONString(modelId)),\"object\":\"model\",\"owned_by\":\"swiftbuddy\"}]}"
                    return Response(status: .ok, headers: swiftBuddyJSONHeaders,
                                    body: .init(byteBuffer: ByteBuffer(string: body)))
                }

                // ── /v1/chat/completions (OpenAI-compatible, streaming + non-streaming) ──
                router.post("/v1/chat/completions") { request, _ -> Response in
                    // 1. Parse body
                    guard let bodyData = try? await request.body.collect(upTo: 4 * 1024 * 1024),
                          let json = try? JSONSerialization.jsonObject(with: Data(buffer: bodyData)) as? [String: Any]
                    else {
                        let err = #"{"error":{"message":"Invalid JSON body","type":"invalid_request_error"}}"#
                        return Response(status: .badRequest, headers: swiftBuddyJSONHeaders,
                                        body: .init(byteBuffer: ByteBuffer(string: err)))
                    }

                    let streamRequested = json["stream"] as? Bool ?? false

                    // 2. Map messages
                    var chatMessages: [ChatMessage] = []
                    if let msgs = json["messages"] as? [[String: Any]] {
                        for m in msgs {
                            let role    = m["role"]    as? String ?? "user"
                            // Content can be a list of parts ([{"type":"text","text":…}]),
                            // which several clients send; it used to arrive as "".
                            let content: String
                            if let text = m["content"] as? String {
                                content = text
                            } else if let parts = m["content"] as? [[String: Any]] {
                                content = parts.compactMap { $0["text"] as? String }.joined(separator: "\n")
                            } else {
                                content = ""
                            }
                            switch role {
                            case "system", "developer": chatMessages.append(.system(content))
                            case "assistant":           chatMessages.append(.assistant(content))
                            case "tool":                chatMessages.append(.tool(content))
                            case "user":                chatMessages.append(.user(content))
                            default:                    chatMessages.append(.user(content))
                            }
                        }
                    }

                    // 3. Build request config from persisted user defaults + per-request overrides
                    var reqConfig = GenerationConfig.load()
                    if let t  = json["temperature"]       as? Double { reqConfig.temperature        = Float(t) }
                    if let p  = json["top_p"]             as? Double { reqConfig.topP               = Float(p) }
                    if let mt = json["max_tokens"]        as? Int    { reqConfig.maxTokens           = mt }
                    if let rp = json["frequency_penalty"] as? Double { reqConfig.repetitionPenalty  = Float(rp) }
                    // Thinking per request, as vLLM and llama.cpp take it; otherwise the
                    // app's Thinking setting applies.
                    if let kwargs = json["chat_template_kwargs"] as? [String: Any],
                       let think = kwargs["enable_thinking"] as? Bool { reqConfig.enableThinking = think }
                    else if let think = json["enable_thinking"] as? Bool { reqConfig.enableThinking = think }
                    let maxTokens = reqConfig.maxTokens
                    func finishReason(_ usage: (prompt: Int, completion: Int)) -> String {
                        usage.completion >= maxTokens ? "length" : "stop"
                    }

                    // No model: say so instead of answering 200 with an empty reply. A
                    // request that arrives mid-generation still waits its turn.
                    let modelId: String
                    switch await engine.state {
                    case .ready(let id): modelId = id
                    case .generating: modelId = await engine.loadedModelId ?? "local"
                    default:
                        let err = #"{"error":{"message":"No model is loaded in SwiftBuddy.","type":"server_error","code":"model_not_loaded"}}"#
                        return Response(status: .serviceUnavailable, headers: swiftBuddyJSONHeaders,
                                        body: .init(byteBuffer: ByteBuffer(string: err)))
                    }
                    let reqId   = "chatcmpl-\(UUID().uuidString.prefix(8))"
                    let created = Int(Date().timeIntervalSince1970)
                    // Escape model ID once — used in both streaming and non-streaming paths.
                    // Slashes in HF model IDs (e.g. "mlx-community/Qwen3") are safe inside
                    // JSON strings, but quotes/control chars in custom model names would break.
                    let escapedModelId = swiftBuddyJSONString(modelId)

                    // Helper: JSON-safe escape for token text using JSONEncoder so ALL
                    // control chars (U+0000–U+001F) are correctly escaped, not just \n/\r/\t.
                    func jsonEscape(_ s: String) -> String {
                        guard let data = try? JSONEncoder().encode(s),
                              let raw = String(data: data, encoding: .utf8) else { return "\"\"" }
                        // JSONEncoder wraps in outer quotes — strip them for inline interpolation
                        return String(raw.dropFirst().dropLast())
                    }

                    func usageJSON(_ usage: (prompt: Int, completion: Int)) -> String {
                        "\"usage\":{\"prompt_tokens\":\(usage.prompt),\"completion_tokens\":\(usage.completion),\"total_tokens\":\(usage.prompt + usage.completion)}"
                    }

                    if streamRequested {
                        // ── SSE streaming ───────────────────────────────────
                        var sseHeaders = HTTPFields()
                        sseHeaders.append(HTTPField(name: .contentType, value: "text/event-stream; charset=utf-8"))
                        sseHeaders.append(HTTPField(name: HTTPField.Name("Cache-Control")!, value: "no-cache"))
                        sseHeaders.append(HTTPField(name: HTTPField.Name("X-Accel-Buffering")!, value: "no"))

                        let sseStream = AsyncStream<ByteBuffer> { cont in
                            let producer = Task {
                                for await token in await engine.generate(messages: chatMessages, config: reqConfig, origin: .api) {
                                    // The closing token carries the counts. Send them as the
                                    // final chunk with `finish_reason`, as OpenAI does.
                                    if let usage = token.usage {
                                        let chunk = "{\"id\":\"\(reqId)\",\"object\":\"chat.completion.chunk\",\"created\":\(created),\"model\":\(escapedModelId),\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"\(finishReason(usage))\"}],\(usageJSON(usage))}"
                                        cont.yield(ByteBuffer(string: "data: \(chunk)\n\n"))
                                        continue
                                    }
                                    // Thinking goes in `reasoning_content`, as other OpenAI-style
                                    // servers send it; it used to be mixed into the answer.
                                    let field = token.isThinking ? "reasoning_content" : "content"
                                    let chunk = "{\"id\":\"\(reqId)\",\"object\":\"chat.completion.chunk\",\"created\":\(created),\"model\":\(escapedModelId),\"choices\":[{\"index\":0,\"delta\":{\"\(field)\":\"\(jsonEscape(token.text))\"},\"finish_reason\":null}]}"
                                    cont.yield(ByteBuffer(string: "data: \(chunk)\n\n"))
                                }
                                cont.yield(ByteBuffer(string: "data: [DONE]\n\n"))
                                cont.finish()
                            }
                            // Client disconnected: stop generating instead of running to
                            // max_tokens while the chat waits for its turn.
                            cont.onTermination = { _ in producer.cancel() }
                        }
                        return Response(status: .ok, headers: sseHeaders,
                                        body: .init(asyncSequence: sseStream))

                    } else {
                        // ── Non-streaming: collect full response ────────────
                        var fullText = ""
                        var reasoning = ""
                        var usage = (prompt: 0, completion: 0)
                        for await token in await engine.generate(messages: chatMessages, config: reqConfig, origin: .api) {
                            if let counts = token.usage { usage = counts; continue }
                            if token.isThinking { reasoning += token.text } else { fullText += token.text }
                        }
                        let reasoningField = reasoning.isEmpty ? "" : ",\"reasoning_content\":\"\(jsonEscape(reasoning))\""
                        let body = "{\"id\":\"\(reqId)\",\"object\":\"chat.completion\",\"created\":\(created),\"model\":\(escapedModelId),\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"\(jsonEscape(fullText))\"\(reasoningField)},\"finish_reason\":\"\(finishReason(usage))\"}],\(usageJSON(usage))}"
                        return Response(status: .ok, headers: swiftBuddyJSONHeaders,
                                        body: .init(byteBuffer: ByteBuffer(string: body)))
                    }
                }
                
                let app = Application(
                    router: router,
                    configuration: .init(address: .hostname(configuration.host, port: configuration.port))
                )

                await MainActor.run {
                    self.isOnline = true
                    self.host = configuration.host
                    self.port = configuration.port
                    self.runningConfiguration = configuration
                    self.restartRequired = false
                }
                ConsoleLog.shared.info("Server online at http://\(configuration.host):\(configuration.port)")

                try await app.runService()
            } catch {
                print("Server failed: \(error)")
                ConsoleLog.shared.error("Server failed: \(error.localizedDescription)")
                await MainActor.run {
                    self.isOnline = false
                    self.runningConfiguration = nil
                    self.restartRequired = false
                }
            }
        }
    }

    @discardableResult
    func saveStartupConfiguration(_ configuration: ServerStartupConfiguration) -> Bool {
        let normalized = configuration.normalized
        let changed = normalized != startupConfiguration
        startupConfiguration = normalized
        host = normalized.host
        port = normalized.port
        normalized.save()
        restartRequired = isOnline && runningConfiguration != nil && runningConfiguration != normalized
        if changed {
            ConsoleLog.shared.info("Server startup configuration saved")
        }
        return changed
    }

    func restart(engine: InferenceEngine) {
        stop()
        start(engine: engine)
    }

    func stop() {
        task?.cancel()
        task = nil
        isOnline = false
        runningConfiguration = nil
        restartRequired = false
    }
}
