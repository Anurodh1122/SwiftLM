// GraphPalaceService.swift - Synaptic Synthesis Engine
import SwiftUI
import SwiftData
#if canImport(MLXInferenceCore)
import MLXInferenceCore
#endif

public enum GraphSynthesisError: LocalizedError {
    case noModelLoaded
    case nothingToSynthesize(String)

    public var errorDescription: String? {
        switch self {
        case .noModelLoaded: return "Load a model first."
        case .nothingToSynthesize(let wing): return "\(wing) has no memories to build a graph from."
        }
    }
}

@MainActor
public final class GraphPalaceService {
    public static let shared = GraphPalaceService()
    public var modelContext: ModelContext?

    private init() {}

    /// Performs the "2nd round of memory construction" by converting raw SQL memory chunks into topological Graph nodes.
    public func buildRelationalGraph(wingName: String, using engine: InferenceEngine? = nil, onProgress: ((Int, Int, String) -> Void)? = nil) async throws {
        print("[GraphPalace] SYNAPTIC SYNTHESIS INITIATED for \(wingName).")
        
        guard let context = modelContext else {
            print("[GraphPalace] Warning: No ModelContext attached.")
            return
        }
        
        // Fetch all MemoryEntries for this wing implicitly by querying the Wing
        let fetchDescriptor = FetchDescriptor<PalaceWing>(predicate: #Predicate { $0.name == wingName })
        guard let wing = try? context.fetch(fetchDescriptor).first else { return }
        
        var extractionTargets: [String] = []
        var rawMemories: [String] = []
        // Every memory except the generated summary. Only `hall_facts` (and audio/vision)
        // used to count, which skipped mined events and preferences and downloaded
        // personas' preferences.
        for room in wing.rooms where room.name != "persona_index" {
            for memory in room.memories {
                rawMemories.append(memory.text)
            }
        }
        
        // Massive Context Batching: Combine micro-chunks into heavy blocks
        // Target ~16,000 characters per extraction shot to maximize KV Cache efficiency
        let targetBlockSize = 16000
        var currentBlock = ""
        
        for text in rawMemories {
            if currentBlock.count + text.count > targetBlockSize {
                extractionTargets.append(currentBlock.trimmingCharacters(in: .whitespacesAndNewlines))
                currentBlock = text + "\n"
            } else {
                currentBlock += text + "\n"
            }
        }
        if !currentBlock.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            extractionTargets.append(currentBlock.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        
        print("[GraphPalace] Synthesizing edges across \(extractionTargets.count) multimodal targets.")
        
        guard let engine = engine else {
            print("[GraphPalace] Engine unavailable or not injected. Skipping LLM generation loop.")
            return
        }
        // With no model loaded every generation ends at once with no text, so the run
        // "finished" having saved nothing. Say so instead.
        guard engine.loadedModelId != nil else {
            throw GraphSynthesisError.noModelLoaded
        }
        guard !extractionTargets.isEmpty else {
            throw GraphSynthesisError.nothingToSynthesize(wingName)
        }

        for (index, target) in extractionTargets.enumerated() {
            // The whole ~16,000-character chunk used to go into the progress line.
            onProgress?(index + 1, extractionTargets.count, "SYNTHESIZING EDGE: \(target.prefix(160))…")
            let prompt = buildGraphPrompt(text: target)
            let stream = engine.generate(messages: [.user(prompt)])
            var response = ""
            for await token in stream where !token.isThinking {
                response += token.text
            }

            if let nodeArray = parseGraphTriples(fromJSONString: response, wingName: wingName) {
                // Deduplicate within the same chunk to prevent SwiftData ephemeral collisions
                var seenIds = Set<String>()
                
                for triple in nodeArray {
                    if !seenIds.contains(triple.id) {
                        seenIds.insert(triple.id)
                        context.insert(triple)
                        print("[GraphPalace] Edge Extracted: \(triple.subject) -> [\(triple.predicate)] -> \(triple.object)")
                    }
                }
                // Save immediately to flush temporary IDs to stable SQLite row-ids, avoiding `.unique` constraint panics on large loops
                try? context.save()
            }
        }
    }
    
    public func synthesizePersonaIndex(wingName: String, using engine: InferenceEngine, onProgress: ((Int, Int, String) -> Void)? = nil) async throws {
        print("[GraphPalace] SYNTHESIZING PERSONA INDEX for \(wingName).")
        onProgress?(1, 1, "DISTILLING GOD-NODES INTO CONDENSED PERSONA INDEX...")
        guard let context = modelContext else { return }
        
        // This persona's links only; the summary was built from every persona's graph.
        let target: String? = wingName
        let fetchDescriptor = FetchDescriptor<KnowledgeGraphTriple>(
            predicate: #Predicate { $0.wingName == target })
        guard let allTriples = try? context.fetch(fetchDescriptor), !allTriples.isEmpty else { return }
        
        // Phase 3: Leiden-style grouping (by Subject for now)
        var groupedTriples: [String: [KnowledgeGraphTriple]] = [:]
        for triple in allTriples {
            groupedTriples[triple.subject, default: []].append(triple)
        }
        
        var indexPrompt = "Create a 100-word dense persona index mapping out the following extracted graph communities. Ignore ambiguous links. Format as a dense summary:\n\n"
        for (subject, edges) in groupedTriples {
            if edges.count > 1 { // Only count "God Nodes" (highly connected)
                indexPrompt += "Node: \(subject)\n"
                for edge in edges {
                    indexPrompt += " - \(edge.predicate) \(edge.object) (Type: \(edge.taxonomy ?? "fact"), Truth: \(edge.confidence ?? "INFERRED"))\n"
                }
                indexPrompt += "\n"
            }
        }
        
        let stream = engine.generate(messages: [.user(indexPrompt)])
        var response = ""
        for await token in stream where !token.isThinking {
            response += token.text
        }
        if let thinkEnd = response.range(of: "</think>") {
            response = String(response[thinkEnd.upperBound...])
        }
        response = response.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !response.isEmpty else { return }

        _ = try await MemoryPalaceService.shared.saveMemories(wingName: wingName, roomName: "persona_index", texts: [response], type: "hall_facts")
        print("[GraphPalace] Persona Index Generated & Saved.")
    }
    
    /// Reads triples out of a model reply.
    ///
    /// Tolerant on purpose: the old parser accepted only a reply that was exactly one JSON
    /// array of string-only objects, so a thinking block, a sentence of preamble, one
    /// non-string value or a reply cut off at the token limit discarded the whole chunk,
    /// and the Mind Palace stayed empty. Each `{...}` object is now parsed on its own.
    public func parseGraphTriples(fromJSONString jsonString: String, wingName: String? = nil) -> [KnowledgeGraphTriple]? {
        var text = jsonString
        if let thinkEnd = text.range(of: "</think>") {
            text = String(text[thinkEnd.upperBound...])
        }

        func string(_ value: Any?) -> String? {
            switch value {
            case let s as String:
                let trimmed = s.trimmingCharacters(in: .whitespacesAndNewlines)
                return trimmed.isEmpty ? nil : trimmed
            case let n as NSNumber: return n.stringValue
            default: return nil
            }
        }

        var triples: [KnowledgeGraphTriple] = []
        var searchStart = text.startIndex
        while let open = text[searchStart...].firstIndex(of: "{"),
            let close = text[open...].firstIndex(of: "}")
        {
            searchStart = text.index(after: close)
            guard let data = String(text[open...close]).data(using: .utf8),
                let dict = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
                let subject = string(dict["subject"]),
                let predicate = string(dict["predicate"]),
                let object = string(dict["object"])
            else { continue }
            triples.append(KnowledgeGraphTriple(
                subject: subject,
                predicate: predicate,
                object: object,
                taxonomy: string(dict["taxonomy"]),
                confidence: string(dict["confidence"]),
                wingName: wingName
            ))
        }

        return triples.isEmpty ? nil : triples
    }
    
    public func buildGraphPrompt(text: String) -> String {
        return """
        Extract an exhaustive list of subject-predicate-object triples from the following text to form a complete knowledge graph.
        Additionally, classify each triple's 'taxonomy' as either 'fact', 'preference', 'decision', or 'relationship'.
        Additionally, assess the 'confidence' of each triple as either 'EXTRACTED' (literal) or 'INFERRED' (deduced).
        Format the output as a pure JSON array of objects, containing the keys "subject", "predicate", "object", "taxonomy", and "confidence".
        Do not output any reasoning. Return ONLY the JSON array.
        
        Text:
        \(text)
        """
    }
}
