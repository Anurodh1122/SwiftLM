import Foundation
import SwiftData
import NaturalLanguage

#if canImport(MLXInferenceCore)
import MLXInferenceCore
#endif

@MainActor
final class MemoryPalaceService {
    static let shared = MemoryPalaceService()
    
    var modelContext: ModelContext?
    
    // Apple's Native Embedding Model
    private let embeddingModel: NLEmbedding? = {
        return NLEmbedding.sentenceEmbedding(for: .english)
    }()
    
    // MARK: - Vector Math
    
    private func cosineSimilarity(a: [Double], b: [Double]) -> Double {
        guard a.count == b.count, a.count > 0 else { return 0.0 }
        
        var dotProduct: Double = 0.0
        var normA: Double = 0.0
        var normB: Double = 0.0
        
        for i in 0..<a.count {
            dotProduct += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        
        if normA == 0.0 || normB == 0.0 { return 0.0 }
        return dotProduct / (sqrt(normA) * sqrt(normB))
    }
    
    private func generateEmbedding(for text: String) -> [Double]? {
        guard let model = embeddingModel else { return nil }
        return model.vector(for: text)
    }
    
    // MARK: - Palace Operations
    
    @discardableResult
    func saveMemory(wingName: String, roomName: String, text: String, type: String = "Facts") throws -> Bool {
        guard let context = modelContext else { throw URLError(.badServerResponse) }
        // Without the on-device sentence embedding the fact used to be dropped silently, so
        // mining reported success and no persona appeared. Keep it: it still belongs to the
        // persona and feeds graph synthesis, though similarity search can't rank it.
        let vector = generateEmbedding(for: text)

        // 1. Semantic Duplicate Guard
        let fetchDesc = FetchDescriptor<MemoryEntry>()
        let existingMemories = try context.fetch(fetchDesc).filter { $0.room?.name == roomName && $0.room?.wing?.name == wingName }
        for mem in existingMemories {
            if mem.text == text { return false }
            if let vector, let emb = mem.embedding {
                let similarity = cosineSimilarity(a: vector, b: emb)
                if similarity > 0.95 {
                    return false // Duplicate blocked
                }
            }
        }
        
        // 2. Find or create Wing
        let fetchWing = FetchDescriptor<PalaceWing>(predicate: #Predicate { $0.name == wingName })
        let wing = (try? context.fetch(fetchWing).first) ?? {
            let w = PalaceWing(name: wingName)
            context.insert(w)
            return w
        }()
        
        // 3. Find or create Room in Wing
        let fetchRoom = FetchDescriptor<PalaceRoom>(predicate: #Predicate { $0.name == roomName && $0.wing?.name == wingName })
        let room = (try? context.fetch(fetchRoom).first) ?? {
            let r = PalaceRoom(name: roomName, wing: wing)
            context.insert(r)
            return r
        }()
        
        // 4. Save Memory
        let entry = MemoryEntry(text: text, hallType: type, embedding: vector, room: room)
        context.insert(entry)
        
        try context.save()
        return true
    }
    
    @discardableResult
    func saveMemories(wingName: String, roomName: String, texts: [String], type: String = "Facts", onProgress: ((Int, Int, String) -> Void)? = nil) async throws -> Int {
        guard let context = modelContext else { throw URLError(.badServerResponse) }
        guard !texts.isEmpty else { return 0 }
        
        let fetchWing = FetchDescriptor<PalaceWing>(predicate: #Predicate { $0.name == wingName })
        let wing = (try? context.fetch(fetchWing).first) ?? {
            let w = PalaceWing(name: wingName)
            context.insert(w)
            return w
        }()
        
        let fetchRoom = FetchDescriptor<PalaceRoom>(predicate: #Predicate { $0.name == roomName && $0.wing?.name == wingName })
        let room = (try? context.fetch(fetchRoom).first) ?? {
            let r = PalaceRoom(name: roomName, wing: wing)
            context.insert(r)
            return r
        }()
        
        let fetchDesc = FetchDescriptor<MemoryEntry>()
        let existingMemories = (try? context.fetch(fetchDesc).filter { $0.room?.name == roomName && $0.room?.wing?.name == wingName }) ?? []
        var existingVectors = existingMemories.compactMap { $0.embedding }
        
        var savedCount = 0
        
        // Batch embedding extraction and insertion
        for (index, text) in texts.enumerated() {
            onProgress?(index + 1, texts.count, text)
            await Task.yield()
            // Kept without an embedding when none is available (see `saveMemory`).
            let vector = generateEmbedding(for: text)

            var isDuplicate = false
            if let vector {
                for emb in existingVectors {
                    if cosineSimilarity(a: vector, b: emb) > 0.95 {
                        isDuplicate = true
                        break
                    }
                }
            }
            if isDuplicate { continue }

            let entry = MemoryEntry(text: text, hallType: type, embedding: vector, room: room)
            context.insert(entry)
            if let vector { existingVectors.append(vector) }
            savedCount += 1
        }
        
        // Single synchronized database transaction
        if savedCount > 0 {
            try context.save()
        }
        
        return savedCount
    }
    
#if canImport(MLXInferenceCore)
    public func synthesizePersonaIndex(wingName: String, using engine: InferenceEngine, onProgress: ((Int, Int, String) -> Void)? = nil) async throws {
        onProgress?(1, 1, "EXTRACTING CORE IDENTITIES VIA VECTOR MATCHING...")
        
        // 1. Vector Search for critical identity aspects
        let extractionCategories = [
            ("Core Identity & Directives", "What is your core identity, main goal, or primary directive?"),
            ("Background & Lore", "What is your origin story, historical background, and past experiences?"),
            ("Tone & Speaking Style", "How do you speak? What is your tone of voice, cadence, and vocabulary?"),
            ("Personality & Demeanor", "What is your personality like? Are you friendly, formal, eccentric, or stoic?"),
            ("Preferences & Dislikes", "What are your personal preferences, likes, dislikes, and hobbies?")
        ]
        
        var combinedContext = ""
        var seenTexts = Set<String>()
        
        for (index, category) in extractionCategories.enumerated() {
            onProgress?(index + 1, extractionCategories.count + 1, "EXTRACTING: \(category.0.uppercased())...")
            
            combinedContext += "\n[\(category.0) Results]\n"
            let matches = try searchMemories(query: category.1, wingName: wingName, topK: 3)
            var categoryHadMatches = false
            
            for match in matches {
                if !seenTexts.contains(match.text) {
                    seenTexts.insert(match.text)
                    combinedContext += "- \(match.text)\n"
                    categoryHadMatches = true
                }
            }
            if !categoryHadMatches {
                combinedContext += "(No direct matches found)\n"
            }
        }
        
        guard !combinedContext.isEmpty else {
            print("[MemoryPalace] No memories found for persona synthesis.")
            return
        }
        
        // 2. Final Download (LLM Combine)
        onProgress?(extractionCategories.count + 1, extractionCategories.count + 1, "SYNTHESIZING FINAL PERSONA (LLM)...")
        let prompt = """
        You are an analytical compiler. Consolidate the following extracted persona memories into a dense, 100-word core identity and talk-tone matrix. 
        Do not output your reasoning or any conversational filler. Only output the final dense summary.
        
        [Vector-Matched Memories]
        \(combinedContext)
        """
        
        let stream = engine.generate(messages: [.user(prompt)])
        var response = ""
        for await token in stream where !token.isThinking {
            response += token.text
        }
        if let thinkEnd = response.range(of: "</think>") {
            response = String(response[thinkEnd.upperBound...])
        }
        let summary = response.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !summary.isEmpty else { return }

        // 3. Save to Palace
        try saveMemory(wingName: wingName, roomName: "persona_index", text: summary, type: "hall_facts")
        print("[MemoryPalace] Vector-Driven Persona Index Generated & Saved.")
    }
#endif
    
    func searchMemories(query: String, wingName: String, roomName: String? = nil, hallType: String? = nil, topK: Int = 5) throws -> [MemoryEntry] {
        guard let context = modelContext else { throw URLError(.badServerResponse) }
        let queryVector = generateEmbedding(for: query)

        let fetchWing = FetchDescriptor<PalaceWing>(predicate: #Predicate { $0.name == wingName })
        guard let wing = try context.fetch(fetchWing).first else { return [] }
        
        var allMemories = wing.rooms.flatMap { $0.memories }
        if let r = roomName { allMemories = allMemories.filter { $0.room?.name == r } }
        if let h = hallType { allMemories = allMemories.filter { $0.hallType == h } }
        
        return sortAndSliceMemories(allMemories, query: query, queryVector: queryVector, topK: topK)
    }

    func searchAllMemories(query: String, topK: Int = 5) throws -> [MemoryEntry] {
        guard let context = modelContext else { throw URLError(.badServerResponse) }
        let queryVector = generateEmbedding(for: query)

        let fetchDesc = FetchDescriptor<MemoryEntry>()
        let allMemories = try context.fetch(fetchDesc)
        return sortAndSliceMemories(allMemories, query: query, queryVector: queryVector, topK: topK)
    }

    /// Ranks by embedding similarity where both sides have a vector, and otherwise by the
    /// share of the query's words the memory contains. Search used to need the on-device
    /// sentence embedding for everything, so without it every lookup came back empty.
    private func sortAndSliceMemories(_ memories: [MemoryEntry], query: String, queryVector: [Double]?, topK: Int) -> [MemoryEntry] {
        let words = Set(query.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count > 2 })
        var scored: [(entry: MemoryEntry, score: Double)] = []
        for mem in memories {
            if let queryVector, let emb = mem.embedding {
                scored.append((mem, cosineSimilarity(a: queryVector, b: emb)))
            } else if !words.isEmpty {
                let text = mem.text.lowercased()
                let hits = words.filter { text.contains($0) }.count
                if hits > 0 {
                    scored.append((mem, Double(hits) / Double(words.count)))
                }
            }
        }
        scored.sort { $0.score > $1.score }
        return scored.prefix(topK).map { $0.entry }
    }

    /// What a persona chat starts from: who the persona is, and its key facts.
    ///
    /// Matched on room names loosely. The chat looked for rooms named exactly "background",
    /// "preferences" and "corpus", but downloaded personas save "BACKGROUND STORY",
    /// "CORE IDENTITY", "CORPUS" and "PREFERENCES", and mined ones use whatever rooms the
    /// model picks, so no persona's identity reached the model. With no matching rooms the
    /// newest memories stand in.
    func identityContext(wingName: String) throws -> (identity: [String], facts: [String]) {
        guard let context = modelContext else { throw URLError(.badServerResponse) }
        let fetchWing = FetchDescriptor<PalaceWing>(predicate: #Predicate { $0.name == wingName })
        guard let wing = try context.fetch(fetchWing).first else { return ([], []) }

        func newestFirst(_ room: PalaceRoom) -> [String] {
            room.memories.sorted { $0.dateAdded > $1.dateAdded }.map(\.text)
        }
        var identity: [String] = []
        var facts: [String] = []
        for room in wing.rooms {
            let name = room.name.lowercased()
            if name == "persona_index" {
                // Synthesis adds a new summary each run; only the latest counts.
                identity.insert(contentsOf: newestFirst(room).prefix(1), at: 0)
            } else if ["identity", "background", "tone", "personality", "persona"].contains(where: { name.contains($0) }) {
                identity += newestFirst(room)
            } else if ["preference", "corpus", "fact"].contains(where: { name.contains($0) }) {
                facts += newestFirst(room)
            }
        }
        if identity.isEmpty && facts.isEmpty {
            facts = wing.rooms.filter { $0.name != "diary" }
                .flatMap(\.memories)
                .sorted { $0.dateAdded > $1.dateAdded }
                .map(\.text)
        }
        return (identity, facts)
    }
    
    func findTunnels(roomName: String) throws -> [String] {
        guard let context = modelContext else { throw URLError(.badServerResponse) }
        let fetchDesc = FetchDescriptor<PalaceRoom>(predicate: #Predicate { $0.name == roomName })
        let rooms = try context.fetch(fetchDesc)
        return rooms.compactMap { $0.wing?.name }
    }
    
    func listRooms(wingName: String) throws -> [String] {
        guard let context = modelContext else { throw URLError(.badServerResponse) }
        let fetchWing = FetchDescriptor<PalaceWing>(predicate: #Predicate { $0.name == wingName })
        guard let wing = try context.fetch(fetchWing).first else { return [] }
        return wing.rooms.map { $0.name }
    }
    
    // Natively pulls raw facts avoiding embedding distance logic 
    func fetchRoomContents(wingName: String, roomName: String) throws -> [String] {
        guard let context = modelContext else { throw URLError(.badServerResponse) }
        let fetchRoom = FetchDescriptor<PalaceRoom>(predicate: #Predicate { 
            $0.name == roomName && $0.wing?.name == wingName 
        })
        guard let room = try context.fetch(fetchRoom).first else { return [] }
        return room.memories.map { $0.text }
    }
    
    // MARK: - Wing Management
    
    func listWings() throws -> [String] {
        guard let context = modelContext else { throw URLError(.badServerResponse) }
        let descriptor = FetchDescriptor<PalaceWing>(sortBy: [SortDescriptor(\.createdDate)])
        let wings = try context.fetch(descriptor)
        return wings.map { $0.name }
    }
    
    func deleteWing(_ name: String) throws {
        guard let context = modelContext else { throw URLError(.badServerResponse) }
        let fetchWing = FetchDescriptor<PalaceWing>(predicate: #Predicate { $0.name == name })
        if let wing = try context.fetch(fetchWing).first {
            context.delete(wing)
            try context.save()
        }
    }
    
    /// Deletes a persona with everything that belongs to it: its rooms and memories
    /// (cascade), its chats and its graph. Deleting only the wing left the chats and the
    /// Mind Palace links behind.
    func deletePersona(_ name: String) throws {
        guard let context = modelContext else { throw URLError(.badServerResponse) }
        let target: String? = name
        for wing in try context.fetch(FetchDescriptor<PalaceWing>(predicate: #Predicate { $0.name == name })) {
            context.delete(wing)
        }
        for session in try context.fetch(FetchDescriptor<ChatSession>(predicate: #Predicate { $0.wingName == target })) {
            context.delete(session)
        }
        for triple in try context.fetch(FetchDescriptor<KnowledgeGraphTriple>(predicate: #Predicate { $0.wingName == target })) {
            context.delete(triple)
        }
        try context.save()
    }

    /// Deletes the memory matching `textMatch`: one containing the text, or failing that one
    /// at least 0.85 similar. It used to delete the top search hit however unrelated.
    @discardableResult
    func deleteMemory(wingName: String, roomName: String?, textMatch: String) throws -> Bool {
        guard let context = modelContext else { throw URLError(.badServerResponse) }
        let fetchWing = FetchDescriptor<PalaceWing>(predicate: #Predicate { $0.name == wingName })
        guard let wing = try context.fetch(fetchWing).first else { return false }
        let candidates = wing.rooms
            .filter { roomName == nil || $0.name == roomName }
            .flatMap(\.memories)

        let needle = textMatch.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        var match: MemoryEntry? = nil
        if !needle.isEmpty {
            match = candidates.first(where: { $0.text.lowercased().contains(needle) })
        }
        if match == nil, let vector = generateEmbedding(for: textMatch) {
            var best: (entry: MemoryEntry, score: Double)? = nil
            for mem in candidates {
                guard let emb = mem.embedding else { continue }
                let score = cosineSimilarity(a: vector, b: emb)
                if score >= 0.85, score > (best?.score ?? 0) { best = (mem, score) }
            }
            match = best?.entry
        }
        guard let match else { return false }
        context.delete(match)
        try context.save()
        return true
    }
    
    // MARK: - Taxonomy & Status
    
    func getCloset(wingName: String, roomName: String) throws -> String {
        guard let context = modelContext else { throw URLError(.badServerResponse) }
        let fetchWing = FetchDescriptor<PalaceWing>(predicate: #Predicate { $0.name == wingName })
        guard let wing = try context.fetch(fetchWing).first else { return "Closet is empty." }
        
        guard let room = wing.rooms.first(where: { $0.name == roomName }) else { return "Closet is empty." }
        
        let memories = room.memories.sorted { $0.dateAdded > $1.dateAdded }
        if memories.isEmpty { return "Closet is empty." }
        
        let facts = memories.map { "[\($0.hallType)] \($0.text)" }.joined(separator: "\n")
        return "Closet for \(wingName)/\(roomName):\n\(facts)"
    }
    
    func getTaxonomy() throws -> String {
        guard let context = modelContext else { throw URLError(.badServerResponse) }
        let descriptor = FetchDescriptor<PalaceWing>(sortBy: [SortDescriptor(\.createdDate)])
        let wings = try context.fetch(descriptor)
        
        var output = "Memory Palace Taxonomy:\n"
        for w in wings {
            output += "Wing: \(w.name) (\(w.rooms.count) rooms)\n"
            for r in w.rooms {
                output += "  - Room: \(r.name) (\(r.memories.count) memories)\n"
            }
        }
        return output
    }
    
    func getPalaceStatus() throws -> (wings: Int, rooms: Int, memories: Int) {
        guard let context = modelContext else { throw URLError(.badServerResponse) }
        let wCount = try context.fetchCount(FetchDescriptor<PalaceWing>())
        let rCount = try context.fetchCount(FetchDescriptor<PalaceRoom>())
        let mCount = try context.fetchCount(FetchDescriptor<MemoryEntry>())
        return (wings: wCount, rooms: rCount, memories: mCount)
    }
    
    // MARK: - Native Fact Checker
    
    private func detectContradiction(existingObject: String, newObject: String) -> Bool {
        // NLEmbedding-based semantic contradiction detection mechanism natively bypassing fact_checker.py limits
        guard let e1 = generateEmbedding(for: existingObject),
              let e2 = generateEmbedding(for: newObject) else { return false }
        
        let similarity = cosineSimilarity(a: e1, b: e2)
        // If similarity is extremely low (< 0.2) but they share the exact same subject+predicate mapping,
        // it triggers the contradiction detector.
        return similarity < 0.2
    }
    
    // MARK: - Tier 5: Temporal Knowledge Graph
    
    @discardableResult
    func addTriple(subject: String, predicate: String, object: String, wingName: String? = nil) throws -> Bool {
        guard let context = modelContext else { throw URLError(.badServerResponse) }

        let targetId = KnowledgeGraphTriple.makeID(subject: subject, predicate: predicate, wingName: wingName)
        let fetchDesc = FetchDescriptor<KnowledgeGraphTriple>(predicate: #Predicate { $0.id == targetId })
        
        if let existing = try context.fetch(fetchDesc).first {
            // Check for direct contradiction before quietly overwriting.
            if detectContradiction(existingObject: existing.object, newObject: object) {
                // Evolve the triple via Contradiction tracking (temporal shift instead of generic overwrite)
                existing.object = "\(object) [Contradicted prior belief: \(existing.object)]"
            } else {
                // Standard Temporal Invalidation
                existing.object = object
            }
            existing.dateObserved = Date()
        } else {
            let triple = KnowledgeGraphTriple(subject: subject, predicate: predicate, object: object, wingName: wingName)
            context.insert(triple)
        }
        
        try context.save()
        return true
    }
    
    func queryEntity(_ subject: String, wingName: String? = nil) throws -> [KnowledgeGraphTriple] {
        guard let context = modelContext else { throw URLError(.badServerResponse) }
        let targetSubj = subject.lowercased()
        let fetchDesc = FetchDescriptor<KnowledgeGraphTriple>()

        // Cannot use lowercased() easily in SwiftData predicates sometimes, so fetch & filter natively for stability
        let allTriples = try context.fetch(fetchDesc)
        return allTriples.filter {
            $0.subject.lowercased() == targetSubj && (wingName == nil || $0.wingName == wingName)
        }
    }
    
    // MARK: - Tier 6: Advanced Support
    
    func getWakeupContext(wingName: String) throws -> String {
        guard let context = modelContext else { throw URLError(.badServerResponse) }
        
        // 1. Fetch Top 5 Triples for this specific wing's context
        let tripleDesc = FetchDescriptor<KnowledgeGraphTriple>(sortBy: [SortDescriptor(\.dateObserved, order: .reverse)])
        let recentTriples = Array((try context.fetch(tripleDesc)).prefix(5))
        
        let kgBlock = recentTriples.map { "[\($0.subject)] \($0.predicate) \($0.object)" }.joined(separator: "\n")
        
        // 2. Fetch last 3 entries from the "diary" room without NSException nesting
        let wingDesc = FetchDescriptor<PalaceWing>(predicate: #Predicate { $0.name == wingName })
        guard let wing = try context.fetch(wingDesc).first else { return "No context found." }
        
        let diaryRoom = wing.rooms.first(where: { $0.name == "diary" })
        let memories = diaryRoom?.memories.sorted { $0.dateAdded > $1.dateAdded } ?? []
        let diaryEntries = Array(memories.prefix(3).reversed())
        
        let diaryBlock = diaryEntries.map { "- \($0.text)" }.joined(separator: "\n")
        
        return """
        --- L0/L1 WAKEUP CONTEXT ---
        [Knowledge Graph]
        \(kgBlock)
        
        [Recent Reflections]
        \(diaryBlock)
        ----------------------------
        """
    }
}
