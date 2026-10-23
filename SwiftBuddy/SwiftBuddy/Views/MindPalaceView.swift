import SwiftUI
import SwiftData

struct MindPalaceView: View {
    @Environment(\.dismiss) private var dismiss
    @Query var triples: [KnowledgeGraphTriple]
    
    // Physics Simulation State
    @State private var nodes: [String: ForceNode] = [:]
    @State private var edges: [ForceEdge] = []
    @State private var isSimulating = false
    @State private var isPhysicsSettled = false
    @State private var draggedNodeId: String? = nil
    @State private var simulationSteps = 0
    @State private var dragFit: GraphFit? = nil
    
    // Timer
    private let timer = Timer.publish(every: 1.0 / 60.0, on: .main, in: .common).autoconnect()
    
    var body: some View {
        ZStack(alignment: .topTrailing) {
            SwiftBuddyTheme.background.ignoresSafeArea()
            
            if triples.isEmpty {
                VStack(spacing: 20) {
                    Image(systemName: "network")
                        .font(.system(size: 60))
                        .foregroundStyle(.secondary)
                    Text("No Synaptic Triples Formed")
                        .font(.title2).bold()
                    // Chatting never fills this graph; only synthesis does.
                    Text("Load a model, then in Chat swipe left on a persona and tap Synthesize Graph.")
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 40)
                }
                // Fill the screen so it centres: the ZStack aligns to the top-right corner
                // (for the macOS close button), which is where this message ended up.
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                GeometryReader { proxy in
                    // Draw in graph coordinates, scaled and centred so the whole graph is
                    // on screen. Positions are unbounded, so drawing them in screen points
                    // pushed larger graphs off the edges. Frozen while a node is dragged so
                    // the view doesn't shift under the finger.
                    let fit = dragFit ?? fitTransform(in: proxy.size)

                    Canvas { context, size in
                        context.translateBy(x: fit.offset.x, y: fit.offset.y)
                        context.scaleBy(x: fit.scale, y: fit.scale)

                        // Draw Edges
                        for edge in edges {
                            if let source = nodes[edge.sourceId], let target = nodes[edge.targetId] {
                                var path = Path()
                                path.move(to: source.position)
                                path.addLine(to: target.position)
                                
                                context.stroke(path, with: .color(.secondary.opacity(0.3)), lineWidth: 1.5)
                                
                                // Midpoint for label
                                let midX = (source.position.x + target.position.x) / 2
                                let midY = (source.position.y + target.position.y) / 2
                                let angle = atan2(target.position.y - source.position.y, target.position.x - source.position.x)
                                
                                context.translateBy(x: midX, y: midY)
                                context.rotate(by: Angle(radians: Double(angle)))
                                // Draw edge predicate label with a slightly larger font
                                context.draw(Text(edge.predicate).font(.caption).bold().foregroundColor(SwiftBuddyTheme.accent), at: .zero)
                                context.rotate(by: Angle(radians: -Double(angle)))
                                context.translateBy(x: -midX, y: -midY)
                            }
                        }
                        
                        // Draw Nodes
                        for (_, node) in nodes {
                            let rect = CGRect(x: node.position.x - 10, y: node.position.y - 10, width: 20, height: 20)
                            context.fill(Path(ellipseIn: rect), with: .color(SwiftBuddyTheme.accent))
                            
                            // Let's drop a little blurred halo for the cyberpunk glow
                            context.stroke(Path(ellipseIn: rect.insetBy(dx: -3, dy: -3)), with: .color(SwiftBuddyTheme.accent.opacity(0.4)), lineWidth: 2)
                            
                            context.draw(Text(node.name).font(.callout.bold()), at: CGPoint(x: node.position.x, y: node.position.y + 26))
                        }
                    }
                    .gesture(
                        DragGesture(minimumDistance: 0)
                            .onChanged { value in
                                let location = fit.toGraph(value.location)
                                // Find nearest
                                if draggedNodeId == nil {
                                    if let nearest = nodes.values.min(by: { distance($0.position, location) < distance($1.position, location) }) {
                                        if distance(nearest.position, location) < 30 / fit.scale {
                                            dragFit = fit
                                            draggedNodeId = nearest.id
                                            isPhysicsSettled = false
                                            simulationSteps = 0
                                        }
                                    }
                                }
                                
                                if let id = draggedNodeId {
                                    nodes[id]?.position = location
                                    nodes[id]?.velocity = .zero
                                }
                            }
                            .onEnded { _ in
                                draggedNodeId = nil
                                dragFit = nil
                            }
                    )
                    .onReceive(timer) { _ in
                        if !isPhysicsSettled {
                            stepSimulation(center: .zero)
                        }
                    }
                }
            }
            #if os(macOS)
            Button(action: { dismiss() }) {
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 24))
                    .foregroundStyle(.secondary.opacity(0.8))
            }
            .buttonStyle(.plain)
            .padding(20)
            #endif
        }
        .onAppear {
            initializeGraph()
        }
        .onChange(of: triples.count) { _, _ in
            initializeGraph() // Re-init if new edges arrive!
        }
        #if os(macOS)
        .navigationTitle("Mind Palace")
        #endif
    }
    
    // MARK: - Simulation Engine
    
    private func initializeGraph() {
        var newNodes: [String: ForceNode] = [:]
        var newEdges: [ForceEdge] = []
        
        for triple in triples {
            let sId = triple.subject.lowercased()
            let oId = triple.object.lowercased()
            
            if newNodes[sId] == nil {
                newNodes[sId] = ForceNode(id: sId, name: triple.subject, position: randomPoint(around: .zero))
            }
            if newNodes[oId] == nil {
                newNodes[oId] = ForceNode(id: oId, name: triple.object, position: randomPoint(around: .zero))
            }
            
            newEdges.append(ForceEdge(sourceId: sId, targetId: oId, predicate: triple.predicate))
        }
        
        self.nodes = newNodes
        self.edges = newEdges
        self.isPhysicsSettled = false
        self.simulationSteps = 0
    }
    
    private func randomPoint(around center: CGPoint) -> CGPoint {
        let r = CGFloat.random(in: 0...200)
        let theta = CGFloat.random(in: 0...(2 * .pi))
        return CGPoint(x: center.x + r * cos(theta), y: center.y + r * sin(theta))
    }
    
    /// Scale and offset that fit every node, with room for its label, into `size`.
    private func fitTransform(in size: CGSize) -> GraphFit {
        let center = CGPoint(x: size.width / 2, y: size.height / 2)
        guard let first = nodes.values.first, size.width > 0, size.height > 0 else {
            return GraphFit(scale: 1, offset: center)
        }
        var minX = first.position.x, maxX = minX, minY = first.position.y, maxY = minY
        for node in nodes.values {
            minX = min(minX, node.position.x); maxX = max(maxX, node.position.x)
            minY = min(minY, node.position.y); maxY = max(maxY, node.position.y)
        }
        let margin: CGFloat = 80 // labels sit below and extend beyond the dots
        let width = maxX - minX + 2 * margin
        let height = maxY - minY + 2 * margin
        // Never enlarge past 1.2x. The lower bound is only a guard: clamping at 0.3x let a
        // wide graph overflow, which left the visible part stuck in a corner.
        let scale = max(0.1, min(1.2, size.width / width, size.height / height))
        let graphCenter = CGPoint(x: (minX + maxX) / 2, y: (minY + maxY) / 2)
        return GraphFit(
            scale: scale,
            offset: CGPoint(x: center.x - graphCenter.x * scale, y: center.y - graphCenter.y * scale))
    }

    private func distance(_ p1: CGPoint, _ p2: CGPoint) -> CGFloat {
        hypot(p1.x - p2.x, p1.y - p2.y)
    }
    
    /// One physics step. Runs on a local copy and publishes once: writing each node field
    /// straight into the `@State` dictionary copied the whole graph on every write, many
    /// times per frame, which is what made the screen stall.
    private func stepSimulation(center: CGPoint) {
        var sim = nodes
        let ids = Array(sim.keys)
        guard !ids.isEmpty else {
            isPhysicsSettled = true
            return
        }
        let k: CGFloat = 0.5 // Spring constant
        let repulsion: CGFloat = 12000 // Increased Node repulsion to spread text
        let damping: CGFloat = 0.85
        // Pull to the centre. At 0.005, with repulsion reaching across the whole graph,
        // groups not linked to each other pushed apart forever and drifted off-screen.
        let centerGravity: CGFloat = 0.01
        let repulsionRange: CGFloat = 600 // Nodes farther apart than this don't push
        let maxSpeed: CGFloat = 40 // Keeps near-overlapping nodes from flinging off-screen

        // 1. Calculate Repulsion (Coulomb)
        for i in 0..<ids.count {
            for j in (i + 1)..<ids.count {
                guard let n1 = sim[ids[i]], let n2 = sim[ids[j]] else { continue }
                let dx = n1.position.x - n2.position.x
                let dy = n1.position.y - n2.position.y
                let dist = max(hypot(dx, dy), 1) // Prevent div 0
                guard dist < repulsionRange else { continue }

                let force = repulsion / (dist * dist)
                let fx = (dx / dist) * force
                let fy = (dy / dist) * force

                if n1.id != draggedNodeId {
                    sim[n1.id]?.velocity.x += fx
                    sim[n1.id]?.velocity.y += fy
                }
                if n2.id != draggedNodeId {
                    sim[n2.id]?.velocity.x -= fx
                    sim[n2.id]?.velocity.y -= fy
                }
            }
        }

        // 2. Calculate Spring Attraction (Hooke's)
        for edge in edges {
            guard let n1 = sim[edge.sourceId], let n2 = sim[edge.targetId] else { continue }

            let dx = n2.position.x - n1.position.x
            let dy = n2.position.y - n1.position.y
            let dist = max(hypot(dx, dy), 1)

            // Target length 280 prevents word bunching. Scaled down: at full strength the
            // spring moved nodes up to 140 px per frame, overshot every step and kept the
            // graph oscillating instead of settling.
            let force = (dist - 280) * k * 0.01
            let fx = (dx / dist) * force
            let fy = (dy / dist) * force

            if n1.id != draggedNodeId {
                sim[n1.id]?.velocity.x += fx
                sim[n1.id]?.velocity.y += fy
            }
            if n2.id != draggedNodeId {
                sim[n2.id]?.velocity.x -= fx
                sim[n2.id]?.velocity.y -= fy
            }
        }

        // 3. Center Gravity & Integration
        var totalDisplacement: CGFloat = 0
        for id in ids where id != draggedNodeId {
            guard var node = sim[id] else { continue }
            node.velocity.x += (center.x - node.position.x) * centerGravity
            node.velocity.y += (center.y - node.position.y) * centerGravity
            node.velocity.x = min(max(node.velocity.x * damping, -maxSpeed), maxSpeed)
            node.velocity.y = min(max(node.velocity.y * damping, -maxSpeed), maxSpeed)
            node.position.x += node.velocity.x
            node.position.y += node.velocity.y
            totalDisplacement += abs(node.velocity.x) + abs(node.velocity.y)
            sim[id] = node
        }
        nodes = sim
        simulationSteps += 1

        // Settled once the average node barely moves, or after ~10 s regardless: the old
        // fixed total (< 0.5 across all nodes) was never reached by larger graphs, so the
        // simulation ran forever.
        if totalDisplacement / CGFloat(ids.count) < 0.05 || simulationSteps > 600 {
            isPhysicsSettled = true
        }
    }
}

// Data structures

/// Maps graph coordinates to the screen: screen = graph × scale + offset.
fileprivate struct GraphFit {
    var scale: CGFloat
    var offset: CGPoint

    func toGraph(_ point: CGPoint) -> CGPoint {
        CGPoint(x: (point.x - offset.x) / scale, y: (point.y - offset.y) / scale)
    }
}

fileprivate struct ForceNode {
    let id: String
    let name: String
    var position: CGPoint
    var velocity: CGPoint = .zero
}

fileprivate struct ForceEdge {
    let sourceId: String
    let targetId: String
    let predicate: String
}
