// PalaceVisualizerView.swift
import SwiftUI
import SwiftData

struct PalaceVisualizerView: View {
    @Environment(\.dismiss) private var dismiss
    @Query(sort: \PalaceWing.createdDate) var wings: [PalaceWing]
    @State private var expandedRooms: Set<String> = []
    
    var body: some View {
        ZStack(alignment: .topTrailing) {
        // Vertical only, sized to the screen: the sideways canvas with 60 pt margins cut
        // rooms and their facts off on iPhone and iPad.
        ScrollView(.vertical, showsIndicators: true) {
            VStack(alignment: .leading, spacing: 0) {
                ForEach(wings) { wing in
                    VStack(alignment: .leading, spacing: 0) {
                        
                        // Tunnel Connection from previous Wing
                        if wing.name != wings.first?.name {
                            TunnelConnector()
                        }
                        
                        // Main Wing Container
                        WingNodeView(wing: wing, expandedRooms: $expandedRooms)
                    }
                }
                
                if wings.isEmpty {
                    VStack(spacing: 20) {
                        Image(systemName: "brain.head.profile")
                            .font(.system(size: 60))
                            .foregroundStyle(.secondary)
                        Text("No Architectures Built")
                            .font(.title2).bold()
                        // The Inspector Registry this pointed to is never shown on iOS.
                        Text("In Chat, tap Get Personas to download one, or make your own in the Miner tab.")
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .padding(.top, 100)
                }
            }
            .padding(16)
        }
        .background(SwiftBuddyTheme.background.ignoresSafeArea())
        
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
        #if os(macOS)
        .navigationTitle("Palace Visualizer")
        #endif
    }
}

struct TunnelConnector: View {
    var body: some View {
        HStack {
            Rectangle()
                .fill(Color.secondary.opacity(0.4))
                .frame(width: 2, height: 40)
                .padding(.leading, 24) // Align to the visual 'tunnel' anchor
            Text("tunnel")
                .font(.caption2.monospaced())
                .foregroundStyle(.secondary)
                .padding(.leading, 4)
            Spacer()
        }
    }
}

/// One persona: its rooms as a grid that wraps to the screen width, and the facts of
/// each opened room full width below. An opened room used to draw its 280 pt box inside
/// the room's 140 pt column with a negative offset, so it was clipped by the sideways
/// scroll view and overlapped the next rooms.
struct WingNodeView: View {
    let wing: PalaceWing
    @Binding var expandedRooms: Set<String>

    private let columns = [GridItem(.adaptive(minimum: 140), spacing: 10)]

    private func key(_ room: PalaceRoom) -> String { "\(wing.name)/\(room.name)" }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Image(systemName: "building.2.crop.circle.fill")
                    .foregroundStyle(SwiftBuddyTheme.accent)
                Text(wing.name)
                    .font(.headline)
                Spacer()
                Text("\(wing.rooms.count) room\(wing.rooms.count == 1 ? "" : "s")")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            LazyVGrid(columns: columns, alignment: .leading, spacing: 10) {
                ForEach(wing.rooms) { room in
                    RoomNodeView(room: room, isExpanded: Binding(
                        get: { expandedRooms.contains(key(room)) },
                        set: { open in
                            if open { expandedRooms.insert(key(room)) }
                            else { expandedRooms.remove(key(room)) }
                        }
                    ))
                }
            }

            ForEach(wing.rooms.filter { expandedRooms.contains(key($0)) }) { room in
                ClosetView(room: room)
                    .transition(.opacity)
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(SwiftBuddyTheme.surface.opacity(0.6))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(SwiftBuddyTheme.divider, lineWidth: 1)
        )
    }
}

/// A room's button: its name and how many facts it holds.
struct RoomNodeView: View {
    let room: PalaceRoom
    @Binding var isExpanded: Bool

    var body: some View {
        Button {
            withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
                isExpanded.toggle()
            }
        } label: {
            VStack(alignment: .leading, spacing: 4) {
                Text(room.name)
                    .font(.subheadline.bold())
                    .lineLimit(2)
                    .multilineTextAlignment(.leading)
                Text("\(room.memories.count) fact\(room.memories.count == 1 ? "" : "s")")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, minHeight: 50, alignment: .leading)
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .background(SwiftBuddyTheme.surface)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(isExpanded ? SwiftBuddyTheme.accent : Color.secondary.opacity(0.4), lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
    }
}

/// An opened room's facts, newest first, wrapping to the full width.
struct ClosetView: View {
    let room: PalaceRoom
    private static let limit = 30

    var body: some View {
        let memories = room.memories.sorted { $0.dateAdded > $1.dateAdded }
        VStack(alignment: .leading, spacing: 10) {
            Label(room.name, systemImage: "cabinet.fill")
                .font(.caption.bold())
                .foregroundStyle(SwiftBuddyTheme.accent)

            ForEach(memories.prefix(Self.limit)) { memory in
                VStack(alignment: .leading, spacing: 3) {
                    Text(memory.hallType)
                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                        .foregroundStyle(SwiftBuddyTheme.accent)
                    Text(memory.text)
                        .font(.footnote)
                        .foregroundStyle(.primary.opacity(0.85))
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Divider()
            }

            if memories.count > Self.limit {
                Text("+ \(memories.count - Self.limit) more")
                    .font(.caption2).italic()
                    .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(SwiftBuddyTheme.background.opacity(0.8))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.secondary.opacity(0.2), lineWidth: 1)
        )
    }
}
