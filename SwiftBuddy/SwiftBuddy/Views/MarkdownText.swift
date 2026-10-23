// MarkdownText.swift — renders a reply's Markdown: headings, lists, quotes, code, tables
import SwiftUI

/// Model replies are Markdown. `Text` only reads inline Markdown from literals, so replies
/// showed raw `**bold**` and `## Heading`. This splits the text into blocks and renders
/// each, with inline styling (bold, italic, code, links) from `AttributedString`.
struct MarkdownText: View {
    let text: String
    var design: Font.Design = .default

    var body: some View {
        let blocks = MarkdownBlock.parse(text)
        VStack(alignment: .leading, spacing: 8) {
            ForEach(blocks.indices, id: \.self) { index in
                view(for: blocks[index])
            }
        }
    }

    @ViewBuilder
    private func view(for block: MarkdownBlock) -> some View {
        switch block {
        case .heading(let level, let text):
            Text(Self.inline(text))
                .font(Self.headingFont(level, design: design))
                .padding(.top, level <= 2 ? 4 : 2)
                .fixedSize(horizontal: false, vertical: true)
        case .paragraph(let text):
            Text(Self.inline(text))
                .font(.system(.body, design: design))
                .fixedSize(horizontal: false, vertical: true)
        case .listItem(let marker, let indent, let text):
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text(marker)
                    .font(.system(.body, design: design))
                    .foregroundStyle(.secondary)
                Text(Self.inline(text))
                    .font(.system(.body, design: design))
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.leading, CGFloat(indent) * 16)
        case .quote(let text):
            HStack(alignment: .top, spacing: 8) {
                RoundedRectangle(cornerRadius: 1.5)
                    .fill(Color.secondary.opacity(0.5))
                    .frame(width: 3)
                Text(Self.inline(text))
                    .font(.system(.body, design: design))
                    .italic()
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        case .code(let language, let code):
            VStack(alignment: .leading, spacing: 4) {
                if !language.isEmpty {
                    Text(language)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                }
                ScrollView(.horizontal, showsIndicators: false) {
                    Text(code)
                        .font(.system(.callout, design: .monospaced))
                }
            }
            .padding(10)
            .background(RoundedRectangle(cornerRadius: 8).fill(Color.primary.opacity(0.07)))
        case .table(let rows):
            ScrollView(.horizontal, showsIndicators: false) {
                Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 6) {
                    ForEach(rows.indices, id: \.self) { r in
                        GridRow {
                            ForEach(rows[r].indices, id: \.self) { c in
                                Text(Self.inline(rows[r][c]))
                                    .font(.system(.callout, design: design))
                                    .fontWeight(r == 0 ? .semibold : .regular)
                            }
                        }
                        if r == 0 { Divider() }
                    }
                }
            }
        case .rule:
            Divider()
        }
    }

    static func inline(_ text: String) -> AttributedString {
        let options = AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        return (try? AttributedString(markdown: text, options: options)) ?? AttributedString(text)
    }

    private static func headingFont(_ level: Int, design: Font.Design) -> Font {
        switch level {
        case 1:  return .system(.title2, design: design).weight(.bold)
        case 2:  return .system(.title3, design: design).weight(.bold)
        case 3:  return .system(.headline, design: design)
        default: return .system(.subheadline, design: design).weight(.semibold)
        }
    }

    /// The reply as plain sentences for reading aloud: no Markdown marks, no code.
    static func plainText(_ text: String) -> String {
        MarkdownBlock.parse(text).compactMap { block -> String? in
            switch block {
            case .heading(_, let t), .paragraph(let t), .listItem(_, _, let t), .quote(let t):
                return String(inline(t).characters)
            case .table(let rows):
                return rows.map { $0.map { String(inline($0).characters) }.joined(separator: ", ") }
                    .joined(separator: ". ")
            case .code, .rule:
                return nil
            }
        }
        .joined(separator: "\n")
    }
}

enum MarkdownBlock {
    case heading(Int, String)
    case paragraph(String)
    case listItem(marker: String, indent: Int, text: String)
    case quote(String)
    case code(language: String, code: String)
    case table([[String]])
    case rule

    static func parse(_ text: String) -> [MarkdownBlock] {
        var blocks: [MarkdownBlock] = []
        var paragraph: [String] = []
        var quote: [String] = []
        var table: [[String]] = []
        var code: [String]? = nil  // non-nil inside a ``` fence
        var language = ""

        func flush() {
            if !paragraph.isEmpty { blocks.append(.paragraph(paragraph.joined(separator: "\n"))); paragraph = [] }
            if !quote.isEmpty { blocks.append(.quote(quote.joined(separator: "\n"))); quote = [] }
            if !table.isEmpty { blocks.append(.table(table)); table = [] }
        }

        for raw in text.replacingOccurrences(of: "\r", with: "").components(separatedBy: "\n") {
            let line = raw.trimmingCharacters(in: .whitespaces)

            if code != nil {
                if line.hasPrefix("```") {
                    blocks.append(.code(language: language, code: code!.joined(separator: "\n")))
                    code = nil
                } else {
                    code!.append(raw)
                }
                continue
            }
            if line.hasPrefix("```") {
                flush()
                language = String(line.dropFirst(3)).trimmingCharacters(in: .whitespaces)
                code = []
                continue
            }
            if line.isEmpty { flush(); continue }

            if line.hasPrefix("|") {
                if !paragraph.isEmpty || !quote.isEmpty { flush() }
                var cells = line.split(separator: "|", omittingEmptySubsequences: false)
                    .map { $0.trimmingCharacters(in: .whitespaces) }
                if cells.first == "" { cells.removeFirst() }
                if cells.last == "" { cells.removeLast() }
                // The |---|:---:| row under the header.
                let isSeparator = !cells.isEmpty && cells.allSatisfy { cell in
                    !cell.isEmpty && cell.allSatisfy { $0 == "-" || $0 == ":" }
                }
                if !isSeparator { table.append(cells) }
                continue
            }
            if !table.isEmpty { flush() }

            let hashes = line.prefix(while: { $0 == "#" }).count
            if (1...6).contains(hashes), line.dropFirst(hashes).first == " " {
                flush()
                blocks.append(.heading(hashes, String(line.dropFirst(hashes + 1))))
                continue
            }

            let compact = line.replacingOccurrences(of: " ", with: "")
            if compact.count >= 3, let first = compact.first, "-*_".contains(first),
               compact.allSatisfy({ $0 == first }) {
                flush()
                blocks.append(.rule)
                continue
            }

            let indent = min(3, raw.prefix(while: { $0 == " " || $0 == "\t" }).count / 2)
            if let bullet = ["- ", "* ", "+ "].first(where: { line.hasPrefix($0) }) {
                flush()
                blocks.append(.listItem(marker: "•", indent: indent, text: String(line.dropFirst(bullet.count))))
                continue
            }
            let digits = line.prefix(while: { $0.isNumber })
            if !digits.isEmpty, digits.count <= 3 {
                let rest = line.dropFirst(digits.count)
                if let mark = rest.first, mark == "." || mark == ")", rest.dropFirst().first == " " {
                    flush()
                    blocks.append(.listItem(marker: "\(digits).", indent: indent, text: String(rest.dropFirst(2))))
                    continue
                }
            }

            if line.hasPrefix(">") {
                if !paragraph.isEmpty { flush() }
                quote.append(String(line.dropFirst()).trimmingCharacters(in: .whitespaces))
                continue
            }
            if !quote.isEmpty { flush() }
            paragraph.append(line)
        }
        // A fence still open while the reply streams in.
        if let code { blocks.append(.code(language: language, code: code.joined(separator: "\n"))) }
        flush()
        return blocks
    }
}
