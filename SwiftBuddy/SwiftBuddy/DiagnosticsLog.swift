// DiagnosticsLog.swift — Console output and memory samples saved to a file (iOS)
//
// A sideloaded app has no console to read, and an out-of-memory kill leaves no trace in
// the app. This sends everything printed (the app's and mlx-swift-lm's) to
// Documents/Logs/console.log, visible in the Files app, and adds a memory sample line
// whenever usage moves. Output is line-buffered, so every finished line has reached the
// file before a kill. The previous run's log is kept as console-previous.log — after a
// crash, that is the one to read.
import Foundation

#if os(iOS)
import os

enum DiagnosticsLog {
    private static var memoryTimer: DispatchSourceTimer?

    static var directory: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Logs", isDirectory: true)
    }

    static func start() {
        let fm = FileManager.default
        try? fm.createDirectory(at: directory, withIntermediateDirectories: true)
        let current = directory.appendingPathComponent("console.log")
        let previous = directory.appendingPathComponent("console-previous.log")
        try? fm.removeItem(at: previous)
        try? fm.moveItem(at: current, to: previous)

        guard freopen(current.path, "a", stdout) != nil else { return }
        freopen(current.path, "a", stderr)
        setvbuf(stdout, nil, _IOLBF, 0)
        setvbuf(stderr, nil, _IONBF, 0)

        let info = Bundle.main.infoDictionary
        print("=== SwiftBuddy \(info?["CFBundleShortVersionString"] ?? "?") (\(info?["CFBundleVersion"] ?? "?")) launched \(Date()) ===")
        startMemorySampler()
    }

    /// One line per 25 MB of change, checked 4 times a second: enough to see the climb
    /// towards the limit without flooding the file while idle.
    private static func startMemorySampler() {
        let timer = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "swiftbuddy.memlog"))
        var lastLogged: UInt64 = 0
        timer.schedule(deadline: .now(), repeating: .milliseconds(250))
        timer.setEventHandler {
            let used = AppMemory.footprint()
            let delta = used > lastLogged ? used - lastLogged : lastLogged - used
            guard delta >= 25 * 1_048_576 else { return }
            lastLogged = used
            let left = UInt64(os_proc_available_memory())
            print(String(
                format: "[mem] %@ %.2f GB used, %.2f GB left",
                ISO8601DateFormatter.string(from: Date(), timeZone: .current, formatOptions: [.withTime, .withColonSeparatorInTime]),
                Double(used) / 1_073_741_824, Double(left) / 1_073_741_824))
        }
        timer.resume()
        memoryTimer = timer
    }
}
#endif
