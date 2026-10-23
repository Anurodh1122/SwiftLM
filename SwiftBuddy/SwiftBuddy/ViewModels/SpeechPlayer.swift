// SpeechPlayer.swift — reads replies aloud with the system voice
import AVFoundation
import SwiftUI

/// Reads one reply at a time with `AVSpeechSynthesizer`, the on-device system voice, so it
/// needs no model or download and runs alongside generation.
@MainActor
final class SpeechPlayer: NSObject, ObservableObject, AVSpeechSynthesizerDelegate {
    static let shared = SpeechPlayer()

    /// The message being read, for its button to show Stop.
    @Published private(set) var speakingID: UUID?

    private let synthesizer = AVSpeechSynthesizer()
    private var current: AVSpeechUtterance?

    override init() {
        super.init()
        synthesizer.delegate = self
    }

    func toggle(id: UUID, text: String) {
        if speakingID == id { stop(); return }
        synthesizer.stopSpeaking(at: .immediate)

        let spoken = MarkdownText.plainText(text)
        guard !spoken.isEmpty else { return }
        #if os(iOS)
        // Plays with the silent switch on and lowers other audio while speaking.
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
        try? AVAudioSession.sharedInstance().setActive(true)
        #endif
        let utterance = AVSpeechUtterance(string: spoken)
        utterance.voice = AVSpeechSynthesisVoice(language: AVSpeechSynthesisVoice.currentLanguageCode())
        current = utterance
        speakingID = id
        synthesizer.speak(utterance)
    }

    func stop() {
        synthesizer.stopSpeaking(at: .immediate)
        finished(current)
    }

    /// Only the utterance still current clears the button: stopping one reply to read
    /// another reports the first as cancelled after the second has started.
    private func finished(_ utterance: AVSpeechUtterance?) {
        guard utterance === current else { return }
        current = nil
        speakingID = nil
        #if os(iOS)
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        #endif
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor in self.finished(utterance) }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor in self.finished(utterance) }
    }
}
