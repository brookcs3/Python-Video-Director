import Foundation
import Combine

/// StrictDeck algorithm for clip management
/// Ensures no clip repeats within a "lap" (full cycle through all clips)
/// Ported from Python visualizer logic
@MainActor
class ClipLibrary: ObservableObject {
    @Published private(set) var clips: [Clip] = []
    @Published private(set) var currentClip: Clip?
    @Published private(set) var lapCount: Int = 0
    @Published private(set) var clipsPlayedThisLap: Int = 0

    private var deck: [Clip] = []
    private var playedThisLap: Set<UUID> = []
    private var onceEverPlayed: Set<UUID> = []

    /// Load clips from a directory
    func loadClips(from urls: [URL]) async throws {
        var loadedClips: [Clip] = []

        for (index, url) in urls.enumerated() {
            let asset = AVURLAsset(url: url)
            let duration = try await asset.load(.duration)

            // Generate thumbnail
            let thumbnail = try await generateThumbnail(for: asset)

            let clip = Clip(url: url, duration: duration, index: index, thumbnail: thumbnail)
            loadedClips.append(clip)
        }

        self.clips = loadedClips
        reshuffleDeck()
    }

    /// Add clips from segmentation
    func addClips(_ newClips: [Clip]) {
        clips.append(contentsOf: newClips)
        reshuffleDeck()
    }

    /// Mark specific clips as "once-ever" (intro/outro clips)
    func markAsOnceEver(_ clipIndices: [Int]) {
        for index in clipIndices {
            if index < clips.count {
                clips[index].isOnceEver = true
            }
        }
    }

    /// Get next clip using StrictDeck algorithm
    func nextClip() -> Clip? {
        guard !clips.isEmpty else { return nil }

        // If deck is empty, start new lap
        if deck.isEmpty {
            reshuffleDeck()
        }

        // Try to find a clip that hasn't played this lap
        while !deck.isEmpty {
            let clip = deck.removeLast()

            // Skip once-ever clips that have already played
            if clip.isOnceEver && onceEverPlayed.contains(clip.id) {
                continue
            }

            // Skip clips played this lap (shouldn't happen but safety check)
            if playedThisLap.contains(clip.id) {
                continue
            }

            // Mark as played
            playedThisLap.insert(clip.id)
            clipsPlayedThisLap = playedThisLap.count

            if clip.isOnceEver {
                onceEverPlayed.insert(clip.id)
            }

            // Update play count
            if let index = clips.firstIndex(where: { $0.id == clip.id }) {
                clips[index].playCount += 1
            }

            currentClip = clip
            return clip
        }

        // Fallback: start new lap if somehow stuck
        reshuffleDeck()
        return nextClip()
    }

    /// Peek at next clip without advancing
    func peekNextClip() -> Clip? {
        deck.last
    }

    /// Force switch to a specific clip (for manual override)
    func jumpToClip(_ clip: Clip) {
        currentClip = clip
        playedThisLap.insert(clip.id)
        clipsPlayedThisLap = playedThisLap.count
    }

    /// Reset for new session
    func reset() {
        playedThisLap.removeAll()
        onceEverPlayed.removeAll()
        lapCount = 0
        clipsPlayedThisLap = 0
        reshuffleDeck()
    }

    // MARK: - Private

    private func reshuffleDeck() {
        // Only include non-once-ever clips, or once-ever clips that haven't played
        deck = clips.filter { clip in
            if clip.isOnceEver {
                return !onceEverPlayed.contains(clip.id)
            }
            return true
        }.shuffled()

        playedThisLap.removeAll()
        clipsPlayedThisLap = 0
        lapCount += 1
    }

    private func generateThumbnail(for asset: AVURLAsset) async throws -> CGImage? {
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 200, height: 200)

        let time = CMTime(seconds: 0.5, preferredTimescale: 600)
        let cgImage = try generator.copyCGImage(at: time, actualTime: nil)
        return cgImage
    }
}

import AVFoundation
