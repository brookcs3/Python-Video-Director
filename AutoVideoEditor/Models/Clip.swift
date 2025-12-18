import Foundation
import AVFoundation
import CoreImage

/// Represents a video clip segment extracted from the source video
struct Clip: Identifiable, Hashable {
    let id: UUID
    let url: URL
    let duration: CMTime
    let thumbnailImage: CGImage?
    let index: Int

    /// Metadata extracted during segmentation
    let averageBrightness: Float
    let dominantColor: CIColor?
    let motionIntensity: Float

    /// Playback state
    var playCount: Int = 0
    var isOnceEver: Bool = false  // Special clips that only play once per session

    init(url: URL, duration: CMTime, index: Int, thumbnail: CGImage? = nil) {
        self.id = UUID()
        self.url = url
        self.duration = duration
        self.thumbnailImage = thumbnail
        self.index = index
        self.averageBrightness = 0.5
        self.dominantColor = nil
        self.motionIntensity = 0.5
    }

    var durationSeconds: Double {
        CMTimeGetSeconds(duration)
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }

    static func == (lhs: Clip, rhs: Clip) -> Bool {
        lhs.id == rhs.id
    }
}

/// Configuration for clip segmentation
struct SegmentationConfig {
    var minClipDuration: Double = 2.0      // Minimum clip length in seconds
    var maxClipDuration: Double = 8.0      // Maximum clip length in seconds
    var preferredClipDuration: Double = 4.0 // Target clip length

    var sceneChangeThreshold: Float = 0.4   // Histogram difference threshold
    var audioOnsetThreshold: Float = 1.2    // Onset detection sensitivity
    var motionThreshold: Float = 0.5        // Optical flow threshold

    var alignToBeat: Bool = true            // Snap cuts to beat grid
    var detectSceneChanges: Bool = true     // Use visual scene detection
    var detectAudioOnsets: Bool = true      // Use audio transient detection

    static let `default` = SegmentationConfig()

    static let fast = SegmentationConfig(
        minClipDuration: 1.5,
        maxClipDuration: 4.0,
        preferredClipDuration: 2.5
    )

    static let slow = SegmentationConfig(
        minClipDuration: 4.0,
        maxClipDuration: 12.0,
        preferredClipDuration: 6.0
    )
}
