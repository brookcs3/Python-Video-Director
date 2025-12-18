import Foundation
import Combine

/// Director mode - controls edit pacing and style
enum DirectorMode: String, CaseIterable {
    case chaos  // Rapid cuts, reactive to every transient
    case flow   // Sustained playback, minimal cuts

    var description: String {
        switch self {
        case .chaos: return "CHAOS - Rapid cuts on beats"
        case .flow: return "FLOW - Sustained playback"
        }
    }
}

/// Playback direction and speed
enum PlaybackBehavior {
    case normal
    case fastForward(rate: Float)  // Triggered by sub-bass
    case rewind(rate: Float)       // Triggered by hi-hats
    case pause

    var rate: Float {
        switch self {
        case .normal: return 1.0
        case .fastForward(let rate): return rate
        case .rewind(let rate): return rate
        case .pause: return 0.0
        }
    }
}

/// Director configuration
struct DirectorConfig {
    // CHAOS mode settings
    var onsetThreshold: Float = 1.2      // Onset strength to trigger cut
    var minCutInterval: Double = 0.3     // Minimum time between cuts

    // FLOW mode settings
    var flowMinDuration: Double = 2.2    // Minimum FLOW duration
    var flowMaxDuration: Double = 4.2    // Maximum FLOW duration
    var flowEntryThreshold: Float = 0.3  // Spectral flux below this → enter FLOW
    var flowExitThreshold: Float = 0.6   // Spectral flux above this → exit FLOW

    // Speed modulation
    var subBassThreshold: Float = 1.2    // Sub-bass level for fast-forward
    var hiHatThreshold: Float = 0.95     // Hi-hat level for rewind
    var fastForwardRate: Float = 1.3
    var rewindRate: Float = -1.1

    // Effect triggers
    var inversionThreshold: Float = 2.5  // Onset strength for inversion
    var hazeChance: Float = 0.15         // Probability of RGB haze per cut

    static let `default` = DirectorConfig()

    static let aggressive = DirectorConfig(
        onsetThreshold: 0.8,
        minCutInterval: 0.2,
        flowMinDuration: 1.5,
        flowMaxDuration: 2.5
    )

    static let relaxed = DirectorConfig(
        onsetThreshold: 1.8,
        minCutInterval: 0.8,
        flowMinDuration: 4.0,
        flowMaxDuration: 8.0
    )
}

/// Main director engine - makes edit decisions based on audio analysis
@MainActor
class Director: ObservableObject {
    @Published private(set) var mode: DirectorMode = .chaos
    @Published private(set) var playback: PlaybackBehavior = .normal
    @Published private(set) var shouldSwitchClip: Bool = false
    @Published private(set) var shouldApplyInversion: Bool = false
    @Published private(set) var shouldApplyHaze: Bool = false

    private var config: DirectorConfig
    private var flowStartTime: Date?
    private var lastCutTime: Date = .distantPast
    private var flowDuration: Double = 0

    init(config: DirectorConfig = .default) {
        self.config = config
    }

    /// Update director state based on audio analysis
    /// Call this every audio buffer (~21ms at 48kHz/1024 samples)
    func update(
        onsetStrength: Float,
        subBassLevel: Float,
        hiHatLevel: Float,
        spectralFlux: Float
    ) {
        // Reset triggers
        shouldSwitchClip = false
        shouldApplyInversion = false
        shouldApplyHaze = false
        playback = .normal

        switch mode {
        case .chaos:
            updateChaosMode(
                onsetStrength: onsetStrength,
                subBassLevel: subBassLevel,
                hiHatLevel: hiHatLevel,
                spectralFlux: spectralFlux
            )

        case .flow:
            updateFlowMode(spectralFlux: spectralFlux)
        }
    }

    /// Acknowledge that a clip switch occurred
    func clipSwitched() {
        shouldSwitchClip = false
        lastCutTime = Date()
    }

    /// Force a specific mode
    func setMode(_ newMode: DirectorMode) {
        mode = newMode
        if newMode == .flow {
            enterFlowMode()
        }
    }

    /// Update configuration
    func updateConfig(_ newConfig: DirectorConfig) {
        config = newConfig
    }

    // MARK: - Private

    private func updateChaosMode(
        onsetStrength: Float,
        subBassLevel: Float,
        hiHatLevel: Float,
        spectralFlux: Float
    ) {
        let now = Date()
        let timeSinceLastCut = now.timeIntervalSince(lastCutTime)

        // Check if we should trigger a cut
        if onsetStrength > config.onsetThreshold && timeSinceLastCut > config.minCutInterval {
            shouldSwitchClip = true

            // Random chance of haze effect on cut
            if Float.random(in: 0...1) < config.hazeChance {
                shouldApplyHaze = true
            }
        }

        // Check for inversion on heavy transients
        if onsetStrength > config.inversionThreshold {
            shouldApplyInversion = true
        }

        // Speed modulation based on frequency bands
        if subBassLevel > config.subBassThreshold {
            playback = .fastForward(rate: config.fastForwardRate)
        } else if hiHatLevel > config.hiHatThreshold {
            playback = .rewind(rate: config.rewindRate)
        }

        // Check if we should enter FLOW mode (low energy section)
        if spectralFlux < config.flowEntryThreshold && timeSinceLastCut > 2.0 {
            enterFlowMode()
        }
    }

    private func updateFlowMode(spectralFlux: Float) {
        guard let flowStart = flowStartTime else {
            // Shouldn't happen, but recover
            mode = .chaos
            return
        }

        let flowElapsed = Date().timeIntervalSince(flowStart)

        // Check if minimum FLOW duration has passed
        if flowElapsed >= flowDuration {
            // Check if energy increased enough to exit FLOW
            if spectralFlux > config.flowExitThreshold {
                exitFlowMode()
            }
        }
    }

    private func enterFlowMode() {
        mode = .flow
        flowStartTime = Date()
        flowDuration = Double.random(in: config.flowMinDuration...config.flowMaxDuration)
        shouldSwitchClip = false
    }

    private func exitFlowMode() {
        mode = .chaos
        flowStartTime = nil
        // Trigger an immediate cut on FLOW exit
        shouldSwitchClip = true
    }
}
