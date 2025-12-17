# iOS Automatic Video Editor - Architecture Design

## Overview

An iOS app that takes a single uploaded video, automatically segments it into smaller clips,
then uses audio-reactive logic to create dynamic edits in real-time.

---

## Core Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER UPLOADS VIDEO                          │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CLIP SEGMENTATION ENGINE                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  1. Analyze video for natural cut points:                    │   │
│  │     - Scene changes (histogram comparison)                   │   │
│  │     - Audio transients (onset detection)                     │   │
│  │     - Motion changes (optical flow analysis)                 │   │
│  │                                                               │   │
│  │  2. Split into segments (configurable: 2-10 seconds each)   │   │
│  │                                                               │   │
│  │  3. Export segments to clip library                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIP LIBRARY                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  StrictDeck Algorithm (from Python visualizer):              │   │
│  │  - Shuffles all clips                                        │   │
│  │  - Prevents repeats within lap                               │   │
│  │  - Special "once-ever" clips for intro/outro                 │   │
│  │  - Thumbnail previews for UI                                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      AUDIO ANALYSIS ENGINE                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  AVAudioEngine + Accelerate (vDSP) for real-time analysis:  │   │
│  │                                                               │   │
│  │  - Onset Detection → triggers clip switches                 │   │
│  │  - Sub-bass (20-80Hz) → triggers fast-forward               │   │
│  │  - Hi-hats (8-16kHz) → triggers rewind                      │   │
│  │  - Spectral Flux → CHAOS vs FLOW mode switching             │   │
│  │  - BPM Detection → sync cuts to beat grid                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         DIRECTOR ENGINE                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  State Machine (CHAOS ↔ FLOW):                               │   │
│  │                                                               │   │
│  │  CHAOS Mode:                                                  │   │
│  │  - Rapid cuts on onset detection                             │   │
│  │  - Random effects triggered                                  │   │
│  │  - High energy sections                                      │   │
│  │                                                               │   │
│  │  FLOW Mode:                                                   │   │
│  │  - Sustained playback (2-4 seconds)                          │   │
│  │  - Minimal cuts                                               │   │
│  │  - Smooth transitions                                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          FX ENGINE                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Metal Shaders for real-time effects:                        │   │
│  │                                                               │   │
│  │  - RGB Haze (channel separation)                             │   │
│  │  - Frame Inversion                                           │   │
│  │  - Glitch effects                                            │   │
│  │  - Color grading                                             │   │
│  │  - Transitions (cross-dissolve, wipe, etc.)                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      OUTPUT OPTIONS                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │   Live Preview   │  │  Record Session  │  │   Export Final   │  │
│  │   (MTKView)      │  │  (AVAssetWriter) │  │   (HDR/ProRes)   │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Swift Implementation Structure

```
AutoVideoEditor/
├── App/
│   ├── AutoVideoEditorApp.swift
│   └── ContentView.swift
│
├── Models/
│   ├── Clip.swift                 # Video clip metadata
│   ├── ClipLibrary.swift          # StrictDeck algorithm
│   └── DirectorState.swift        # CHAOS/FLOW state machine
│
├── Services/
│   ├── VideoSegmenter.swift       # Split uploaded video into clips
│   ├── AudioAnalyzer.swift        # Real-time audio analysis
│   ├── Director.swift             # Edit decision logic
│   └── VideoExporter.swift        # HDR-preserving export
│
├── Effects/
│   ├── Shaders.metal              # Metal compute shaders
│   ├── RGBHaze.swift              # Channel separation effect
│   ├── FrameInvert.swift          # Inversion effect
│   └── EffectPipeline.swift       # Chain effects together
│
├── Views/
│   ├── ImportView.swift           # Video upload UI
│   ├── SegmentationView.swift     # Show segmentation progress
│   ├── PreviewView.swift          # Live preview (MTKView)
│   ├── ControlsView.swift         # Play/pause/record controls
│   └── ExportView.swift           # Export options
│
└── Utilities/
    ├── FFT.swift                  # vDSP wrapper for frequency analysis
    └── Extensions.swift           # Helper extensions
```

---

## Key Components

### 1. VideoSegmenter (Clip Creation)

```swift
class VideoSegmenter {
    /// Analyze video and find optimal cut points
    func analyzeForCutPoints(asset: AVAsset) async -> [CMTime] {
        var cutPoints: [CMTime] = []

        // Method 1: Scene detection via histogram comparison
        let sceneChanges = await detectSceneChanges(asset: asset)

        // Method 2: Audio onset detection
        let audioOnsets = await detectAudioOnsets(asset: asset)

        // Method 3: Motion-based cuts (optical flow)
        let motionChanges = await detectMotionChanges(asset: asset)

        // Merge and filter cut points
        cutPoints = mergeCutPoints(sceneChanges, audioOnsets, motionChanges)

        return cutPoints
    }

    /// Extract clips from video at specified cut points
    func extractClips(asset: AVAsset, cutPoints: [CMTime]) async throws -> [Clip] {
        var clips: [Clip] = []

        for i in 0..<cutPoints.count - 1 {
            let startTime = cutPoints[i]
            let endTime = cutPoints[i + 1]
            let timeRange = CMTimeRange(start: startTime, end: endTime)

            let clip = try await exportClip(from: asset, timeRange: timeRange, index: i)
            clips.append(clip)
        }

        return clips
    }
}
```

### 2. ClipLibrary (StrictDeck Algorithm)

```swift
class ClipLibrary: ObservableObject {
    @Published var clips: [Clip] = []
    @Published var currentClip: Clip?

    private var deck: [Clip] = []
    private var playedThisLap: Set<UUID> = []
    private var lapCount = 0

    /// Get next clip using StrictDeck shuffle algorithm
    func nextClip() -> Clip {
        // If deck is empty, reshuffle
        if deck.isEmpty {
            deck = clips.shuffled()
            playedThisLap.removeAll()
            lapCount += 1
        }

        // Find a clip that hasn't been played this lap
        while let clip = deck.popLast() {
            if !playedThisLap.contains(clip.id) {
                playedThisLap.insert(clip.id)
                currentClip = clip
                return clip
            }
        }

        // Fallback: return any clip
        return clips.randomElement()!
    }
}
```

### 3. AudioAnalyzer (Real-time Analysis)

```swift
class AudioAnalyzer: ObservableObject {
    private let audioEngine = AVAudioEngine()
    private let fftSize = 2048

    @Published var onsetDetected = false
    @Published var subBassLevel: Float = 0
    @Published var hiHatLevel: Float = 0
    @Published var spectralFlux: Float = 0

    func startAnalyzing() {
        let inputNode = audioEngine.inputNode
        let format = inputNode.outputFormat(forBus: 0)

        inputNode.installTap(onBus: 0, bufferSize: UInt32(fftSize), format: format) { buffer, time in
            self.analyzeBuffer(buffer)
        }

        try? audioEngine.start()
    }

    private func analyzeBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData?[0] else { return }

        // Perform FFT using Accelerate
        var magnitudes = performFFT(channelData, count: Int(buffer.frameLength))

        // Extract frequency bands
        subBassLevel = extractBand(magnitudes, lowHz: 20, highHz: 80)
        hiHatLevel = extractBand(magnitudes, lowHz: 8000, highHz: 16000)

        // Onset detection
        onsetDetected = detectOnset(magnitudes)

        // Spectral flux for CHAOS/FLOW detection
        spectralFlux = calculateSpectralFlux(magnitudes)
    }
}
```

### 4. Director (Edit Decision Engine)

```swift
class Director: ObservableObject {
    enum Mode { case chaos, flow }

    @Published var mode: Mode = .chaos
    @Published var shouldSwitch = false
    @Published var playbackRate: Float = 1.0

    private let flowDuration: Double = 3.0  // seconds in FLOW mode
    private var flowStartTime: Date?

    func update(audio: AudioAnalyzer) {
        switch mode {
        case .chaos:
            // Trigger cuts on onset
            if audio.onsetDetected {
                shouldSwitch = true
            }

            // Fast-forward on sub-bass
            if audio.subBassLevel > 1.2 {
                playbackRate = 1.3
            }

            // Rewind on hi-hats
            if audio.hiHatLevel > 0.95 {
                playbackRate = -1.1
            }

            // Switch to FLOW if energy drops
            if audio.spectralFlux < 0.3 {
                enterFlowMode()
            }

        case .flow:
            // Stay in FLOW for minimum duration
            if let start = flowStartTime,
               Date().timeIntervalSince(start) > flowDuration {

                // Switch back to CHAOS if energy increases
                if audio.spectralFlux > 0.6 {
                    mode = .chaos
                }
            }
        }
    }

    private func enterFlowMode() {
        mode = .flow
        flowStartTime = Date()
        shouldSwitch = false
    }
}
```

---

## Segmentation Strategies

### Option 1: Time-Based (Simple)
Split every N seconds (2-5 seconds per clip)

### Option 2: Beat-Based
Detect BPM, split on downbeats

### Option 3: Scene-Based (Recommended)
Analyze visual content for natural scene boundaries:
- Histogram difference threshold
- Edge detection changes
- Color palette shifts

### Option 4: Hybrid
Combine all methods, prefer cuts that align with:
1. Scene changes
2. Audio transients
3. Beat grid

---

## Export Options

### Single Window Mode
Direct recording of the live preview:
```swift
let exporter = VideoExporter(size: CGSize(width: 1920, height: 1080))
exporter.startRecording(to: outputURL)
// ... frames written during playback ...
exporter.finishRecording()
```

### Double Window Mode
Record two independent streams, combine in post:
```swift
// Record each window
let window1 = VideoExporter(outputURL: "window1.mov")
let window2 = VideoExporter(outputURL: "window2.mov")

// Combine after
VideoExporter.combine(window1, window2, layout: .horizontal, output: "combined.mov")
```

---

## MCP Integration Plan

### XcodeBuildMCP
- Build and run on simulator
- Capture test videos from simulator
- Run unit tests for audio analysis

### Apple Docs MCP
- Query AVFoundation APIs for segmentation
- Look up AVAudioEngine best practices
- Research Metal shader patterns

### DeepThinking MCP
- Use `causal` mode for debugging audio→edit relationships
- Use `systems-thinking` for architecture decisions
- Use `algorithmic` mode for optimizing StrictDeck

---

## Next Steps

1. Create Xcode project structure
2. Implement VideoSegmenter with scene detection
3. Implement ClipLibrary with StrictDeck
4. Implement AudioAnalyzer with vDSP
5. Implement Director state machine
6. Create Metal shaders for effects
7. Build SwiftUI interface
8. Implement VideoExporter with HDR
