import Foundation
import AVFoundation
import CoreImage
import Accelerate

/// Segments a source video into smaller clips based on scene changes, audio, and motion
class VideoSegmenter: ObservableObject {
    @Published var progress: Double = 0
    @Published var status: String = "Ready"
    @Published var isProcessing: Bool = false

    private var config: SegmentationConfig

    init(config: SegmentationConfig = .default) {
        self.config = config
    }

    /// Main entry point: analyze video and extract clips
    func segmentVideo(at url: URL, outputDirectory: URL) async throws -> [Clip] {
        await MainActor.run {
            isProcessing = true
            status = "Loading video..."
            progress = 0
        }

        let asset = AVURLAsset(url: url)

        // Step 1: Find cut points
        await MainActor.run { status = "Analyzing for cut points..." }
        let cutPoints = try await findCutPoints(in: asset)

        await MainActor.run {
            status = "Found \(cutPoints.count) cut points"
            progress = 0.3
        }

        // Step 2: Extract clips at cut points
        await MainActor.run { status = "Extracting clips..." }
        let clips = try await extractClips(from: asset, at: cutPoints, to: outputDirectory)

        await MainActor.run {
            status = "Complete! Created \(clips.count) clips"
            progress = 1.0
            isProcessing = false
        }

        return clips
    }

    // MARK: - Cut Point Detection

    private func findCutPoints(in asset: AVURLAsset) async throws -> [CMTime] {
        var allCutPoints: [CMTime] = [.zero]

        let duration = try await asset.load(.duration)
        let durationSeconds = CMTimeGetSeconds(duration)

        // Method 1: Scene change detection
        if config.detectSceneChanges {
            let sceneChanges = try await detectSceneChanges(in: asset)
            allCutPoints.append(contentsOf: sceneChanges)
        }

        // Method 2: Audio onset detection
        if config.detectAudioOnsets {
            let audioOnsets = try await detectAudioOnsets(in: asset)
            allCutPoints.append(contentsOf: audioOnsets)
        }

        // Method 3: Fixed interval fallback (ensure we have cuts)
        let fixedCuts = generateFixedIntervalCuts(duration: durationSeconds)
        allCutPoints.append(contentsOf: fixedCuts)

        // Always include the end
        allCutPoints.append(duration)

        // Sort and filter cut points
        return filterCutPoints(allCutPoints, duration: duration)
    }

    /// Detect scene changes using histogram comparison
    private func detectSceneChanges(in asset: AVURLAsset) async throws -> [CMTime] {
        var sceneChanges: [CMTime] = []

        let duration = try await asset.load(.duration)
        let durationSeconds = CMTimeGetSeconds(duration)

        // Sample frames at regular intervals
        let sampleInterval: Double = 0.25  // 4 samples per second
        let sampleCount = Int(durationSeconds / sampleInterval)

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        generator.maximumSize = CGSize(width: 160, height: 90)  // Low res for speed

        var previousHistogram: [Float]?

        for i in 0..<sampleCount {
            let time = CMTime(seconds: Double(i) * sampleInterval, preferredTimescale: 600)

            do {
                let cgImage = try generator.copyCGImage(at: time, actualTime: nil)
                let histogram = computeHistogram(for: cgImage)

                if let prevHist = previousHistogram {
                    let difference = histogramDifference(prevHist, histogram)

                    if difference > config.sceneChangeThreshold {
                        sceneChanges.append(time)
                    }
                }

                previousHistogram = histogram
            } catch {
                // Skip frames that can't be generated
                continue
            }

            await MainActor.run {
                progress = 0.1 + (0.1 * Double(i) / Double(sampleCount))
            }
        }

        return sceneChanges
    }

    /// Detect audio onsets (transients) for cut points
    private func detectAudioOnsets(in asset: AVURLAsset) async throws -> [CMTime] {
        var onsets: [CMTime] = []

        guard let audioTrack = try await asset.loadTracks(withMediaType: .audio).first else {
            return []
        }

        let reader = try AVAssetReader(asset: asset)
        let outputSettings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
            AVSampleRateKey: 44100
        ]

        let output = AVAssetReaderTrackOutput(track: audioTrack, outputSettings: outputSettings)
        reader.add(output)
        reader.startReading()

        let bufferSize = 2048
        var previousEnergy: Float = 0
        var sampleIndex = 0
        let sampleRate: Double = 44100

        while let sampleBuffer = output.copyNextSampleBuffer() {
            guard let dataBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else { continue }

            var lengthAtOffset: Int = 0
            var totalLength: Int = 0
            var dataPointer: UnsafeMutablePointer<Int8>?

            CMBlockBufferGetDataPointer(dataBuffer, atOffset: 0, lengthAtOffsetOut: &lengthAtOffset,
                                        totalLengthOut: &totalLength, dataPointerOut: &dataPointer)

            guard let data = dataPointer else { continue }

            let floatCount = totalLength / MemoryLayout<Float>.size
            let floatPointer = data.withMemoryRebound(to: Float.self, capacity: floatCount) { $0 }

            // Calculate energy
            var energy: Float = 0
            vDSP_measqv(floatPointer, 1, &energy, vDSP_Length(floatCount))
            energy = sqrt(energy / Float(floatCount))

            // Onset detection: look for sudden energy increase
            let onset = energy / max(previousEnergy, 0.001)
            if onset > config.audioOnsetThreshold && previousEnergy > 0.01 {
                let time = CMTime(seconds: Double(sampleIndex) / sampleRate, preferredTimescale: 600)
                onsets.append(time)
            }

            previousEnergy = energy * 0.9 + previousEnergy * 0.1  // Smooth
            sampleIndex += floatCount
        }

        await MainActor.run {
            progress = 0.25
        }

        return onsets
    }

    /// Generate fixed interval cuts as fallback
    private func generateFixedIntervalCuts(duration: Double) -> [CMTime] {
        var cuts: [CMTime] = []
        var time = config.preferredClipDuration

        while time < duration - config.minClipDuration {
            cuts.append(CMTime(seconds: time, preferredTimescale: 600))
            time += config.preferredClipDuration
        }

        return cuts
    }

    /// Filter and sort cut points to ensure valid clip durations
    private func filterCutPoints(_ points: [CMTime], duration: CMTime) -> [CMTime] {
        // Remove duplicates and sort
        var unique = Array(Set(points)).sorted { CMTimeCompare($0, $1) < 0 }

        // Filter to ensure minimum clip duration
        var filtered: [CMTime] = [.zero]

        for point in unique {
            guard let last = filtered.last else { continue }

            let gap = CMTimeGetSeconds(point) - CMTimeGetSeconds(last)

            // Skip if too close to previous cut
            if gap < config.minClipDuration {
                continue
            }

            // If gap is too large, add intermediate cuts
            if gap > config.maxClipDuration {
                let intermediateCount = Int(gap / config.preferredClipDuration)
                for i in 1...intermediateCount {
                    let intermediate = CMTimeAdd(last, CMTime(seconds: Double(i) * config.preferredClipDuration, preferredTimescale: 600))
                    if CMTimeCompare(intermediate, point) < 0 {
                        filtered.append(intermediate)
                    }
                }
            }

            filtered.append(point)
        }

        // Ensure end is included
        if let last = filtered.last, CMTimeCompare(last, duration) < 0 {
            filtered.append(duration)
        }

        return filtered
    }

    // MARK: - Clip Extraction

    private func extractClips(from asset: AVURLAsset, at cutPoints: [CMTime], to outputDir: URL) async throws -> [Clip] {
        var clips: [Clip] = []

        // Create output directory if needed
        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

        for i in 0..<(cutPoints.count - 1) {
            let startTime = cutPoints[i]
            let endTime = cutPoints[i + 1]
            let timeRange = CMTimeRange(start: startTime, end: endTime)

            let outputURL = outputDir.appendingPathComponent("clip_\(String(format: "%03d", i)).mov")

            try await exportClip(from: asset, timeRange: timeRange, to: outputURL)

            // Generate thumbnail
            let generator = AVAssetImageGenerator(asset: asset)
            generator.appliesPreferredTrackTransform = true
            let thumbnailTime = CMTimeAdd(startTime, CMTime(seconds: 0.5, preferredTimescale: 600))
            let thumbnail = try? generator.copyCGImage(at: thumbnailTime, actualTime: nil)

            let clip = Clip(
                url: outputURL,
                duration: CMTimeSubtract(endTime, startTime),
                index: i,
                thumbnail: thumbnail
            )
            clips.append(clip)

            await MainActor.run {
                progress = 0.3 + (0.7 * Double(i) / Double(cutPoints.count - 1))
                status = "Extracting clip \(i + 1) of \(cutPoints.count - 1)..."
            }
        }

        return clips
    }

    private func exportClip(from asset: AVURLAsset, timeRange: CMTimeRange, to outputURL: URL) async throws {
        // Remove existing file
        try? FileManager.default.removeItem(at: outputURL)

        guard let exportSession = AVAssetExportSession(asset: asset, presetName: AVAssetExportPresetHighestQuality) else {
            throw SegmentationError.exportFailed
        }

        exportSession.outputURL = outputURL
        exportSession.outputFileType = .mov
        exportSession.timeRange = timeRange

        await exportSession.export()

        if let error = exportSession.error {
            throw error
        }
    }

    // MARK: - Histogram Analysis

    private func computeHistogram(for image: CGImage) -> [Float] {
        let width = image.width
        let height = image.height
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        let bitsPerComponent = 8

        var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)

        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return []
        }

        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Compute grayscale histogram (64 bins)
        var histogram = [Float](repeating: 0, count: 64)
        let binSize = 256 / 64

        for i in stride(from: 0, to: pixelData.count, by: bytesPerPixel) {
            let r = Float(pixelData[i])
            let g = Float(pixelData[i + 1])
            let b = Float(pixelData[i + 2])
            let gray = Int((0.299 * r + 0.587 * g + 0.114 * b))
            let bin = min(gray / binSize, 63)
            histogram[bin] += 1
        }

        // Normalize
        let total = Float(width * height)
        for i in 0..<histogram.count {
            histogram[i] /= total
        }

        return histogram
    }

    private func histogramDifference(_ h1: [Float], _ h2: [Float]) -> Float {
        guard h1.count == h2.count else { return 1.0 }

        var sum: Float = 0
        for i in 0..<h1.count {
            sum += abs(h1[i] - h2[i])
        }

        return sum / 2.0  // Normalize to 0-1 range
    }
}

enum SegmentationError: Error {
    case exportFailed
    case noVideoTrack
    case noAudioTrack
}
