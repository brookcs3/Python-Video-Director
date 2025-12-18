import Foundation
import AVFoundation
import CoreVideo
import VideoToolbox

/// HDR-preserving video exporter using AVAssetWriter
/// Mirrors the FFmpeg pipeline from Python but uses native iOS APIs
class VideoExporter: ObservableObject {
    @Published var isRecording: Bool = false
    @Published var recordedDuration: Double = 0
    @Published var status: String = "Ready"

    private var assetWriter: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var pixelBufferAdaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var audioInput: AVAssetWriterInput?

    private var startTime: CMTime?
    private var frameCount: Int = 0

    private let outputSize: CGSize
    private let frameRate: Double
    private let isHDR: Bool

    init(size: CGSize = CGSize(width: 1920, height: 1080), frameRate: Double = 30, hdr: Bool = true) {
        self.outputSize = size
        self.frameRate = frameRate
        self.isHDR = hdr
    }

    // MARK: - Recording

    /// Start recording to a file
    func startRecording(to outputURL: URL) throws {
        guard !isRecording else { return }

        // Remove existing file
        try? FileManager.default.removeItem(at: outputURL)

        // Create asset writer
        assetWriter = try AVAssetWriter(outputURL: outputURL, fileType: .mov)

        // Configure video input
        var videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: outputSize.width,
            AVVideoHeightKey: outputSize.height,
        ]

        if isHDR {
            // HDR settings for 10-bit HEVC
            videoSettings[AVVideoColorPropertiesKey] = [
                AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_2020,
                AVVideoTransferFunctionKey: AVVideoTransferFunction_SMPTE_ST_2084_PQ,
                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_2020
            ]

            videoSettings[AVVideoCompressionPropertiesKey] = [
                AVVideoAverageBitRateKey: 40_000_000,  // 40 Mbps
                AVVideoProfileLevelKey: kVTProfileLevel_HEVC_Main10_AutoLevel,
                AVVideoAllowFrameReorderingKey: true
            ]
        } else {
            videoSettings[AVVideoCompressionPropertiesKey] = [
                AVVideoAverageBitRateKey: 20_000_000,  // 20 Mbps
                AVVideoProfileLevelKey: kVTProfileLevel_HEVC_Main_AutoLevel
            ]
        }

        videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        videoInput?.expectsMediaDataInRealTime = true

        // Pixel buffer attributes
        let pixelBufferAttributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: isHDR ? kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange : kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: outputSize.width,
            kCVPixelBufferHeightKey as String: outputSize.height,
            kCVPixelBufferMetalCompatibilityKey as String: true
        ]

        pixelBufferAdaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: videoInput!,
            sourcePixelBufferAttributes: pixelBufferAttributes
        )

        assetWriter?.add(videoInput!)

        // Start writing
        assetWriter?.startWriting()
        assetWriter?.startSession(atSourceTime: .zero)

        startTime = nil
        frameCount = 0
        isRecording = true
        status = "Recording..."
    }

    /// Write a video frame
    func writeFrame(_ pixelBuffer: CVPixelBuffer, at time: CMTime) {
        guard isRecording,
              let videoInput = videoInput,
              let adaptor = pixelBufferAdaptor,
              videoInput.isReadyForMoreMediaData else {
            return
        }

        // Set start time on first frame
        if startTime == nil {
            startTime = time
        }

        let presentationTime = CMTimeSubtract(time, startTime!)
        adaptor.append(pixelBuffer, withPresentationTime: presentationTime)

        frameCount += 1
        recordedDuration = CMTimeGetSeconds(presentationTime)
    }

    /// Write a frame from a CGImage (convenience method)
    func writeFrame(_ image: CGImage, at time: CMTime) {
        guard let pixelBuffer = createPixelBuffer(from: image) else { return }
        writeFrame(pixelBuffer, at: time)
    }

    /// Stop recording and finalize the file
    func stopRecording() async {
        guard isRecording else { return }

        videoInput?.markAsFinished()
        audioInput?.markAsFinished()

        await assetWriter?.finishWriting()

        isRecording = false
        status = "Recording complete (\(frameCount) frames)"
    }

    // MARK: - Combining Videos

    /// Combine two videos side-by-side or stacked
    static func combineVideos(
        video1: URL,
        video2: URL,
        output: URL,
        layout: CombineLayout
    ) async throws {
        let asset1 = AVURLAsset(url: video1)
        let asset2 = AVURLAsset(url: video2)

        // Get video tracks
        guard let track1 = try await asset1.loadTracks(withMediaType: .video).first,
              let track2 = try await asset2.loadTracks(withMediaType: .video).first else {
            throw ExportError.noVideoTrack
        }

        let size1 = try await track1.load(.naturalSize)
        let size2 = try await track2.load(.naturalSize)
        let duration = try await asset1.load(.duration)

        // Calculate output size
        let outputSize: CGSize
        switch layout {
        case .horizontal:
            outputSize = CGSize(width: size1.width + size2.width, height: max(size1.height, size2.height))
        case .vertical:
            outputSize = CGSize(width: max(size1.width, size2.width), height: size1.height + size2.height)
        }

        // Create composition
        let composition = AVMutableComposition()
        let videoComposition = AVMutableVideoComposition()

        // Add video tracks
        guard let compositionTrack1 = composition.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid),
              let compositionTrack2 = composition.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid) else {
            throw ExportError.compositionFailed
        }

        let timeRange = CMTimeRange(start: .zero, duration: duration)
        try compositionTrack1.insertTimeRange(timeRange, of: track1, at: .zero)
        try compositionTrack2.insertTimeRange(timeRange, of: track2, at: .zero)

        // Create layer instructions
        let instruction = AVMutableVideoCompositionInstruction()
        instruction.timeRange = timeRange

        let layerInstruction1 = AVMutableVideoCompositionLayerInstruction(assetTrack: compositionTrack1)
        let layerInstruction2 = AVMutableVideoCompositionLayerInstruction(assetTrack: compositionTrack2)

        // Position videos based on layout
        switch layout {
        case .horizontal:
            layerInstruction1.setTransform(.identity, at: .zero)
            layerInstruction2.setTransform(CGAffineTransform(translationX: size1.width, y: 0), at: .zero)
        case .vertical:
            layerInstruction1.setTransform(.identity, at: .zero)
            layerInstruction2.setTransform(CGAffineTransform(translationX: 0, y: size1.height), at: .zero)
        }

        instruction.layerInstructions = [layerInstruction1, layerInstruction2]

        videoComposition.instructions = [instruction]
        videoComposition.renderSize = outputSize
        videoComposition.frameDuration = CMTime(value: 1, timescale: 30)

        // Export
        try? FileManager.default.removeItem(at: output)

        guard let exportSession = AVAssetExportSession(asset: composition, presetName: AVAssetExportPresetHighestQuality) else {
            throw ExportError.exportSessionFailed
        }

        exportSession.outputURL = output
        exportSession.outputFileType = .mov
        exportSession.videoComposition = videoComposition

        await exportSession.export()

        if let error = exportSession.error {
            throw error
        }
    }

    // MARK: - Private Helpers

    private func createPixelBuffer(from image: CGImage) -> CVPixelBuffer? {
        let width = image.width
        let height = image.height

        var pixelBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferMetalCompatibilityKey: true
        ]

        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else {
            return nil
        }

        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        return buffer
    }
}

// MARK: - Supporting Types

enum CombineLayout {
    case horizontal  // Side-by-side
    case vertical    // Stacked
}

enum ExportError: Error {
    case noVideoTrack
    case compositionFailed
    case exportSessionFailed
}
