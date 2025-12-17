import Foundation
import AVFoundation
import Accelerate

/// Real-time audio analysis using AVAudioEngine and vDSP
/// Ported from Python librosa-based AudioBrain
@MainActor
class AudioAnalyzer: ObservableObject {
    // MARK: - Published Properties (for UI binding)

    @Published private(set) var isRunning: Bool = false
    @Published private(set) var onsetStrength: Float = 0
    @Published private(set) var subBassLevel: Float = 0      // 20-80 Hz
    @Published private(set) var bassLevel: Float = 0         // 80-250 Hz
    @Published private(set) var midLevel: Float = 0          // 250-4000 Hz
    @Published private(set) var hiHatLevel: Float = 0        // 8000-16000 Hz
    @Published private(set) var spectralFlux: Float = 0
    @Published private(set) var rmsLevel: Float = 0
    @Published private(set) var peakLevel: Float = 0

    // MARK: - Audio Engine

    private let audioEngine = AVAudioEngine()
    private var inputNode: AVAudioInputNode { audioEngine.inputNode }

    // MARK: - FFT Setup

    private let fftSize = 2048
    private var fftSetup: FFTSetup?
    private var log2n: vDSP_Length = 0

    // FFT buffers
    private var realBuffer: [Float] = []
    private var imagBuffer: [Float] = []
    private var windowBuffer: [Float] = []
    private var magnitudes: [Float] = []
    private var previousMagnitudes: [Float] = []

    // Onset detection
    private var previousRMS: Float = 0
    private var onsetHistory: [Float] = []
    private let onsetHistorySize = 10

    // Sample rate (will be set when starting)
    private var sampleRate: Double = 48000

    // MARK: - Initialization

    init() {
        setupFFT()
    }

    deinit {
        if let fftSetup = fftSetup {
            vDSP_destroy_fftsetup(fftSetup)
        }
    }

    // MARK: - Public Methods

    /// Start real-time audio analysis from microphone/system audio
    func start() throws {
        guard !isRunning else { return }

        // Configure audio session (iOS)
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .measurement)
        try session.setActive(true)
        #endif

        let format = inputNode.outputFormat(forBus: 0)
        sampleRate = format.sampleRate

        // Install tap on input node
        inputNode.installTap(onBus: 0, bufferSize: UInt32(fftSize), format: format) { [weak self] buffer, time in
            self?.processAudioBuffer(buffer)
        }

        try audioEngine.start()
        isRunning = true
    }

    /// Stop audio analysis
    func stop() {
        guard isRunning else { return }

        inputNode.removeTap(onBus: 0)
        audioEngine.stop()
        isRunning = false
    }

    /// Analyze audio from a file (for offline analysis)
    func analyzeFile(at url: URL) async throws -> [AudioAnalysisFrame] {
        var frames: [AudioAnalysisFrame] = []

        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        sampleRate = format.sampleRate

        let bufferSize = AVAudioFrameCount(fftSize)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: bufferSize) else {
            throw AudioAnalysisError.bufferCreationFailed
        }

        var frameIndex = 0

        while file.framePosition < file.length {
            try file.read(into: buffer)
            processAudioBuffer(buffer)

            let frame = AudioAnalysisFrame(
                time: Double(frameIndex) * Double(fftSize) / sampleRate,
                onsetStrength: onsetStrength,
                subBassLevel: subBassLevel,
                bassLevel: bassLevel,
                midLevel: midLevel,
                hiHatLevel: hiHatLevel,
                spectralFlux: spectralFlux,
                rmsLevel: rmsLevel
            )
            frames.append(frame)

            frameIndex += 1
        }

        return frames
    }

    // MARK: - Private Methods

    private func setupFFT() {
        log2n = vDSP_Length(log2(Double(fftSize)))
        fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2))

        let halfSize = fftSize / 2
        realBuffer = [Float](repeating: 0, count: halfSize)
        imagBuffer = [Float](repeating: 0, count: halfSize)
        magnitudes = [Float](repeating: 0, count: halfSize)
        previousMagnitudes = [Float](repeating: 0, count: halfSize)

        // Create Hann window
        windowBuffer = [Float](repeating: 0, count: fftSize)
        vDSP_hann_window(&windowBuffer, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))
    }

    private func processAudioBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData?[0] else { return }
        let frameCount = Int(buffer.frameLength)

        // Apply window function
        var windowedSignal = [Float](repeating: 0, count: fftSize)
        let copyCount = min(frameCount, fftSize)
        vDSP_vmul(channelData, 1, windowBuffer, 1, &windowedSignal, 1, vDSP_Length(copyCount))

        // Compute RMS
        var rms: Float = 0
        vDSP_rmsqv(windowedSignal, 1, &rms, vDSP_Length(copyCount))
        Task { @MainActor in
            self.rmsLevel = rms
        }

        // Compute peak
        var peak: Float = 0
        vDSP_maxmgv(windowedSignal, 1, &peak, vDSP_Length(copyCount))
        Task { @MainActor in
            self.peakLevel = peak
        }

        // Perform FFT
        performFFT(windowedSignal)

        // Extract frequency bands
        let bands = extractFrequencyBands()

        // Compute spectral flux
        let flux = computeSpectralFlux()

        // Compute onset strength
        let onset = computeOnsetStrength(rms: rms)

        // Update published properties on main thread
        Task { @MainActor in
            self.subBassLevel = bands.subBass
            self.bassLevel = bands.bass
            self.midLevel = bands.mid
            self.hiHatLevel = bands.hiHat
            self.spectralFlux = flux
            self.onsetStrength = onset
        }
    }

    private func performFFT(_ signal: [Float]) {
        guard let fftSetup = fftSetup else { return }

        var signal = signal

        // Pack into split complex format
        signal.withUnsafeMutableBufferPointer { signalPtr in
            var splitComplex = DSPSplitComplex(realp: &realBuffer, imagp: &imagBuffer)

            signalPtr.baseAddress?.withMemoryRebound(to: DSPComplex.self, capacity: fftSize / 2) { complexPtr in
                vDSP_ctoz(complexPtr, 2, &splitComplex, 1, vDSP_Length(fftSize / 2))
            }

            // Perform FFT
            vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(FFT_FORWARD))

            // Compute magnitudes
            vDSP_zvmags(&splitComplex, 1, &magnitudes, 1, vDSP_Length(fftSize / 2))
        }

        // Convert to dB scale
        var one: Float = 1.0
        vDSP_vdbcon(magnitudes, 1, &one, &magnitudes, 1, vDSP_Length(magnitudes.count), 1)
    }

    private func extractFrequencyBands() -> (subBass: Float, bass: Float, mid: Float, hiHat: Float) {
        let binWidth = sampleRate / Double(fftSize)

        // Frequency ranges
        let subBassRange = frequencyToBinRange(lowHz: 20, highHz: 80, binWidth: binWidth)
        let bassRange = frequencyToBinRange(lowHz: 80, highHz: 250, binWidth: binWidth)
        let midRange = frequencyToBinRange(lowHz: 250, highHz: 4000, binWidth: binWidth)
        let hiHatRange = frequencyToBinRange(lowHz: 8000, highHz: 16000, binWidth: binWidth)

        return (
            subBass: averageMagnitude(in: subBassRange),
            bass: averageMagnitude(in: bassRange),
            mid: averageMagnitude(in: midRange),
            hiHat: averageMagnitude(in: hiHatRange)
        )
    }

    private func frequencyToBinRange(lowHz: Double, highHz: Double, binWidth: Double) -> Range<Int> {
        let lowBin = max(0, Int(lowHz / binWidth))
        let highBin = min(magnitudes.count - 1, Int(highHz / binWidth))
        return lowBin..<max(lowBin + 1, highBin)
    }

    private func averageMagnitude(in range: Range<Int>) -> Float {
        guard range.count > 0 else { return 0 }

        var sum: Float = 0
        for i in range {
            sum += magnitudes[i]
        }
        return sum / Float(range.count)
    }

    private func computeSpectralFlux() -> Float {
        var flux: Float = 0

        for i in 0..<magnitudes.count {
            let diff = magnitudes[i] - previousMagnitudes[i]
            // Half-wave rectification (only positive changes)
            if diff > 0 {
                flux += diff
            }
        }

        // Update previous magnitudes
        previousMagnitudes = magnitudes

        // Normalize
        return flux / Float(magnitudes.count)
    }

    private func computeOnsetStrength(rms: Float) -> Float {
        // Onset = ratio of current RMS to smoothed previous RMS
        let onset = rms / max(previousRMS, 0.001)

        // Update history for adaptive threshold
        onsetHistory.append(onset)
        if onsetHistory.count > onsetHistorySize {
            onsetHistory.removeFirst()
        }

        // Smooth previous RMS
        previousRMS = previousRMS * 0.9 + rms * 0.1

        return onset
    }
}

// MARK: - Supporting Types

struct AudioAnalysisFrame {
    let time: Double
    let onsetStrength: Float
    let subBassLevel: Float
    let bassLevel: Float
    let midLevel: Float
    let hiHatLevel: Float
    let spectralFlux: Float
    let rmsLevel: Float
}

enum AudioAnalysisError: Error {
    case bufferCreationFailed
    case fileReadFailed
}
