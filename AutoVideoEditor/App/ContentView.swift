import SwiftUI
import AVKit
import PhotosUI

struct ContentView: View {
    @EnvironmentObject var clipLibrary: ClipLibrary
    @EnvironmentObject var audioAnalyzer: AudioAnalyzer
    @EnvironmentObject var director: Director
    @EnvironmentObject var segmenter: VideoSegmenter

    @State private var selectedVideoItem: PhotosPickerItem?
    @State private var sourceVideoURL: URL?
    @State private var isPlaying = false
    @State private var showingClipLibrary = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Main preview area
                ZStack {
                    if let clip = clipLibrary.currentClip {
                        VideoPlayerView(url: clip.url, isPlaying: $isPlaying)
                    } else {
                        importPromptView
                    }

                    // Overlay: Director mode indicator
                    VStack {
                        HStack {
                            DirectorModeIndicator(mode: director.mode)
                            Spacer()
                            if segmenter.isProcessing {
                                ProgressView(value: segmenter.progress)
                                    .frame(width: 100)
                            }
                        }
                        .padding()

                        Spacer()
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color.black)

                // Audio levels visualization
                AudioLevelsView(analyzer: audioAnalyzer)
                    .frame(height: 60)
                    .padding(.horizontal)

                // Controls
                ControlsView(
                    isPlaying: $isPlaying,
                    clipLibrary: clipLibrary,
                    director: director
                )
                .padding()
            }
            .navigationTitle("Auto Video Editor")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    PhotosPicker(selection: $selectedVideoItem, matching: .videos) {
                        Image(systemName: "square.and.arrow.down")
                    }
                }

                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        showingClipLibrary = true
                    } label: {
                        Image(systemName: "rectangle.stack")
                    }
                    .disabled(clipLibrary.clips.isEmpty)
                }
            }
            .sheet(isPresented: $showingClipLibrary) {
                ClipLibraryView(clipLibrary: clipLibrary)
            }
            .onChange(of: selectedVideoItem) { _, newItem in
                Task {
                    await handleVideoSelection(newItem)
                }
            }
            .onAppear {
                startAudioAnalysis()
            }
        }
    }

    // MARK: - Views

    private var importPromptView: some View {
        VStack(spacing: 20) {
            Image(systemName: "video.badge.plus")
                .font(.system(size: 60))
                .foregroundColor(.gray)

            Text("Import a video to get started")
                .font(.headline)
                .foregroundColor(.gray)

            Text("The video will be automatically segmented into clips for dynamic editing")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)

            PhotosPicker(selection: $selectedVideoItem, matching: .videos) {
                Label("Choose Video", systemImage: "photo.on.rectangle")
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
        }
    }

    // MARK: - Actions

    private func handleVideoSelection(_ item: PhotosPickerItem?) async {
        guard let item = item else { return }

        do {
            // Load video data
            guard let videoData = try await item.loadTransferable(type: Data.self) else { return }

            // Save to temporary file
            let tempURL = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension("mov")
            try videoData.write(to: tempURL)

            sourceVideoURL = tempURL

            // Create output directory for clips
            let clipsDir = FileManager.default.temporaryDirectory
                .appendingPathComponent("AutoEditorClips")
                .appendingPathComponent(UUID().uuidString)

            // Segment the video
            let clips = try await segmenter.segmentVideo(at: tempURL, outputDirectory: clipsDir)

            // Load clips into library
            await clipLibrary.addClips(clips)

            // Start playback
            isPlaying = true

        } catch {
            print("Error loading video: \(error)")
        }
    }

    private func startAudioAnalysis() {
        do {
            try audioAnalyzer.start()
        } catch {
            print("Failed to start audio analysis: \(error)")
        }
    }
}

// MARK: - Supporting Views

struct VideoPlayerView: View {
    let url: URL
    @Binding var isPlaying: Bool

    @State private var player: AVPlayer?

    var body: some View {
        VideoPlayer(player: player)
            .onAppear {
                player = AVPlayer(url: url)
                if isPlaying {
                    player?.play()
                }
            }
            .onChange(of: url) { _, newURL in
                player = AVPlayer(url: newURL)
                if isPlaying {
                    player?.play()
                }
            }
            .onChange(of: isPlaying) { _, playing in
                if playing {
                    player?.play()
                } else {
                    player?.pause()
                }
            }
    }
}

struct DirectorModeIndicator: View {
    let mode: DirectorMode

    var body: some View {
        Text(mode.rawValue.uppercased())
            .font(.caption.bold())
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(mode == .chaos ? Color.red : Color.blue)
            .foregroundColor(.white)
            .cornerRadius(4)
    }
}

struct AudioLevelsView: View {
    @ObservedObject var analyzer: AudioAnalyzer

    var body: some View {
        HStack(spacing: 8) {
            LevelBar(label: "Sub", value: analyzer.subBassLevel / 100, color: .red)
            LevelBar(label: "Bass", value: analyzer.bassLevel / 100, color: .orange)
            LevelBar(label: "Mid", value: analyzer.midLevel / 100, color: .yellow)
            LevelBar(label: "Hi", value: analyzer.hiHatLevel / 100, color: .green)
            LevelBar(label: "Flux", value: analyzer.spectralFlux / 10, color: .blue)
        }
    }
}

struct LevelBar: View {
    let label: String
    let value: Float
    let color: Color

    var body: some View {
        VStack(spacing: 2) {
            GeometryReader { geo in
                ZStack(alignment: .bottom) {
                    Rectangle()
                        .fill(Color.gray.opacity(0.3))
                    Rectangle()
                        .fill(color)
                        .frame(height: geo.size.height * CGFloat(min(1, max(0, value))))
                }
            }
            Text(label)
                .font(.system(size: 8))
                .foregroundColor(.secondary)
        }
    }
}

struct ControlsView: View {
    @Binding var isPlaying: Bool
    @ObservedObject var clipLibrary: ClipLibrary
    @ObservedObject var director: Director

    var body: some View {
        HStack(spacing: 30) {
            // Previous clip
            Button {
                // Jump to previous (not implemented in this version)
            } label: {
                Image(systemName: "backward.fill")
                    .font(.title2)
            }
            .disabled(clipLibrary.clips.isEmpty)

            // Play/Pause
            Button {
                isPlaying.toggle()
            } label: {
                Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                    .font(.title)
            }

            // Next clip
            Button {
                _ = clipLibrary.nextClip()
            } label: {
                Image(systemName: "forward.fill")
                    .font(.title2)
            }
            .disabled(clipLibrary.clips.isEmpty)

            Spacer()

            // Mode toggle
            Button {
                director.setMode(director.mode == .chaos ? .flow : .chaos)
            } label: {
                Text(director.mode == .chaos ? "CHAOS" : "FLOW")
                    .font(.caption.bold())
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(director.mode == .chaos ? Color.red : Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(6)
            }

            // Clip counter
            Text("\(clipLibrary.clipsPlayedThisLap)/\(clipLibrary.clips.count)")
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }
}

struct ClipLibraryView: View {
    @ObservedObject var clipLibrary: ClipLibrary
    @Environment(\.dismiss) var dismiss

    let columns = [GridItem(.adaptive(minimum: 100))]

    var body: some View {
        NavigationStack {
            ScrollView {
                LazyVGrid(columns: columns, spacing: 10) {
                    ForEach(clipLibrary.clips) { clip in
                        ClipThumbnailView(clip: clip, isCurrent: clip.id == clipLibrary.currentClip?.id)
                            .onTapGesture {
                                clipLibrary.jumpToClip(clip)
                                dismiss()
                            }
                    }
                }
                .padding()
            }
            .navigationTitle("Clip Library (\(clipLibrary.clips.count))")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

struct ClipThumbnailView: View {
    let clip: Clip
    let isCurrent: Bool

    var body: some View {
        VStack(spacing: 4) {
            if let thumbnail = clip.thumbnailImage {
                Image(decorative: thumbnail, scale: 1)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: 100, height: 60)
                    .clipped()
            } else {
                Rectangle()
                    .fill(Color.gray.opacity(0.3))
                    .frame(width: 100, height: 60)
            }

            Text(String(format: "%.1fs", clip.durationSeconds))
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .overlay(
            RoundedRectangle(cornerRadius: 4)
                .stroke(isCurrent ? Color.blue : Color.clear, lineWidth: 2)
        )
    }
}

#Preview {
    ContentView()
        .environmentObject(ClipLibrary())
        .environmentObject(AudioAnalyzer())
        .environmentObject(Director())
        .environmentObject(VideoSegmenter())
}
