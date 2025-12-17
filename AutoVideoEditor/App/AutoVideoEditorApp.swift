import SwiftUI

@main
struct AutoVideoEditorApp: App {
    @StateObject private var clipLibrary = ClipLibrary()
    @StateObject private var audioAnalyzer = AudioAnalyzer()
    @StateObject private var director = Director()
    @StateObject private var segmenter = VideoSegmenter()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(clipLibrary)
                .environmentObject(audioAnalyzer)
                .environmentObject(director)
                .environmentObject(segmenter)
        }
    }
}
