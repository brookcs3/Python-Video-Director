"""
Video Exporter - HDR-preserving video export for visualizers

Usage:
    # In visualizer, add recording capability:
    from video_exporter import VideoExporter

    exporter = VideoExporter("output.mov", width, height, fps)
    exporter.start()

    # In main loop:
    exporter.write_frame(frame)

    # When done:
    exporter.stop()

    # To combine two recordings side-by-side:
    from video_exporter import combine_videos
    combine_videos("window1.mov", "window2.mov", "combined.mov", layout="horizontal")
"""

import subprocess
import sys
import os


class VideoExporter:
    """HDR-preserving video exporter using FFmpeg pipeline."""

    def __init__(self, output_path, width, height, fps, audio_source=None):
        """
        Initialize exporter.

        Args:
            output_path: Output video file path
            width: Frame width
            height: Frame height
            fps: Frames per second
            audio_source: Optional path to video file to copy audio from
        """
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.audio_source = audio_source
        self.process = None
        self.frame_count = 0
        self.recording = False

    def start(self):
        """Start the FFmpeg encoding process."""
        if self.recording:
            print("Already recording!")
            return False

        ffmpeg_cmd = [
            'ffmpeg',
            '-y',                           # Overwrite output
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{self.width}x{self.height}',
            '-pix_fmt', 'bgr24',            # OpenCV BGR format
            '-r', str(self.fps),
            '-i', '-',                      # Pipe input
        ]

        # Add audio source if provided
        if self.audio_source and os.path.exists(self.audio_source):
            ffmpeg_cmd.extend(['-i', self.audio_source])
            map_args = ['-map', '0:v', '-map', '1:a', '-c:a', 'copy', '-shortest']
        else:
            map_args = []

        ffmpeg_cmd.extend([
            *map_args,
            '-vf', 'format=p010le',         # 10-bit YUV for HDR
            '-c:v', 'hevc_videotoolbox',    # Mac HW encoder
            '-tag:v', 'hvc1',               # QuickTime compatibility
            '-profile:v', 'main10',         # 10-bit HEVC
            '-b:v', '40M',                  # High bitrate for HDR
            '-color_range', 'tv',
            # HDR metadata: BT.2020 + PQ
            '-bsf:v', 'hevc_metadata=colour_primaries=9:transfer_characteristics=16:matrix_coefficients=9',
            self.output_path
        ])

        print(f"[VideoExporter] Starting recording: {self.output_path}")
        print(f"[VideoExporter] Resolution: {self.width}x{self.height} @ {self.fps}fps")

        self.process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL  # Suppress FFmpeg output
        )
        self.recording = True
        self.frame_count = 0
        return True

    def write_frame(self, frame):
        """Write a frame to the video file."""
        if not self.recording or self.process is None:
            return False

        try:
            self.process.stdin.write(frame.tobytes())
            self.frame_count += 1
            return True
        except BrokenPipeError:
            print("[VideoExporter] Error: FFmpeg pipe broke!")
            # Ensure the FFmpeg process is properly cleaned up
            self.stop()
            return False

    def stop(self):
        """Stop recording and finalize the video file."""
        if self.process is None:
            return

        self.process.stdin.close()
        self.process.wait()

        if self.process.returncode == 0:
            print(f"[VideoExporter] SUCCESS: {self.output_path}")
            print(f"[VideoExporter] Total frames: {self.frame_count}")
        else:
            print(f"[VideoExporter] Error: FFmpeg exited with code {self.process.returncode}")

        self.recording = False
        self.process = None


def combine_videos(video1_path, video2_path, output_path, layout="horizontal"):
    """
    Combine two videos into a single output.

    Args:
        video1_path: Path to first video (left/top)
        video2_path: Path to second video (right/bottom)
        output_path: Output combined video path
        layout: "horizontal" (side-by-side) or "vertical" (stacked)

    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(video1_path):
        print(f"Error: {video1_path} not found")
        return False
    if not os.path.exists(video2_path):
        print(f"Error: {video2_path} not found")
        return False

    # Filter for layout
    if layout == "horizontal":
        filter_complex = "[0:v][1:v]hstack=inputs=2[v]"
    elif layout == "vertical":
        filter_complex = "[0:v][1:v]vstack=inputs=2[v]"
    else:
        print(f"Error: Unknown layout '{layout}'. Use 'horizontal' or 'vertical'.")
        return False

    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-i', video1_path,
        '-i', video2_path,
        '-filter_complex', filter_complex,
        '-map', '[v]',
        '-map', '0:a?',                     # Audio from first video if present
        '-vf', 'format=p010le',
        '-c:v', 'hevc_videotoolbox',
        '-tag:v', 'hvc1',
        '-profile:v', 'main10',
        '-b:v', '60M',                      # Higher bitrate for combined
        '-color_range', 'tv',
        '-bsf:v', 'hevc_metadata=colour_primaries=9:transfer_characteristics=16:matrix_coefficients=9',
        '-c:a', 'copy',
        output_path
    ]

    print(f"[Combine] Merging videos ({layout})...")
    print(f"  Input 1: {video1_path}")
    print(f"  Input 2: {video2_path}")
    print(f"  Output:  {output_path}")

    result = subprocess.run(ffmpeg_cmd, stderr=subprocess.PIPE)

    if result.returncode == 0:
        print(f"[Combine] SUCCESS: {output_path}")
        return True
    else:
        print(f"[Combine] Error: FFmpeg failed")
        print(result.stderr.decode())
        return False


# CLI interface for combining videos
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Combine two videos with HDR preservation")
    parser.add_argument("video1", help="First video (left/top)")
    parser.add_argument("video2", help="Second video (right/bottom)")
    parser.add_argument("-o", "--output", default="combined.mov", help="Output path")
    parser.add_argument("-l", "--layout", choices=["horizontal", "vertical"],
                        default="horizontal", help="Layout type")

    args = parser.parse_args()

    success = combine_videos(args.video1, args.video2, args.output, args.layout)
    sys.exit(0 if success else 1)
