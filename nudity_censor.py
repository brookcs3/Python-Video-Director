from nudenet import NudeDetector
import cv2
import numpy as np
import os
import subprocess
import sys
import json

# --- LAUNCHER CONFIG (auto-populated by launcher.py) ---
TEMP_CONFIG = "/tmp/nudity_censor_config.json"
if os.path.exists(TEMP_CONFIG):
    with open(TEMP_CONFIG, 'r') as f:
        _launcher_config = json.load(f)
    INPUT_VIDEO = _launcher_config.get('input_video', "/Users/cameronbrooks/Desktop/RT.mov")
    OUTPUT_VIDEO = _launcher_config.get('output_video', "/Users/cameronbrooks/Desktop/RT-C.mov")
    MODEL_FILE = _launcher_config.get('model_path', "/Users/cameronbrooks/Developer/VideoDir/models/640m.onnx")
else:
    # --- CONFIGURATION (defaults for standalone use) ---
    INPUT_VIDEO = "/Users/cameronbrooks/Desktop/RT.mov"
    OUTPUT_VIDEO = "/Users/cameronbrooks/Desktop/RT-C.mov"
    MODEL_FILE = "/Users/cameronbrooks/Developer/VideoDir/models/640m.onnx"

# --- STYLE SETTINGS ---
HORIZONTAL_MULTIPLIER = 0.7
VERTICAL_MULTIPLIER = 0.7
CENSOR_STYLE = "pixelate"
BLUR_STRENGTH = 93
PIXELATE_BLOCKS = 4
FEATHER_ENABLED = True
FEATHER_AMOUNT = 11
COLOR_BURN_EDGE = False
BURN_INTENSITY = 0.6

# --- HDR SUPPORT ---
HDR_ENABLED = True
HDR_MULTIPLIER = 1.0
CLASSES_TO_BLOCK = ['FEMALE_BREAST_EXPOSED']
DETECTION_THRESHOLD = 0.02


def main():
    if not os.path.exists(MODEL_FILE):
        print(f"ERROR: Missing {MODEL_FILE}")
        return

    print(f"Loading AI Model: {MODEL_FILE}...")
    detector = NudeDetector(model_path=MODEL_FILE, inference_resolution=640)
    print("Model Loaded.")

    cap = cv2.VideoCapture(INPUT_VIDEO)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if width == 0:
        print("Error: Video did not load. Check filename and permissions.")
        return

    print(f"Source: {width}x{height} @ {fps}fps")

    # --- FFMPEG PIPELINE SETUP ---
    # Uses hardware HEVC encoder with proper HDR metadata preservation
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',                           # Overwrite output if it exists
        '-f', 'rawvideo',               # Input format for the pipe
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',      # Size of incoming raw frames
        '-pix_fmt', 'bgr24',            # Pixel format (OpenCV's default)
        '-r', str(fps),                 # Frame rate
        '-i', '-',                      # Input from Pipe (stdin)
        '-i', INPUT_VIDEO,              # Input 1: Original file (for Audio)

        '-map', '0:v',                  # Use Video stream from Pipe (input 0)
        '-map', '1:a',                  # Use Audio stream from Original file (input 1)

        '-vf', 'format=p010le',         # Convert 8-bit input to 10-bit YUV for main10 profile

        '-c:v', 'hevc_videotoolbox',    # Mac Hardware Encoder (Fast)
        '-tag:v', 'hvc1',               # Tag as hvc1 for QuickTime Player compatibility
        '-profile:v', 'main10',         # Force 10-bit HEVC profile
        '-b:v', '40M',                  # High Video Bitrate (40Mbps for HDR)
        '-color_range', 'tv',

        # BITSTREAM FILTER: Force HDR Metadata
        # 9=BT.2020, 16=SMPTE ST 2084 (PQ), 9=BT.2020nc
        '-bsf:v', 'hevc_metadata=colour_primaries=9:transfer_characteristics=16:matrix_coefficients=9',

        '-c:a', 'copy',                 # Copy Audio Stream (no re-encode)
        '-shortest',                    # Stop when shortest stream ends
        OUTPUT_VIDEO
    ]

    print("Starting FFmpeg Pipeline...")
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=sys.stderr)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # === AI Detection ===
        rgb_for_ai = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if HDR_ENABLED:
            rgb_for_ai = rgb_for_ai.astype(np.float32) * HDR_MULTIPLIER
            rgb_for_ai = np.clip(rgb_for_ai, 0, 255).astype(np.uint8)

        detections = detector.detect(rgb_for_ai)

        for d in detections:
            if d['class'] in CLASSES_TO_BLOCK and d['score'] > DETECTION_THRESHOLD:
                box = d['box']
                x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])

                new_w = int(w * HORIZONTAL_MULTIPLIER)
                new_h = int(h * VERTICAL_MULTIPLIER)
                new_x = x + (w - new_w) // 2
                new_y = y + (h - new_h) // 2

                if new_w <= 0 or new_h <= 0:
                    continue

                # Clamp to frame boundaries
                new_y = max(0, new_y)
                new_x = max(0, new_x)
                end_y = min(height, new_y + new_h)
                end_x = min(width, new_x + new_w)

                new_h = end_y - new_y
                new_w = end_x - new_x

                if new_w <= 0 or new_h <= 0:
                    continue

                roi = frame[new_y:end_y, new_x:end_x]
                if roi.size == 0:
                    continue

                if CENSOR_STYLE == "pixelate":
                    pixel_size = max(1, PIXELATE_BLOCKS)
                    small = cv2.resize(roi, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
                    pixelated = cv2.resize(small, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                    frame[new_y:end_y, new_x:end_x] = pixelated

                elif CENSOR_STYLE == "blur":
                    blurred = cv2.GaussianBlur(roi, (BLUR_STRENGTH, BLUR_STRENGTH), 0)
                    frame[new_y:end_y, new_x:end_x] = blurred

                else:  # "black"
                    cv2.rectangle(frame, (new_x, new_y), (end_x, end_y), (0, 0, 0), -1)

                print(f"[Frame {frame_count}] Censored {d['class']} ({int(d['score']*100)}%)")

        # === Write processed frame to FFmpeg's stdin ===
        try:
            process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            print("Error: FFmpeg pipe broke. Is FFmpeg installed and in your PATH?")
            break

        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...", end='\r')

    cap.release()

    # --- Close the pipe and wait for FFmpeg to finish ---
    process.stdin.close()
    process.wait()

    if process.returncode == 0:
        print(f"\nSUCCESS! HDR-preserving output with audio saved as: {OUTPUT_VIDEO}")
        print("Play it to confirm audio, HDR, and censorship.")
    else:
        print(f"\nFFmpeg process exited with error code {process.returncode}.")
        print("Check FFmpeg's output for details.")


if __name__ == "__main__":
    main()
