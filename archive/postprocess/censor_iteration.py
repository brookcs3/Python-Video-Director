from nudenet import NudeDetector
import cv2
import numpy as np
import os

# --- CONFIGURATION ---
INPUT_VIDEO = "/Users/cameronbrooks/Desktop/Screen Recording 2025-12-01 at 11.25.55 AM.mov"
OUTPUT_VIDEO = "/Users/cameronbrooks/Desktop/Screen Recording 2025-12-01 at 11.25.55 AM_censored.mov"

# This must match the new file you downloaded
MODEL_FILE = "/Users/cameronbrooks/Developer/VideoDir/640m.onnx"

# --- STYLE SETTINGS ---
# Shrink the detected bounding box toward center (1.0 = full box, 0.3 = 30% of box)
HORIZONTAL_MULTIPLIER = 0.7
VERTICAL_MULTIPLIER = 0.7

# Censor style: "blur", "pixelate", or "black"
CENSOR_STYLE = "pixelate"
BLUR_STRENGTH = 93  # Must be odd number. Higher = more blur (try 21-99)
PIXELATE_BLOCKS = 4  # Lower = chunkier pixels (try 2-10)

# Edge feathering
FEATHER_ENABLED = True
FEATHER_AMOUNT = 11  # Pixels of feather/gradient at edges (try 5-30)
COLOR_BURN_EDGE = False  # Apply color burn effect to feathered edge
BURN_INTENSITY = 0.6  # 0.0 = no darkening, 1.0 = full black edge (try 0.1-0.5)

CLASSES_TO_BLOCK = [
    'FEMALE_BREAST_EXPOSED'
]
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
    
    # Check if video loaded
    if width == 0:
        print("Error: Video did not load. Check filename.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    print("Running PARANOID MODE (Threshold: 10%)...")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        # --- FIX 1: COLOR CORRECTION ---
        # OpenCV is BGR, AI needs RGB. We convert a copy for the AI.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect on the RGB copy
        detections = detector.detect(rgb_frame)
        
        found_on_frame = False
        
        for d in detections:
            if d['class'] in CLASSES_TO_BLOCK and d['score'] > 0.10:
                box = d['box']
                x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])

                # Shrink box toward center using multipliers
                new_w = int(w * HORIZONTAL_MULTIPLIER)
                new_h = int(h * VERTICAL_MULTIPLIER)
                new_x = x + (w - new_w) // 2
                new_y = y + (h - new_h) // 2

                # Apply censor effect to the shrunken area
                if new_w > 0 and new_h > 0:
                    roi = frame[new_y:new_y + new_h, new_x:new_x + new_w]

                    if CENSOR_STYLE == "blur":
                        # Gaussian blur
                        blurred = cv2.GaussianBlur(roi, (BLUR_STRENGTH, BLUR_STRENGTH), 0)
                        frame[new_y:new_y + new_h, new_x:new_x + new_w] = blurred

                    elif CENSOR_STYLE == "pixelate":
                        # Pixelate by shrinking then enlarging
                        pixel_size = max(1, PIXELATE_BLOCKS)
                        small = cv2.resize(roi, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
                        pixelated = cv2.resize(small, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

                        if FEATHER_ENABLED:
                            # Create gradient mask - 1.0 in center, 0.0 at edges
                            mask = np.zeros((new_h, new_w), dtype=np.float32)
                            center = (new_w // 2, new_h // 2)

                            # Inner ellipse (sharp pixelation zone)
                            inner_w = max(1, new_w // 2 - FEATHER_AMOUNT)
                            inner_h = max(1, new_h // 2 - FEATHER_AMOUNT)
                            cv2.ellipse(mask, center, (inner_w, inner_h), 0, 0, 360, 1.0, -1)

                            # Blur mask to create smooth gradient
                            feather_kernel = FEATHER_AMOUNT * 2 + 1
                            mask = cv2.GaussianBlur(mask, (feather_kernel, feather_kernel), 0)

                            # Create blurred/smeared version for the edges
                            blur_kernel = FEATHER_AMOUNT * 4 + 1
                            if blur_kernel % 2 == 0:
                                blur_kernel += 1
                            smeared = cv2.GaussianBlur(pixelated, (blur_kernel, blur_kernel), 0)

                            # Blend: sharp pixelated center -> blurry smeared edges
                            mask_3ch = mask[:, :, np.newaxis]
                            result = (pixelated.astype(np.float32) * mask_3ch +
                                     smeared.astype(np.float32) * (1 - mask_3ch))

                            # Apply color burn darkening to edges
                            if COLOR_BURN_EDGE:
                                edge_mask = (1.0 - mask)[:, :, np.newaxis]
                                result = result * (1.0 - edge_mask * BURN_INTENSITY)

                            frame[new_y:new_y + new_h, new_x:new_x + new_w] = np.clip(result, 0, 255).astype(np.uint8)
                        else:
                            frame[new_y:new_y + new_h, new_x:new_x + new_w] = pixelated

                    else:  # "black"
                        cv2.rectangle(frame, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 0, 0), -1)

                print(f"[Frame {frame_count}] Found: {d['class']} ({int(d['score']*100)}%)")
                found_on_frame = True

        out.write(frame)
        
        if frame_count % 10 == 0 and not found_on_frame:
            print(f"Processing frame {frame_count} (Nothing found)...")

    cap.release()
    out.release()
    print(f"Done. Check {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()