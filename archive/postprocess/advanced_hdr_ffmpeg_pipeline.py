import cv2
import numpy as np
import os
import subprocess
import json
import time
import sys
from nudenet import NudeDetector
from PIL import Image
from pixelgreat import pixelgreat, Direction, ScreenType

# --- CONFIGURATION ---
# Update these paths to your local environment
INPUT_VIDEO = "/Users/cameronbrooks/Desktop/dd2.mov"
OUTPUT_VIDEO = "/Users/cameronbrooks/Desktop/dd2_censored.mov"
MASK_DIR = "/tmp/masks"
MASK_MOV = "/Users/cameronbrooks/Desktop/dd2_mask.mov"

# --- STYLE SETTINGS ---
BASE_MULTIPLIER = 0.85         # Base expansion factor (1.0 = exact box, higher = larger area)
AUTO_ASPECT = False            # Auto-adjust based on box shape
CENSOR_STYLE = "pixelate"      # "blur", "pixelate"
PIXELATE_BLOCKS = 20        # Target size to downscale to (smaller = more pixelated)
FEATHER_PIXELS = 15      # Feather/blend edge size in pixels (0 = hard edge)
CLASSES_TO_BLOCK = ['FEMALE_BREAST_EXPOSED', 'FEMALE_BREAST_COVERED']
CLASSES_FOR_SHAPE = ['FEMALE_BREAST_EXPOSED_NIPPLE']  # Use these to help define mask shape
CONFIDENCE_THRESHOLD = 0.0001

# --- VERIFICATION SETTINGS ---
VERIFY_CENSORSHIP = False       # Re-check censored frame to ensure nothing visible
MAX_VERIFY_PASSES = 10         # Maximum re-censoring passes per frame (each pass degrades more)

# --- PIXELGREAT CRT SETTINGS ---
USE_PIXELGREAT = False         # Use pixelgreat CRT effect instead of plain pixelation
PG_SCREEN_TYPE = ScreenType.CRT_MONITOR
PG_PIXEL_SIZE = 3
PG_PIXEL_ASPECT = 0.35
PG_DIRECTION = Direction.HORIZONTAL
PG_SCANLINE_SPACING = 0.33
PG_PIXEL_PADDING = 0.1
PG_WASHOUT = 0.90
PG_BLUR = 0.90
PG_BLOOM_SIZE = 0.80
PG_SCANLINE_SIZE = 0.10
PG_SCANLINE_BLUR = 0.9
PG_SCANLINE_STRENGTH = 0.10
PG_BLOOM_STRENGTH = 0.0
PG_GRID_STRENGTH = 0.1

# --- HELPER FUNCTIONS ---

def get_video_metadata(filepath):
    """
    Extracts resolution, frame rate, and HDR color metadata using ffprobe.
    """
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", "-show_frames", "-read_intervals", "%+#1",
        filepath
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode!= 0:
        raise RuntimeError(f"FFprobe failed: {result.stderr}")
    
    data = json.loads(result.stdout)
    video_stream = next((s for s in data['streams'] if s['codec_type'] == 'video'), None)
    
    if not video_stream:
        raise ValueError("No video stream found.")
    
    # Parse FPS - prefer avg_frame_rate for accurate value (r_frame_rate can be timebase)
    fps_str = video_stream.get('avg_frame_rate', video_stream.get('r_frame_rate', '30/1'))
    if '/' in fps_str:
        num, den = map(int, fps_str.split('/'))
        fps = num / den if den != 0 else 48
    else:
        fps = float(fps_str)

    # Detect if HDR (loose check based on transfer characteristics)
    color_transfer = video_stream.get('color_transfer', 'unknown')
    is_hdr = color_transfer == 'smpte2084' or color_transfer == 'arib-std-b67'
    
    # Extract HDR10 mastering display metadata if present
    side_data = video_stream.get('side_data_list', [])
    master_display = None
    content_light = None
    for sd in side_data:
        if sd.get('side_data_type') == 'Mastering display metadata':
            # Format: G(x,y)B(x,y)R(x,y)WP(x,y)L(max,min)
            master_display = f"G({sd.get('green_x', '13250')},{sd.get('green_y', '34500')})" \
                           f"B({sd.get('blue_x', '7500')},{sd.get('blue_y', '3000')})" \
                           f"R({sd.get('red_x', '33000')},{sd.get('red_y', '16000')})" \
                           f"WP({sd.get('white_point_x', '16635')},{sd.get('white_point_y', '16450')})" \
                           f"L({sd.get('max_luminance', '10000000')},{sd.get('min_luminance', '52')})"
        elif sd.get('side_data_type') == 'Content light level metadata':
            max_cll = sd.get('max_content', 1000)
            max_fall = sd.get('max_average', 400)
            content_light = f"{max_cll},{max_fall}"

    return {
        'width': int(video_stream['width']),
        'height': int(video_stream['height']),
        'fps': fps,
        'bitrate': video_stream.get('bit_rate', '35000000'),
        'pix_fmt': 'yuv420p10le',
        'color_primaries': video_stream.get('color_primaries', 'bt2020'),
        'color_transfer': video_stream.get('color_transfer', 'smpte2084'),
        'color_space': video_stream.get('color_space', 'bt2020nc'),
        'is_hdr': is_hdr,
        'master_display': master_display,
        'content_light': content_light
    }

def start_ffmpeg_decoder(filepath):
    """
    Starts FFmpeg to decode the video into raw yuv420p10le stream.
    """
    cmd = [
        "ffmpeg", "-i", filepath,
        "-f", "image2pipe",
        "-pix_fmt", "yuv420p10le",
        "-vcodec", "rawvideo",
        "-"
    ]
    # bufsize is critical for 4K streams to prevent pipe blocking
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)

def start_ffmpeg_encoder(filepath, meta, has_audio=True):
    """
    Starts FFmpeg to encode raw stream back to HEVC Main10 using VideoToolbox.
    Optimized for HDR10 quality preservation with full metadata.
    """
    # Calculate target bitrate - use source bitrate as base
    source_bitrate = int(meta.get('bitrate', 32000000))
    target_bitrate = max(source_bitrate, 35000000)  # At least 35 Mbps for HDR quality
    
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "yuv420p10le",
        "-s", f"{meta['width']}x{meta['height']}",
        "-r", str(meta['fps']),
        "-color_primaries", meta['color_primaries'],
        "-color_trc", meta['color_transfer'],
        "-colorspace", meta['color_space'],
        "-i", "-",
        
        # HDR metadata via filter
        "-vf", "setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc,format=yuv420p10le",
        
        # M4 Hardware Acceleration
        "-c:v", "hevc_videotoolbox",
        "-profile:v", "main10",
        "-b:v", str(target_bitrate),
        "-maxrate", str(int(target_bitrate * 1.5)),
        "-bufsize", str(int(target_bitrate * 2)),
        
        # HDR Tagging
        "-tag:v", "hvc1",
        
        # Output
        filepath
    ]
    
    # Don't pipe stderr - let it go to console for debugging
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=None, bufsize=10**8)

def apply_censorship_yuv10(y, u, v, detections, width, height):
    """
    Applies censorship directly to 10-bit YUV planes.
    Uses shape hints from covered detections to create better masks.
    """
    # Collect all detections by type for smarter mask generation
    exposed_dets = []
    covered_dets = []
    
    for d in detections:
        if d['score'] > CONFIDENCE_THRESHOLD:
            box = d['box']
            det_info = {
                'box': box,
                'score': d['score'],
                'class': d['class'],
                'center': (box[0] + box[2]/2, box[1] + box[3]/2),
                'area': box[2] * box[3]
            }
            if d['class'] in CLASSES_TO_BLOCK:
                exposed_dets.append(det_info)
            elif d['class'] in CLASSES_FOR_SHAPE:
                covered_dets.append(det_info)
    
    for exp in exposed_dets:
        box = exp['box']
        x, y_box, w, h = map(int, box)
        exp_center = exp['center']
        
        # Find overlapping or nearby covered detection for shape guidance
        best_shape_match = None
        best_overlap_score = 0
        
        for cov in covered_dets:
            cov_box = cov['box']
            cx, cy, cw, ch = cov_box
            
            # Calculate overlap (IoU-like) between exposed and covered
            x1 = max(x, cx)
            y1 = max(y_box, cy)
            x2 = min(x + w, cx + cw)
            y2 = min(y_box + h, cy + ch)
            
            if x2 > x1 and y2 > y1:
                overlap_area = (x2 - x1) * (y2 - y1)
                union_area = w * h + cw * ch - overlap_area
                iou = overlap_area / union_area if union_area > 0 else 0
                
                # Also consider center distance
                dist = ((cov['center'][0] - exp_center[0])**2 + 
                        (cov['center'][1] - exp_center[1])**2)**0.5
                dist_score = 1.0 / (1.0 + dist / max(w, h))
                
                # Combined score: IoU + proximity
                score = iou * 0.7 + dist_score * 0.3
                
                if score > best_overlap_score:
                    best_overlap_score = score
                    best_shape_match = cov
        
        # Boundary Checks
        x = max(0, x); y_box = max(0, y_box)
        w = min(width - x, w); h = min(height - y_box, h)
        
        if w <= 0 or h <= 0: continue

        # Apply multipliers - use shape match to influence aspect if available
        if best_shape_match and best_overlap_score > 0.1:
            # Blend the aspect ratios between exposed and covered
            exp_aspect = w / h if h > 0 else 1.0
            cov_w, cov_h = best_shape_match['box'][2], best_shape_match['box'][3]
            cov_aspect = cov_w / cov_h if cov_h > 0 else 1.0
            
            # Weighted blend based on overlap score
            blended_aspect = exp_aspect * (1 - best_overlap_score) + cov_aspect * best_overlap_score
            
            if blended_aspect > 1.0:
                h_mult = BASE_MULTIPLIER
                v_mult = BASE_MULTIPLIER * blended_aspect
            else:
                h_mult = BASE_MULTIPLIER / blended_aspect
                v_mult = BASE_MULTIPLIER
        elif AUTO_ASPECT:
            aspect = w / h if h > 0 else 1.0
            if aspect > 1.0:
                h_mult = BASE_MULTIPLIER
                v_mult = BASE_MULTIPLIER * aspect
            else:
                h_mult = BASE_MULTIPLIER / aspect
                v_mult = BASE_MULTIPLIER
        else:
            h_mult = BASE_MULTIPLIER
            v_mult = BASE_MULTIPLIER
        
        new_w = int(w * h_mult)
        new_h = int(h * v_mult)
        new_x = x + (w - new_w) // 2
        new_y = y_box + (h - new_h) // 2
        
        # Boundary checks again after resizing
        new_x = max(0, new_x); new_y = max(0, new_y)
        new_w = min(width - new_x, new_w)
        new_h = min(height - new_y, new_h)

        if new_w <= 0 or new_h <= 0: continue

        # --- Full RGB Processing for Pixelgreat ---
        roi_y = y[new_y:new_y+new_h, new_x:new_x+new_w].copy()
        
        if CENSOR_STYLE == "pixelate" and USE_PIXELGREAT:
            try:
                # Get matching chroma regions for RGB conversion
                cx, cy = new_x // 2, new_y // 2
                cw, ch = new_w // 2, new_h // 2
                
                if cw <= 0 or ch <= 0:
                    raise ValueError("Chroma region too small")
                
                roi_u = u[cy:cy+ch, cx:cx+cw].copy()
                roi_v = v[cy:cy+ch, cx:cx+cw].copy()
                
                # Upsample chroma to match luma size
                roi_u_full = cv2.resize(roi_u, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                roi_v_full = cv2.resize(roi_v, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                # Convert 10-bit YUV to 8-bit RGB for pixelgreat
                # Scale from 10-bit (0-1023) to 8-bit (0-255)
                y_8 = (roi_y >> 2).astype(np.uint8)
                u_8 = (roi_u_full >> 2).astype(np.uint8)
                v_8 = (roi_v_full >> 2).astype(np.uint8)
                
                # Combine into YCrCb (OpenCV uses YCrCb order, not YCbCr)
                yuv_8 = cv2.merge([y_8, v_8, u_8])  # Y, Cr(V), Cb(U)
                rgb_8 = cv2.cvtColor(yuv_8, cv2.COLOR_YCrCb2RGB)
                
                # Convert to PIL Image for pixelgreat
                pil_img = Image.fromarray(rgb_8)
                
                # Apply pixelgreat CRT effect
                pil_result = pixelgreat(
                    pil_img,
                    screen_type=PG_SCREEN_TYPE,
                    pixel_size=PG_PIXEL_SIZE,
                    pixel_aspect=PG_PIXEL_ASPECT,
                    direction=PG_DIRECTION,
                    scanline_spacing=PG_SCANLINE_SPACING,
                    pixel_padding=PG_PIXEL_PADDING,
                    washout=PG_WASHOUT,
                    blur=PG_BLUR,
                    bloom_size=PG_BLOOM_SIZE,
                    scanline_size=PG_SCANLINE_SIZE,
                    scanline_blur=PG_SCANLINE_BLUR,
                    scanline_strength=PG_SCANLINE_STRENGTH,
                    bloom_strength=PG_BLOOM_STRENGTH,
                    grid_strength=PG_GRID_STRENGTH
                )
                
                # Convert back to numpy and resize to original dimensions
                rgb_result = np.array(pil_result)
                rgb_result = cv2.resize(rgb_result, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                
                # Convert RGB back to YCrCb
                ycrcb_result = cv2.cvtColor(rgb_result, cv2.COLOR_RGB2YCrCb)
                y_result, cr_result, cb_result = cv2.split(ycrcb_result)
                
                # Convert back to 10-bit
                censored = (y_result.astype(np.uint16) << 2)
                censored_u_full = (cb_result.astype(np.uint16) << 2)
                censored_v_full = (cr_result.astype(np.uint16) << 2)
                
                # Downsample chroma for later use
                censored_u = cv2.resize(censored_u_full, (cw, ch), interpolation=cv2.INTER_AREA)
                censored_v = cv2.resize(censored_v_full, (cw, ch), interpolation=cv2.INTER_AREA)
                pixelgreat_chroma = True
            except Exception as e:
                print(f"Pixelgreat error: {e}, falling back to plain pixelate")
                tiny_size = max(1, PIXELATE_BLOCKS)
                small = cv2.resize(roi_y, (tiny_size, tiny_size), interpolation=cv2.INTER_AREA)
                censored = cv2.resize(small, (new_h, new_w), interpolation=cv2.INTER_NEAREST)
                # Apply averaging blur to soften pixelation edges
                censored = cv2.blur(censored, (5, 5))
                pixelgreat_chroma = False
        elif CENSOR_STYLE == "pixelate":
            # First layer: normal pixelation
            tiny_size = max(1, PIXELATE_BLOCKS)
            small = cv2.resize(roi_y, (tiny_size, tiny_size), interpolation=cv2.INTER_AREA)
            censored = cv2.resize(small, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            # Apply averaging blur to soften pixelation edges
            censored = cv2.blur(censored, (5, 5))
            
            # Second layer: coarser pixelation with median blur, 20% opacity overlay
            tiny_size2 = 6
            small2 = cv2.resize(roi_y, (tiny_size2, tiny_size2), interpolation=cv2.INTER_AREA)
            censored2 = cv2.resize(small2, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            censored2 = cv2.medianBlur(censored2.astype(np.uint16), 5)
            # Blend: 80% first layer + 20% second layer
            censored = (censored.astype(np.float32) * 0.8 + censored2.astype(np.float32) * 0.2).astype(np.uint16)
            
            pixelgreat_chroma = False
        elif CENSOR_STYLE == "blur":
            roi_float = roi_y.astype(np.float32)
            # Very strong blur - large kernel with high sigma
            ksize = min(new_w | 1, new_h | 1, 255)  # Ensure odd, max 255
            censored = cv2.GaussianBlur(roi_float, (ksize, ksize), 0).astype(np.uint16)
            pixelgreat_chroma = False
        else:
            censored = roi_y
            pixelgreat_chroma = False
        
        # Always create rounded/elliptical mask for natural look
        mask = np.zeros((new_h, new_w), dtype=np.float32)
        
        # If we have a good shape match, use its geometry to guide the mask
        if best_shape_match and best_overlap_score > 0.15:
            cov_box = best_shape_match['box']
            cov_w, cov_h = int(cov_box[2]), int(cov_box[3])
            cov_aspect = cov_w / cov_h if cov_h > 0 else 1.0
            
            # Calculate rotation angle from the relative positions
            dx = best_shape_match['center'][0] - exp_center[0]
            dy = best_shape_match['center'][1] - exp_center[1]
            angle = np.degrees(np.arctan2(dy, dx)) if abs(dx) > 1 or abs(dy) > 1 else 0
            
            # Create ellipse with shape-matched aspect ratio and slight rotation
            ellipse_w = int(new_w * 0.95)
            ellipse_h = int(ellipse_w / cov_aspect) if cov_aspect > 0 else new_h
            ellipse_h = min(ellipse_h, int(new_h * 0.95))
            ellipse_w = min(ellipse_w, int(new_w * 0.95))
            
            # Apply slight rotation based on relative position
            rotation = angle * 0.4  # Subtle rotation
            cv2.ellipse(mask, (new_w // 2, new_h // 2), 
                       (max(1, ellipse_w // 2), max(1, ellipse_h // 2)), 
                       rotation, 0, 360, 1.0, -1)
        else:
            # Default: simple ellipse filling most of the region
            ellipse_w = int(new_w * 0.95)
            ellipse_h = int(new_h * 0.95)
            cv2.ellipse(mask, (new_w // 2, new_h // 2), 
                       (max(1, ellipse_w // 2), max(1, ellipse_h // 2)), 
                       0, 0, 360, 1.0, -1)
        
        # Feather the edges for smooth blending using bilinear downscale/upscale
        feather = max(FEATHER_PIXELS, 3)  # Minimum feathering for smooth edges
        if new_w > feather * 2 and new_h > feather * 2:
            # Bilinear blur: downscale then upscale for smooth feathering
            shrink_factor = max(1, feather // 2)
            small_size = (max(1, new_w // shrink_factor), max(1, new_h // shrink_factor))
            mask_small = cv2.resize(mask, small_size, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask_small, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = np.clip(mask, 0, 1)
        
        # Blend censored with original using rounded mask
        blended = (roi_y.astype(np.float32) * (1 - mask) + censored.astype(np.float32) * mask).astype(np.uint16)
        y[new_y:new_y+new_h, new_x:new_x+new_w] = blended

        # --- Chroma Processing (U/V) ---
        cx, cy = new_x // 2, new_y // 2
        cw, ch = new_w // 2, new_h // 2
        
        if cw > 0 and ch > 0:
            # Downsample luma mask for chroma
            mask_c = cv2.resize(mask, (cw, ch), interpolation=cv2.INTER_LINEAR)
            
            # If pixelgreat was used, we already have the processed chroma
            if pixelgreat_chroma:
                roi_u_orig = u[cy:cy+ch, cx:cx+cw].copy()
                roi_v_orig = v[cy:cy+ch, cx:cx+cw].copy()
                
                blended_u = (roi_u_orig.astype(np.float32) * (1 - mask_c) + censored_u.astype(np.float32) * mask_c).astype(np.uint16)
                blended_v = (roi_v_orig.astype(np.float32) * (1 - mask_c) + censored_v.astype(np.float32) * mask_c).astype(np.uint16)
                u[cy:cy+ch, cx:cx+cw] = blended_u
                v[cy:cy+ch, cx:cx+cw] = blended_v
            else:
                for plane in [u, v]:
                    roi_c = plane[cy:cy+ch, cx:cx+cw].copy()
                    if CENSOR_STYLE == "pixelate":
                        # First layer: normal pixelation
                        tiny_size = max(1, PIXELATE_BLOCKS)
                        small = cv2.resize(roi_c, (tiny_size, tiny_size), interpolation=cv2.INTER_AREA)
                        censored_c = cv2.resize(small, (cw, ch), interpolation=cv2.INTER_NEAREST)
                        censored_c = cv2.blur(censored_c, (13, 13))
                        
                        # Second layer: coarser pixelation with median blur, 20% opacity
                        tiny_size2 = 6
                        small2 = cv2.resize(roi_c, (tiny_size2, tiny_size2), interpolation=cv2.INTER_AREA)
                        censored_c2 = cv2.resize(small2, (cw, ch), interpolation=cv2.INTER_NEAREST)
                        censored_c2 = cv2.medianBlur(censored_c2.astype(np.uint16), 5)
                        # Blend: 80% first layer + 20% second layer
                        censored_c = (censored_c.astype(np.float32) * 0.8 + censored_c2.astype(np.float32) * 0.2).astype(np.uint16)
                    elif CENSOR_STYLE == "blur":
                        # Very strong blur for chroma
                        ksize = min(cw | 1, ch | 1, 255)  # Ensure odd, max 255
                        censored_c = cv2.GaussianBlur(roi_c.astype(np.float32), (ksize, ksize), 0).astype(np.uint16)
                    else:
                        censored_c = roi_c
                    
                    blended_c = (roi_c.astype(np.float32) * (1 - mask_c) + censored_c.astype(np.float32) * mask_c).astype(np.uint16)
                    plane[cy:cy+ch, cx:cx+cw] = blended_c

    return y, u, v

def main():
    if not os.path.exists(INPUT_VIDEO):
        print(f"Error: {INPUT_VIDEO} not found.")
        return
        
    os.makedirs(MASK_DIR, exist_ok=True)

    print("--- 1. Analyzing HDR Metadata ---")
    meta = get_video_metadata(INPUT_VIDEO)
    print(f"Resolution: {meta['width']}x{meta['height']}")
    print(f"Colorspace: {meta['color_primaries']} | Transfer: {meta['color_transfer']}")
    print(f"Using M4 Hardware Acceleration (hevc_videotoolbox)")

    print("--- 2. Initializing AI Engine (CoreML/ANE) ---")
    # Initialize NudeDetector. 
    # NOTE: To force ANE, we ensure input images are resized to 320x320 
    # before detection to use the specific model graph.
    detector = NudeDetector() 

    print("--- 3. Starting Pipeline ---")
    decoder = start_ffmpeg_decoder(INPUT_VIDEO)
    encoder = start_ffmpeg_encoder(OUTPUT_VIDEO, meta)

    # Calculate buffer sizes for yuv420p10le
    # 2 bytes per pixel
    width = meta['width']
    height = meta['height']
    y_size = width * height * 2
    uv_width = width // 2
    uv_height = height // 2
    uv_size = uv_width * uv_height * 2
    frame_size_bytes = y_size + (uv_size * 2)

    frame_idx = 0
    start_time = time.time()

    try:
        while True:
            # Read Raw Frame
            in_bytes = decoder.stdout.read(frame_size_bytes)
            if not in_bytes:
                break
            
            # Convert bytes to numpy arrays (Planar YUV)
            # Y Plane
            y_plane = np.frombuffer(in_bytes[0:y_size], dtype=np.uint16).reshape((height, width)).copy()
            # U Plane
            u_plane = np.frombuffer(in_bytes[y_size:y_size+uv_size], dtype=np.uint16).reshape((uv_height, uv_width)).copy()
            # V Plane
            v_plane = np.frombuffer(in_bytes[y_size+uv_size:], dtype=np.uint16).reshape((uv_height, uv_width)).copy()

            # --- AI PRE-PROCESSING ---
            # NudeNet expects RGB 8-bit. We create a fast copy.
            # 1. Downshift to 8-bit
            y_8 = (y_plane >> 2).astype(np.uint8)
            u_8 = (u_plane >> 2).astype(np.uint8)
            v_8 = (v_plane >> 2).astype(np.uint8)
            
            # 2. Resize Chroma to 4:4:4 for OpenCV conversion
            u_8_full = cv2.resize(u_8, (width, height), interpolation=cv2.INTER_NEAREST)
            v_8_full = cv2.resize(v_8, (width, height), interpolation=cv2.INTER_NEAREST)
            
            # 3. YCbCr to RGB conversion (Y, Cr, Cb order for OpenCV)
            # U = Cb, V = Cr in standard naming
            ycrcb_merged = cv2.merge([y_8, v_8_full, u_8_full])  # Y, Cr(V), Cb(U)
            rgb_frame = cv2.cvtColor(ycrcb_merged, cv2.COLOR_YCrCb2RGB)
            
            # 4. Resize for ANE Optimization (Static Shape)
            detect_w, detect_h = 640, 640
            rgb_small = cv2.resize(rgb_frame, (detect_w, detect_h))
            
            # Scale factors for detection coordinates
            scale_x = width / detect_w
            scale_y = height / detect_h
            
            def scale_detections(dets):
                """Scale detections from detect size to full resolution."""
                scaled = []
                for d in dets:
                    box = d.get('box', [0, 0, 0, 0])
                    bx, by, bw, bh = box
                    if max(bx, by, bw, bh) <= 1.0:
                        bx, by, bw, bh = bx * detect_w, by * detect_h, bw * detect_w, bh * detect_h
                    scaled_box = [bx * scale_x, by * scale_y, bw * scale_x, bh * scale_y]
                    new_d = d.copy()
                    new_d['box'] = scaled_box
                    scaled.append(new_d)
                return scaled
            
            def has_blocked_detections(dets):
                """Check if any detections match CLASSES_TO_BLOCK."""
                for d in dets:
                    if d['class'] in CLASSES_TO_BLOCK and d['score'] > CONFIDENCE_THRESHOLD:
                        return True
                return False
            
            # --- INITIAL INFERENCE ---
            detections = detector.detect(rgb_small)
            scaled_detections = scale_detections(detections)
            
            # --- ITERATIVE CENSORSHIP WITH VERIFICATION ---
            y_out, u_out, v_out = y_plane, u_plane, v_plane
            verify_pass = 0
            
            while has_blocked_detections(scaled_detections) and verify_pass < MAX_VERIFY_PASSES:
                # Apply censorship to same regions (repeated passes degrade further)
                y_out, u_out, v_out = apply_censorship_yuv10(
                    y_out, u_out, v_out, scaled_detections, width, height
                )
                
                if not VERIFY_CENSORSHIP:
                    break
                
                # Re-create RGB from censored YUV for verification
                y_8_verify = (y_out >> 2).astype(np.uint8)
                u_8_verify = (u_out >> 2).astype(np.uint8)
                v_8_verify = (v_out >> 2).astype(np.uint8)
                u_8_full_v = cv2.resize(u_8_verify, (width, height), interpolation=cv2.INTER_NEAREST)
                v_8_full_v = cv2.resize(v_8_verify, (width, height), interpolation=cv2.INTER_NEAREST)
                ycrcb_verify = cv2.merge([y_8_verify, v_8_full_v, u_8_full_v])
                rgb_verify = cv2.cvtColor(ycrcb_verify, cv2.COLOR_YCrCb2RGB)
                rgb_small_verify = cv2.resize(rgb_verify, (detect_w, detect_h))
                
                # Re-run detection on censored frame
                detections = detector.detect(rgb_small_verify)
                scaled_detections = scale_detections(detections)
                
                verify_pass += 1
                if has_blocked_detections(scaled_detections):
                    sys.stdout.write(f"\r[Frame {frame_idx}] Pass {verify_pass}: Still detecting, re-applying...")
                    sys.stdout.flush()
                    sys.stdout.flush()

            # --- WRITE OUTPUT ---
            encoder.stdin.write(y_out.tobytes())
            encoder.stdin.write(u_out.tobytes())
            encoder.stdin.write(v_out.tobytes())

            # Progress Logging
            frame_idx += 1
            if frame_idx % 48 == 0:
                elapsed = time.time() - start_time
                fps = frame_idx / elapsed
                sys.stdout.write(f"\rProcessing: Frame {frame_idx} @ {fps:.1f} FPS")
                sys.stdout.flush()

    except Exception as e:
        print(f"\nCritical Error: {e}")
    finally:
        print("\nCleaning up pipes...")
        # Close encoder first (it needs all data)
        if encoder:
            try:
                encoder.stdin.close()
                encoder.wait()
            except:
                pass
        # Then terminate decoder (suppress broken pipe messages)
        if decoder:
            try:
                decoder.stdout.close()
                decoder.terminate()
                decoder.wait()
            except:
                pass

    print("--- 4. Muxing Audio ---")
    # The raw video pipe lost the audio. We mux it back from source.
    final_file = OUTPUT_VIDEO.replace(".mov", "_final.mov")
    # Mux audio from original file into the processed video
    mux_cmd = [
        "ffmpeg", "-y",
        "-i", OUTPUT_VIDEO,
        "-i", INPUT_VIDEO,
        "-c", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        final_file
    ]
    subprocess.run(mux_cmd)
    
    # Replace intermediate file
    if os.path.exists(final_file):
        os.replace(final_file, OUTPUT_VIDEO)
        print(f"Success! Output: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()