"""
Test script for pixelgreat - CRT/LCD style pixelation effect
"""
from PIL import Image
import pixelgreat as pg

# Use the test pattern image
image_in = Image.open("/Users/cameronbrooks/Developer/VideoDir/PM5544.png")
print(f"Input image size: {image_in.size}")

# Test with specific settings from filename:
# TV_pxs10_pxa0.33_dH_ssp0.33_pxp0.25_w0.50_b0.50_bs0.50_r0.50_ssz0.75_sb0.25_sst1.00_bls1.00_gs1.00_pxt
print("Testing CRT_TV with custom settings...")
image_out = pg.pixelgreat(
    image=image_in,
    pixel_size=10,
    output_scale=1,
    screen_type=pg.ScreenType.CRT_TV,
    direction=pg.Direction.HORIZONTAL,
    pixel_aspect=0.33,
    scanline_spacing=0.33,
    pixel_padding=0.25,
    washout=0.50,
    blur=0.50,
    bloom_size=0.50,
    rounding=0.50,
    scanline_size=0.75,
    scanline_blur=0.25,
    scanline_strength=1.00,
    bloom_strength=1.00,
    grid_strength=1.00,
    pixelate=True
)
image_out.save("/Users/cameronbrooks/Desktop/pixelgreat_custom_tv.png")
print("Saved custom CRT TV output")

# Clean up
image_in.close()
image_out.close()

print("\nDone! Check ~/Desktop/pixelgreat_custom_tv.png")
