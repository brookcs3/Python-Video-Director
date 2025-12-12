# Cinematic AI Audio Visualizer

A dual-window, audio-reactive video sampler that emulates VJ/video editing techniques in real-time, plus HDR-aware post-processing tools.

---

## Quick Start

### 1. Run the Dual Visualizer
```bash
# Terminal 1
python visualizer_window1.py

# Terminal 2
python visualizer_window2.py
```

### 2. Play Music Through BlackHole
Both windows react to audio routed through BlackHole. They run different clip pools to create visual variety.

### 3. Post-Processing (Optional)
After recording your visualizer output, censor any nudity:
```bash
# Edit INPUT_VIDEO and OUTPUT_VIDEO paths in the script first
python nudity_censor.py
```

---

## Main Scripts

| Script | Purpose |
|--------|---------|
| `visualizer_window1.py` | Primary visualizer window - clips 59/60 play once ever |
| `visualizer_window2.py` | Secondary visualizer window - clip 62 plays once ever |
| `nudity_censor.py` | HDR-aware video post-processor using NudeNet AI |

---

## How It Works

### The Visualizer
- **62 video clips** in a "deck" that shuffles and plays through without repeats
- **Audio analysis** via Librosa: onset detection, sub-bass, hi-hats, chroma/key changes
- **Chaos/Flow modes**: Rapid cuts vs. sustained playback, controlled by a "Director" algorithm
- **Elastic time**: Hi-hats trigger rewind, kicks trigger fast-forward
- **FX**: RGB channel separation ("haze") and inversion on heavy transients

### The Post-Processor
- Uses NudeNet AI to detect and pixelate/blur sensitive content
- HDR-aware: preserves 10-bit color data and Dolby Vision metadata
- Configurable censor style: pixelate, blur, or black box

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/VideoDir.git
cd VideoDir
```

### 2. Create Virtual Environment (for post-processing)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Download the AI Model
The nudity detection model (~40MB) is not included in the repo. Download from:
- [NudeNet 640m.onnx](https://github.com/notAI-tech/NudeNet/releases)

Place it in the `models/` folder:
```
VideoDir/
├── models/
│   └── 640m.onnx      # <-- Put model here
├── clips/             # Video clips go here
├── launcher.py
└── ...
```

### 4. Audio Routing (macOS)
Install [BlackHole 2ch](https://existential.audio/blackhole/) for audio loopback.

## Requirements

**Python 3.8+** with dependencies in `requirements.txt`:
```
opencv-python
pyaudio
librosa
numpy
nudenet
```

**FFmpeg** (required for HDR video processing):
```bash
brew install ffmpeg
```

**Audio Routing:** macOS BlackHole 2ch (or similar loopback) as input device.

**Video Files:** 62 clips in `clips/` folder:
- `clips/grok-video-d481b7fd-0998-4b3b-82cc-c9e2a5c1aade.mp4`
- `clips/grok-video-d481b7fd-0998-4b3b-82cc-c9e2a5c1aade-2.mp4`
- ... through `-62.mp4`

---

## Configuration

### Visualizer Tuning (in script)
- `ONSET_THRESHOLD`: Sensitivity for beat-triggered cuts (lower = more cuts)
- `FLOW_INTERVAL`: Seconds before Director considers a break from chaos
- `FLOW_DURATION`: How long "flow" (no-cut) periods last

### Post-Processor Settings (in script)
- `CENSOR_STYLE`: "pixelate", "blur", or "black"
- `DETECTION_THRESHOLD`: AI confidence threshold (lower = more aggressive)
- `HDR_ENABLED`: Toggle HDR brightness recovery for detection

---

## Archive

Old iterations preserved in `archive/` with meaningful names:

```
archive/
├── visualizer/
│   ├── 01_basic_visualizer.py
│   ├── 02_strict_shuffle.py
│   ├── ...
│   ├── 18_wildcard_song_triggers.py
│   └── 22_rc_clip_62_only.py
│
└── postprocess/
    ├── advanced_hdr_ffmpeg_pipeline.py  (FFmpeg + pixelgreat CRT effects)
    ├── censor_iteration.py
    └── pixelgreat_crt_test.py
```

---

## Project Evolution

1. **v1-v3**: Basic pulse/frequency visualization
2. **v4**: Video bank sampler (multiple clips)
3. **v5**: Librosa audio features (key changes, transients)
4. **v6**: Director algorithm (chaos/flow pacing)
5. **v7**: Dynamic thresholds, strict deck fairness
6. **v8**: Dual-window with once-ever special clips
7. **Current**: Refined tuning + HDR post-processing pipeline
