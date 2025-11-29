Here is a comprehensive README.md file encapsulating our entire journey, the technical evolution, and the current state of the project. You can save this as `README.md` in your project folder.

------



# Cinematic AI Audio Visualizer (24-Channel Sampler)





### 1. Concept & Goal



The primary goal of this project is to move beyond standard "frequency-bar" music visualization and create a system that emulates **human video editing and VJing techniques** in real-time.

Instead of simply flashing lights to a beat, this tool acts as a **Non-Linear Video Sampler**. It utilizes a bank of 24 related video clips (variations of the same visual theme) and "remixes" them live. The system aims to balance **Chaos** (rapid, glitch-heavy reactions to high-energy audio) with **Flow** (cinematic, uninterrupted playback), mimicking professional editing pacing.



### 2. Current Progress (The "Refined Dynamic" Build)



We are currently running the **Refined Dynamic Threshold Script**. This version solves previous issues of "hyper-activity" and random triggering by implementing:

- **Dynamic Audio Thresholds:** The system calculates a "rolling average" of volume over the last second. Triggers (Kicks/Hats) only fire if the current sound is significantly louder (e.g., 1.2x) than that average, allowing the visualizer to adapt to both quiet intros and loud drops automatically.
- **Strict "Fairness" Deck:** The sampler tracks every clip played. It forces a rotation through all 24 clips before allowing any repeats, ensuring the full breadth of the visual library is used.
- **The "Director" Engine:** A pacing algorithm that randomly switches between "Chaos Mode" (reactive cutting) and "Flow Mode" (forced sustain), preventing viewer fatigue.
- **Cooldown Logic:** A hard limit (0.5s) on clip switching to prevent the "strobe light" effect during rapid-fire drum rolls.



### 3. Techniques & Methods



This project relies on a hybrid of Computer Vision and Digital Signal Processing (DSP).

**The Tech Stack:**

- **Python:** Core logic.
- **Librosa:** Advanced audio feature extraction (Mel Spectrograms, Onset Strength, Chroma/Key Analysis).
- **OpenCV (cv2):** Real-time video manipulation and rendering.
- **PyAudio:** Real-time audio stream capture (via BlackHole).

**Key Algorithms:**

- **Mel-Spectrogram Analysis:** separating audio into **Sub-Bass** (Kick) and **High-Frequency** (Hat) bands to trigger different visual effects.
- **Chroma Similarity:** Detecting harmonic changes (chord progressions). A significant drop in harmonic similarity triggers a "Scene Cut."
- **Elastic Time:** Playback speed is not linear. High-hats trigger negative speed (Rewind), while Kicks trigger positive acceleration (Fast Forward).
- **RGB Haze:** A custom glitch effect that separates the Red and Blue channels and offsets them horizontally based on the intensity of the transient (Kick drum).



### 4. Project History (Evolution)



- **v1 (The Pulse):** Simple volume detection. The entire video flashed brightness based on global loudness.
- **v2 (Frequency Split):** We introduced `FFT` to separate Bass (Red) from Treble (Blue).
- **v3 (The Glitch):** Introduced "Slicing" and "Artifacting" to mimic Chris Cunningham/Aphex Twin aesthetics.
- **v4 (The Sampler):** Moved from single-video manipulation to a **Video Bank** of 24 clips, allowing the script to "jump" between files.
- **v5 (Librosa & Features):** Switched from basic FFT to Librosa to detect *musical features* (Key changes, transients) rather than just volume.
- **v6 (The Director):** Added logic to force "Long Takes" to emulate traditional editing flow.
- **v7 (Current):** Refined math from "Fixed Thresholds" to "Relative Dynamic Thresholds" and added strict cooldowns for a polished, professional output.



### 5. Setup & Requirements



**Prerequisites:**

- **Audio Routing:** macOS **BlackHole 2ch** (or similar loopback driver) set as the input device.

- **Python Libraries:**

  Bash

  ```
  pip install opencv-python pyaudio numpy librosa
  ```

File Structure:

The script expects 24 video files in a specific naming convention:

1. `filename.mp4` (Base file)
2. `filename-2.mp4`
3. `...`
4. `filename-24.mp4`

**Key Tuning Parameters (in the script):**

- `KICK_SENSITIVITY`: How much louder than the average the bass must be to trigger a fast-forward/glitch.
- `HAT_SENSITIVITY`: How much louder than the average the highs must be to trigger a rewind.
- `FLOW_INTERVAL`: Seconds before the "Director" considers taking a break from the chaos.
- `CLIP_SWITCH_COOLDOWN`: Minimum time between video cuts (prevents flickering).