import cv2
import pyaudio
import numpy as np
import librosa
import random
import os
import collections

# --- CONFIGURATION ---
BASE_PATH = "/Users/cameronbrooks/Downloads/grok-video-d481b7fd-0998-4b3b-82cc-c9e2a5c1aade"
EXTENSION = ".mp4"
TOTAL_CLIPS = 24

# AUDIO SETTINGS
RATE = 44100
CHUNK = 2048
HOP_LENGTH = 1024

class VideoBank:
    """Manages 24 video files like a sampler bank"""
    def __init__(self):
        self.captures = []
        self.paths = []
        self.current_idx = 0
        self.total_frames = 0
        
        print("\n--- LOADING VIDEO BANK (This may take a moment) ---")
        
        # 1. Generate Paths
        # Pattern: file.mp4, file-2.mp4, ... file-24.mp4
        self.paths.append(f"{BASE_PATH}{EXTENSION}") # The first one (no number)
        for i in range(2, TOTAL_CLIPS + 1):
            self.paths.append(f"{BASE_PATH}-{i}{EXTENSION}")
            
        # 2. Open Captures
        for i, path in enumerate(self.paths):
            if not os.path.exists(path):
                print(f"Warning: File not found: {path}")
                continue
                
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print(f"Failed to open: {path}")
                continue
            
            # Use the first valid clip to set the master frame count
            if self.total_frames == 0:
                self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.captures.append(cap)
            print(f"Loaded Clip {i+1}/{TOTAL_CLIPS}")

        if not self.captures:
            raise Exception("No videos loaded! Check your file paths.")
        
        print(f"Bank Ready. {len(self.captures)} Clips Online.")

    def get_frame(self, master_frame_idx, clip_idx=None):
        """Reads a frame from a specific clip at a specific time"""
        if clip_idx is None: clip_idx = self.current_idx
        
        # Wrap index safely
        clip_idx = clip_idx % len(self.captures)
        
        cap = self.captures[clip_idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, master_frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            # Fallback if read fails (end of loop?)
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
        return frame

    def switch_clip(self, mode="random"):
        """Logic to hop between clips"""
        if mode == "random":
            # Hard jump (Teleport)
            self.current_idx = random.randint(0, len(self.captures) - 1)
        elif mode == "next":
            # Slide to neighbor
            self.current_idx = (self.current_idx + 1) % len(self.captures)
        elif mode == "prev":
            self.current_idx = (self.current_idx - 1) % len(self.captures)
            
        return self.current_idx

class AudioBrain:
    def __init__(self):
        self.chroma_hist = collections.deque(maxlen=10)
        
    def listen(self, buffer):
        # Float32 conversion for Librosa
        y = np.frombuffer(buffer, dtype=np.float32)
        
        # 1. Onset Strength (The "Kick" Detector)
        onset = librosa.onset.onset_strength(y=y, sr=RATE, hop_length=HOP_LENGTH)
        onset_val = np.mean(onset) if len(onset) > 0 else 0
        
        # 2. Bands (Spectral Energy)
        S = librosa.feature.melspectrogram(y=y, sr=RATE, n_mels=128, fmax=12000, hop_length=HOP_LENGTH)
        S_db = librosa.power_to_db(S, ref=np.max)
        current = S_db[:, -1] # Last slice
        
        sub_bass = (np.mean(current[0:5]) + 80) / 80.0
        hi_hats = (np.mean(current[80:128]) + 80) / 80.0
        
        # 3. Chroma (Key Change Detector)
        chroma = librosa.feature.chroma_stft(y=y, sr=RATE, hop_length=HOP_LENGTH)
        curr_chroma = np.mean(chroma, axis=1)
        
        similarity = 0.67
        if len(self.chroma_hist) > 0:
            avg_hist = np.mean(self.chroma_hist, axis=0)
            norm_a = np.linalg.norm(curr_chroma)
            norm_b = np.linalg.norm(avg_hist)
            if norm_a > 0 and norm_b > 0:
                similarity = np.dot(curr_chroma, avg_hist) / (norm_a * norm_b)
        
        self.chroma_hist.append(curr_chroma)
        
        return onset_val, sub_bass, hi_hats, similarity

class FXProcess:
    @staticmethod
    def apply(frame, onset, sub, similarity):
        # 1. RGB SPLIT (triggered by low similarity/key change)
        if similarity < 0.85:
            shift = int((1.0 - similarity) * 40)
            b, g, r = cv2.split(frame)
            b = np.roll(b, shift, axis=1)
            r = np.roll(r, -shift, axis=1)
            frame = cv2.merge([b, g, r])

        # 2. ZOOM PUNCH (triggered by Sub Bass)
        if sub > 0.8:
            h, w = frame.shape[:2]
            zoom = 1.0 + (sub - 0.8) * 0.6 # Scale factor
            
            # Crop center
            cx, cy = w//2, h//2
            new_w, new_h = int(w / zoom), int(h / zoom)
            x1 = cx - new_w // 2
            y1 = cy - new_h // 3
            
            crop = frame[y1:y1+new_h, x1:x1+new_w]
            if crop.size > 0:
                frame = cv2.resize(crop, (w, h))

        # 3. INVERT (triggered by extreme Onset)
        if onset > 2.0:
            frame = cv2.bitwise_not(frame)
            
        return frame

def main():
    p = pyaudio.PyAudio()
    
    # DEVICE SELECTION
    print("\n--- AUDIO DEVICES ---")
    blackhole_idx = None
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if "BlackHole" in dev['name']: blackhole_idx = i
            
    idx = blackhole_idx if blackhole_idx is not None else int(input("Enter BlackHole Index: "))

    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True,
                    input_device_index=idx, frames_per_buffer=CHUNK)

    # INITIALIZE SYSTEMS
    bank = VideoBank()
    brain = AudioBrain()
    
    master_clock = 0
    print("\nVisualizer Live. Playing from Bank of 24.")

    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            onset, sub, hats, similarity = brain.listen(data)
            
            # --- THE DIRECTOR LOGIC (Moving clips) ---
            
            # 1. Hard Beat (Onset) -> Random Jump
            if onset > 1.1:
                bank.switch_clip(mode="random")
                
            # 2. Key Change / Chord Change -> Slide Next
            elif similarity < 0.6:
                bank.switch_clip(mode="next")
                
            # 3. High Energy Hats -> Flash a distant clip temporarily?
            # (We stick to the current clip but maybe advance time faster)
            speed = 1
            if sub > 0.8: speed = 1.3 # Fast forward on bass
            if hats > 0.9: speed = -1.1 # Rewind/Glitch on Hi-Hats?
            
            # --- RENDER ---
            master_clock = (master_clock + speed) % bank.total_frames
            
            # Ensure index is positive
            if master_clock < 0: master_clock += bank.total_frames
            
            raw_frame = bank.get_frame(master_clock)
            final_frame = FXProcess.apply(raw_frame, onset, sub, similarity)
            
            # Overlay Info (Optional)
            # cv2.putText(final_frame, f"CLIP: {bank.current_idx + 1}", (50, 50), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('24-Clip Sampler', final_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            break

    stream.stop_stream()
    p.terminate()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()