import cv2
import pyaudio
import numpy as np
import librosa
import random
import os
import collections
import time

# --- CONFIGURATION ---
BASE_PATH = "/Users/cameronbrooks/Developer/VideoDir/grok-video-d481b7fd-0998-4b3b-82cc-c9e2a5c1aade"
EXTENSION = ".mp4"
TOTAL_CLIPS = 62

# AUDIO SETTINGS
RATE = 44100
CHUNK = 2048
HOP_LENGTH = 512

# PACING (Fixed/Tight)
FLOW_INTERVAL = 11.0 
FLOW_DURATION = 3.4 

def get_biased_start_frame(total_frames):
    """
    Weighted Needle Drop with Safety Buffer.
    Ensures we never start so late that the clip ends instantly.
    """
    # Frame approximations (assuming ~24fps)
    f_025 = 6
    f_100 = 24
    f_300 = 72
    
    # BUFFER: Never start within the last 2 seconds (48 frames)
    # This prevents the "Frame 139 of 144" issue.
    buffer = 48
    max_start_frame = max(0, total_frames - buffer)

    r = random.random()
    
    if r < 0.70:
        # 70%: THE SWEET SPOT (1s to 3s)
        low = f_100
        high = min(f_300, max_start_frame)
        if low >= high: return 0 
        return random.randint(low, high)

    elif r < 0.90:
        # 20%: EARLY BUILD (0.25s to 1s)
        low = f_025
        high = min(f_100, max_start_frame)
        if low >= high: return 0
        return random.randint(low, high)

    elif r < 0.98:
        # 8%: LATE START (3s to 4s, but safe)
        low = f_300
        high = max_start_frame
        if low >= high: return max(0, max_start_frame - 10)
        return random.randint(low, high)

    else:
        # 2%: RAW START
        return 0

class StrictDeck:
    def __init__(self):
        self.captures = {} 
        self.paths = []
        self.current_idx = 0
        self.available_indices = []
        
        print("\n--- LOADING ANCHOR DECK ---")
        
        self.paths.append(f"{BASE_PATH}{EXTENSION}") 
        for i in range(2, TOTAL_CLIPS + 1):
            self.paths.append(f"{BASE_PATH}-{i}{EXTENSION}")
            
        for i, p in enumerate(self.paths):
            if not os.path.exists(p): continue
            cap = cv2.VideoCapture(p)
            if cap.isOpened():
                self.captures[i] = cap
                if not hasattr(self, 'total_frames'):
                    self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.refill_pool()
        self.pick_next_card()
        print(f"Deck Online. {len(self.captures)} clips loaded.")

    def refill_pool(self):
        print(f">> POOL EMPTY. Refilling with all {len(self.captures)} clips. <<")
        self.available_indices = list(self.captures.keys())

    def pick_next_card(self):
        if not self.available_indices:
            self.refill_pool()
            
        # RANDOM CHOICE from remaining pool
        next_idx = random.choice(self.available_indices)
        self.available_indices.remove(next_idx)
        self.current_idx = next_idx
        
        remaining_display = sorted([i + 1 for i in self.available_indices])
        print(f"Playing Clip {next_idx + 1} | Pool: {len(remaining_display)} left")
        return next_idx

    def get_frame(self, frame_idx):
        cap = self.captures[self.current_idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return frame

class FXEngine:
    @staticmethod
    def rgb_haze(frame, intensity):
        if intensity <= 0.1: return frame
        # Aggressive offset
        offset = int(intensity * 65)
        if offset == 0: return frame
        b, g, r = cv2.split(frame)
        b = np.roll(b, offset, axis=1) 
        r = np.roll(r, -offset+4, axis=1) 
        return cv2.addWeighted(frame, 0.5, cv2.merge([b, g, r]), 0.5, 0)

class Director:
    def __init__(self):
        self.state = "CHAOS"
        self.last_switch = time.time()
        self.break_start = 0
        
    def get_state(self):
        now = time.time()
        if self.state == "CHAOS":
            if now - self.last_switch > FLOW_INTERVAL:
                if random.random() > 0.7:
                    self.state = "FLOW"
                    self.break_start = now
                    print("\n>>> DIRECTOR: LET IT FLOW (No cuts)")
                else:
                    self.last_switch = now
        elif self.state == "FLOW":
            if now - self.break_start > FLOW_DURATION:
                self.state = "CHAOS"
                self.last_switch = now
                print("\n>>> DIRECTOR: CUT. Back to Chaos.")
        return self.state

class AudioBrain:
    def __init__(self):
        self.chroma_hist = collections.deque(maxlen=10)
        
    def process(self, buffer):
        y = np.frombuffer(buffer, dtype=np.float32)
        
        onset = librosa.onset.onset_strength(y=y, sr=RATE, hop_length=HOP_LENGTH)
        onset_val = np.mean(onset) if len(onset) > 0 else 0
        
        S = librosa.feature.melspectrogram(y=y, sr=RATE, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        cur = S_db[:, -1]
        
        sub = (np.mean(cur[0:5]) + 80) / 80.0
        hats = (np.mean(cur[80:128]) + 80) / 80.0
        
        chroma = librosa.feature.chroma_stft(y=y, sr=RATE, hop_length=HOP_LENGTH)
        curr_chroma = np.mean(chroma, axis=1)
        sim = 0.6
        if len(self.chroma_hist) > 0:
            avg = np.mean(self.chroma_hist, axis=0)
            sim = np.dot(curr_chroma, avg) / (np.linalg.norm(curr_chroma)*np.linalg.norm(avg))
        
        self.chroma_hist.append(curr_chroma)
        return onset_val, sub, hats, sim

def main():
    p = pyaudio.PyAudio()
    
    print("\n--- AUDIO SETUP ---")
    bh_idx = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if "BlackHole" in info['name']: bh_idx = i
    
    idx = bh_idx if bh_idx is not None else int(input("Index: "))

    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True,
                    input_device_index=idx, frames_per_buffer=CHUNK)

    deck = StrictDeck()
    brain = AudioBrain()
    director = Director()
    
    # Bias Start
    master_clock = get_biased_start_frame(deck.total_frames)
    frame_counter = 0
    print("\nANCHOR VISUALIZER LIVE.")

    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            onset, sub, hats, similarity = brain.process(data)
            mode = director.get_state()
            
            speed = 1
            
            # Anchor Logic: Sharp Rewinds
            if hats > 0.95:
                speed = -1.1 
            elif sub > 0.91:
                speed = 1.3  

            master_clock += speed
            
            should_switch = False
            switch_reason = ""
            
            if master_clock >= deck.total_frames:
                should_switch = True
                master_clock = 0
                switch_reason = "End of Clip"
            elif master_clock < 0:
                should_switch = True
                master_clock = deck.total_frames - 1
                switch_reason = "Rewound to Start"
            
            if mode == "CHAOS":
                if onset > 1.1:
                    should_switch = True
                    switch_reason = f"ONSET ({onset:.2f})"
                elif similarity < 0.6:
                    should_switch = True
                    switch_reason = "KEY CHANGE"
            
            frame_counter += 1
            if frame_counter % 10 == 0:
                print(f"[{mode}] Onset:{onset:.2f} | Sub:{sub:.2f}")

            if should_switch:
                print(f"\n>>> SWITCH: {switch_reason}")
                deck.pick_next_card()
                master_clock = get_biased_start_frame(deck.total_frames)
                print(f"   -> Start Frame: {master_clock}")

            frame = deck.get_frame(master_clock)
            
            # FX: RGB Haze (The Anchor signature)
            haze = min(onset / 2.5, 1.0)
            if haze > 0.15:
                frame = FXEngine.rgb_haze(frame, haze)
            
            if mode == "CHAOS" and onset > 2.5:
                frame = cv2.bitwise_not(frame)

            cv2.imshow('Script 1: Anchor', frame)
            
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