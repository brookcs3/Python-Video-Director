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
HOP_LENGTH = 1024

# PACING
FLOW_INTERVAL = random.uniform(9, 14)
FLOW_DURATION = random.uniform(2.2, 4.2)

# --- SONG MAP ---
# Must match Script 1 exactly for sync
SONG_TRIGGERS = [26, 38, 64, 67, 86, 102, 114, 139, 147, 154, 158]

def get_biased_start_frame(total_frames):
    f_100 = 24
    f_300 = 72
    buffer = 48
    max_start_frame = max(0, total_frames - buffer)
    r = random.random()
    if r < 0.70:
        low = f_100
        high = min(f_300, max_start_frame)
        if low >= high: return 0 
        return random.randint(low, high)
    elif r < 0.90:
        low = 6
        high = min(f_100, max_start_frame)
        if low >= high: return 0
        return random.randint(low, high)
    elif r < 0.98:
        low = f_300
        high = max_start_frame
        if low >= high: return max(0, max_start_frame - 10)
        return random.randint(low, high)
    else:
        return 0 

class StrictDeck:
    def __init__(self):
        self.captures = {} 
        self.paths = []
        self.current_idx = 0
        self.available_indices = []
        print("\n--- LOADING WILDCARD DECK ---")
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
        self.available_indices = list(self.captures.keys())

    def pick_next_card(self):
        if not self.available_indices: self.refill_pool()
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
        offset = int(intensity * 65)
        if offset == 0: return frame
        b, g, r = cv2.split(frame)
        b = np.roll(b, offset, axis=1) 
        r = np.roll(r, -offset+4, axis=1) 
        return cv2.addWeighted(frame, 0.5, cv2.merge([b, g, r]), 0.5, 0)

    @staticmethod
    def zoom_punch(frame, intensity):
        if intensity <= 0.2: return frame
        scale = 1.0 + (intensity * 0.3)
        h, w = frame.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2
        zoomed = cv2.resize(frame, (new_w, new_h))
        return zoomed[start_y:start_y+h, start_x:start_x+w]

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
                else:
                    self.last_switch = now
        elif self.state == "FLOW":
            if now - self.break_start > FLOW_DURATION:
                self.state = "CHAOS"
                self.last_switch = now
        return self.state

class AudioBrain:
    def __init__(self):
        self.chroma_hist = collections.deque(maxlen=10)
    def process(self, buffer):
        y = np.frombuffer(buffer, dtype=np.float32)
        onset = librosa.onset.onset_strength(y=y, sr=RATE, hop_length=HOP_LENGTH)
        onset_val = np.mean(onset) if len(onset) > 0 else 0
        S = librosa.feature.melspectrogram(y=y, sr=RATE*2, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        cur = S_db[:, -1]
        sub = (np.mean(cur[0:5]) + 80) / 80.0
        hats = (np.mean(cur[80:128]) + 80) / 180.0
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
    # STEREO
    stream = p.open(format=pyaudio.paFloat32, channels=2, rate=RATE, input=True,
                    input_device_index=idx, frames_per_buffer=CHUNK)

    deck = StrictDeck()
    brain = AudioBrain()
    director = Director()
    
    master_clock = get_biased_start_frame(deck.total_frames)
    frame_counter = 0
    
    # START SONG TIMER
    start_time = time.time()
    print("\nWILDCARD VISUALIZER LIVE. (Start Music Now)")

    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            onset, sub, hats, similarity = brain.process(data)
            mode = director.get_state()
            
            # --- SONG MAP LOGIC ---
            elapsed = time.time() - start_time
            triggers_passed = 0
            for t in SONG_TRIGGERS:
                if elapsed >= t: triggers_passed += 1
            
            # Script 2: Even = Zoom, Odd = Haze (OPPOSITE of Script 1)
            role_idx = triggers_passed % 2
            
            speed = 1
            if hats > 0.95: speed = -1.1 
            elif sub > 1.2: speed = 1.3  
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
                switch_reason = "Rewound"
            
            if mode == "CHAOS":
                if onset > 1.1:
                    should_switch = True
                    switch_reason = "ONSET"
                elif similarity < 0.6:
                    should_switch = True
                    switch_reason = "KEY CHANGE"
            
            frame_counter += 1
            if frame_counter % 10 == 0:
                role_name = "BOUNCE (Zoom)" if role_idx == 0 else "GLITCH (Haze)"
                time_str = time.strftime("%M:%S", time.gmtime(elapsed))
                print(f"[{time_str}] {role_name} | Onset:{onset:.2f}")

            if should_switch:
                print(f"\n>>> SWITCH: {switch_reason}")
                deck.pick_next_card()
                master_clock = get_biased_start_frame(deck.total_frames)

            frame = deck.get_frame(master_clock)
            
            # --- APPLY FX ---
            if role_idx == 0:
                # ROLE A: ZOOM PUNCH
                if sub > 0.8:
                    punch = min((sub - 0.8) * 3, 1.0)
                    frame = FXEngine.zoom_punch(frame, punch)
            else:
                # ROLE B: RGB HAZE
                haze = min(onset / 6.5, 1.0)
                if haze > 0.15: frame = FXEngine.rgb_haze(frame, haze)
            
            if mode == "CHAOS" and onset > 2.5:
                frame = cv2.bitwise_not(frame)

            cv2.imshow('Script 2: Wildcard', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        except KeyboardInterrupt: break
        except Exception as e: print(e); break

    stream.stop_stream()
    p.terminate()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()