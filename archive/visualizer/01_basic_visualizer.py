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
FLOW_INTERVAL = 11.0 
FLOW_DURATION = 3.4 

class StrictDeck:
    def __init__(self):
        self.captures = {} 
        self.paths = []
        self.current_path = ""
        self.all_indices = list(range(TOTAL_CLIPS))
        self.draw_pile = []
        
        print("\n--- LOADING STRICT DECK ---")
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
        
        self.shuffle_deck()
        self.pick_next_card()
        print(f"Deck Online. {len(self.captures)} clips.")

    def shuffle_deck(self):
        print(f">> DECK EMPTY. Reshuffling. <<")
        self.draw_pile = self.all_indices.copy()
        random.shuffle(self.draw_pile)

    def pick_next_card(self):
        if not self.draw_pile: self.shuffle_deck()
        next_idx = self.draw_pile.pop()
        self.current_idx = next_idx
        remaining = len(self.draw_pile)
        print(f"Playing Clip {next_idx + 1} (Cards remaining: {remaining})")
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
        offset = int(intensity * 55)
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

    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True,
                    input_device_index=idx, frames_per_buffer=CHUNK)

    deck = StrictDeck()
    brain = AudioBrain()
    director = Director()
    
    master_clock = 0
    frame_counter = 0
    print("\nVisualizer Live. Waiting for Audio (or Auto-Pilot)...")

    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            onset, sub, hats, similarity = brain.process(data)
            mode = director.get_state()
            
            speed = 1
            speed_str = "Normal"
            
            # --- DEBUGGING THRESHOLDS ---
            # Checking against Hardcoded Limits: Hats > 0.95, Sub > 1.2
            if hats > 0.95:
                speed = -1.1 
                speed_str = "REWIND (Hat > 0.95)"
            elif sub > 1.2:
                speed = 1.3  
                speed_str = "FAST (Sub > 1.2)"

            master_clock += speed
            
            should_switch = False
            switch_reason = ""
            
            # Auto-Pilot
            if master_clock >= deck.total_frames:
                should_switch = True
                master_clock = 0
                switch_reason = "End of Clip"
            elif master_clock < 0:
                should_switch = True
                master_clock = deck.total_frames - 1
                switch_reason = "Rewound to Start"
            
            # Audio Trigger
            if mode == "CHAOS":
                # Hardcoded Onset > 1.1
                if onset > 1.1:
                    should_switch = True
                    switch_reason = f"ONSET ({onset:.2f} > 1.1)"
                elif similarity < 0.6:
                    should_switch = True
                    switch_reason = "KEY CHANGE"
            
            # --- CONSOLE DEBUGGING ---
            frame_counter += 1
            if frame_counter % 10 == 0:
                # Print real-time values vs Hard Limits
                print(f"[{mode}] Onset:{onset:.2f} | Sub:{sub:.2f} (Need > 1.2) | Hats:{hats:.2f} (Need > 0.95) | Speed:{speed_str}")

            if should_switch:
                print(f"\n>>> SWITCHING CLIP: {switch_reason}")
                deck.pick_next_card()

            frame = deck.get_frame(master_clock)
            
            # FX: RGB Haze
            haze = min(onset / 7.5, 1.0)
            if haze > 0.15:
                frame = FXEngine.rgb_haze(frame, haze)
            
            if mode == "CHAOS" and onset > 2.5:
                frame = cv2.bitwise_not(frame)

            cv2.imshow('Script 2: Fixed Debug', frame)
            
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