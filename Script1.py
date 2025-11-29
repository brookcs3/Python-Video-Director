import cv2
import pyaudio
import numpy as np
import librosa
import random
import os
import collections
import time

# --- CONFIGURATION ---
BASE_PATH = "/Users/cameronbrooks/Downloads/grok-video-d481b7fd-0998-4b3b-82cc-c9e2a5c1aade"
EXTENSION = ".mp4"
TOTAL_CLIPS = 24

# AUDIO SETTINGS
RATE = 44100
CHUNK = 2048
HOP_LENGTH = 1024

# SENSITIVITY (Corrected Logic: Higher = Harder to trigger)
# 1.1 = Trigger if 10% louder than average
# 1.3 = Trigger if 30% louder than average
KICK_SENSITIVITY = 1.1
HAT_SENSITIVITY = 1.2

# PACING
FLOW_INTERVAL = 11.0
FLOW_DURATION = 3.0
CLIP_SWITCH_COOLDOWN = 0.5 # Minimum seconds between cuts

class StrictDeck:
    def __init__(self):
        self.captures = {} 
        self.paths = []
        self.current_idx = 0
        self.all_indices = list(range(TOTAL_CLIPS))
        self.draw_pile = []
        self.last_switch_time = 0 # Track cooldown
        
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
        # Enforce Cooldown
        if (time.time() - self.last_switch_time) < CLIP_SWITCH_COOLDOWN:
            return self.current_idx

        if not self.draw_pile: self.shuffle_deck()
        self.current_idx = self.draw_pile.pop()
        self.last_switch_time = time.time()
        return self.current_idx

    def get_frame(self, frame_idx):
        cap = self.captures[self.current_idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return frame

class FXEngine:
    @staticmethod
    def rgb_haze(frame, intensity):
        if intensity <= 0.2: return frame
        offset = int(intensity * 55)
        if offset == 0: return frame
        b, g, r = cv2.split(frame)
        b = np.roll(b, offset, axis=1)
        r = np.roll(r, -offset+4, axis=1)
        return cv2.addWeighted(frame, 0.6, cv2.merge([b, g, r]), 0.4, 0)

class AudioBrain:
    def __init__(self):
        self.sub_hist = collections.deque(maxlen=20)
        self.hat_hist = collections.deque(maxlen=20)
        self.chroma_hist = collections.deque(maxlen=10)
        
    def process(self, buffer):
        y = np.frombuffer(buffer, dtype=np.float32)
        
        # Features
        S = librosa.feature.melspectrogram(y=y, sr=RATE, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        cur = S_db[:, -1]
        
        sub_raw = (np.mean(cur[0:5]) + 80) / 80.0
        hat_raw = (np.mean(cur[80:128]) + 80) / 480.0
        
        # --- FIXED MATH ---
        # 1. Get Averages
        avg_sub = sum(self.sub_hist) / len(self.sub_hist) if self.sub_hist else 0.01
        avg_hat = sum(self.hat_hist) / len(self.hat_hist) if self.hat_hist else 0.01
        
        # 2. Update History
        self.sub_hist.append(sub_raw)
        self.hat_hist.append(hat_raw)
        
        # 3. Calculate Targets (Multiply, don't Divide)
        kick_target = avg_sub * KICK_SENSITIVITY
        hat_target = avg_hat * HAT_SENSITIVITY
        
        is_kick = sub_raw > kick_target
        is_hat = hat_raw > hat_target
        
        onset = librosa.onset.onset_strength(y=y, sr=RATE, hop_length=HOP_LENGTH)
        onset_val = np.mean(onset) if len(onset) > 0 else 0
        
        chroma = librosa.feature.chroma_stft(y=y, sr=RATE, hop_length=HOP_LENGTH)
        curr_chroma = np.mean(chroma, axis=1)
        sim = 0.6
        if len(self.chroma_hist) > 0:
            avg = np.mean(self.chroma_hist, axis=0)
            sim = np.dot(curr_chroma, avg) / (np.linalg.norm(curr_chroma)*np.linalg.norm(avg))
        self.chroma_hist.append(curr_chroma)
        
        return onset_val, is_kick, is_hat, sim, sub_raw, hat_raw, avg_sub, avg_hat, kick_target

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
                    print("\n>>> DIRECTOR EVENT: Entering FLOW Mode")
                else:
                    self.last_switch = now
        elif self.state == "FLOW":
            if now - self.break_start > FLOW_DURATION:
                self.state = "CHAOS"
                self.last_switch = now
                print("\n>>> DIRECTOR EVENT: Back to CHAOS Mode")
        return self.state

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
    print("\nVisualizer Live. (Dynamic Corrected).")

    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            onset, is_kick, is_hat, similarity, sub_raw, hat_raw, avg_sub, avg_hat, kick_target = brain.process(data)
            mode = director.get_state()
            
            speed = 1
            speed_str = "Normal"
            
            # Changed Priority: Kick overrides Hat for cleaner rhythm
            if is_kick:
                speed = -1.5 
                speed_str = "REWIND (Kick)"
            elif is_hat:
                speed = 1.3 
                speed_str = "FAST (Hat)"

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
            
            # Audio Interrupt (Chaos Mode Only)
            if mode == "CHAOS":
                # Raised Onset Threshold from 1.0 to 2.0 based on your logs
                if onset > 2.0:
                    should_switch = True
                    switch_reason = f"ONSET SPIKE ({onset:.2f} > 1.8)"
                # Lowered Similarity Threshold from 0.6 to 0.45
                elif similarity < 0.45:
                    should_switch = True
                    switch_reason = f"KEY CHANGE ({similarity:.2f} < 0.45)"

            # --- DEBUG LOGGING ---
            frame_counter += 1
            if frame_counter % 10 == 0:
                # This log now shows the CORRECT math (Target is higher than average)
                print(f"[{mode}] Onset:{onset:.2f} | Sub:{sub_raw:.2f} (Target > {kick_target:.2f}) | Speed:{speed_str}")

            if should_switch:
                # Logic inside deck will prevent switch if too fast (Cooldown)
                prev_idx = deck.current_idx
                new_idx = deck.pick_next_card()
                if prev_idx != new_idx:
                     print(f"\n>>> SWITCHING CLIP: {switch_reason}")

            frame = deck.get_frame(master_clock)
            
            # FX
            if is_kick:
                frame = FXEngine.rgb_haze(frame, 0.8)
            
            if mode == "CHAOS" and onset > 2.5:
                frame = cv2.bitwise_not(frame)

            cv2.imshow('Script 1 Refined', frame)
            
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