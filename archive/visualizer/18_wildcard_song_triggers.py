import cv2
import pyaudio
import numpy as np
import librosa
import random
import os
import collections
import time
import warnings

# --- CONFIGURATION ---
BASE_PATH = "/Users/cameronbrooks/Developer/VideoDir/grok-video-d481b7fd-0998-4b3b-82cc-c9e2a5c1aade"
EXTENSION = ".mp4"
TOTAL_CLIPS = 62
RATE = 44100
CHUNK = 2048

# SONG MAP
SONG_TRIGGERS = [13.02,25.23, 28.28, 38.13 ,63.23, 66.28, 76.12, 85.27, 89.02, 114.08, 126.26, 135.17, 139.11, 142.16, 144.14]
SILENCE_TIMEOUT = 10.0


def get_biased_start_frame(total_frames):
    f_100 = 24 
    f_300 = 72
    buffer = 48
    max_start = max(0, total_frames - buffer)
    r = random.random()
    if r < 0.70: return random.randint(f_100, min(f_300, max_start)) 
    elif r < 0.90: return random.randint(6, min(f_100, max_start))   
    elif r < 0.98: return random.randint(f_300, max_start)           
    return 0

class StrictDeck:
    def __init__(self):
        self.captures = {} 
        self.paths = []
        self.current_idx = 0
        self.all_indices = list(range(TOTAL_CLIPS))
        self.draw_pile = []
        self.last_switch_time = 0
        self.active_cap = None
        self.total_frames = 0
        self.width = 0
        self.height = 0
        
        print("\n--- LOADING WILDCARD DECK (LAZY MODE) ---")
        self.paths.append(f"{BASE_PATH}{EXTENSION}") 
        for i in range(2, TOTAL_CLIPS + 1):
            self.paths.append(f"{BASE_PATH}-{i}{EXTENSION}")
            
        valid_count = 0
        for p in self.paths:
            if os.path.exists(p): valid_count += 1
        
        self.shuffle_deck()
        self.pick_next_card()
        print(f"Deck Online. {valid_count} clips indexed.")

    def shuffle_deck(self):
        self.draw_pile = self.all_indices.copy()
        random.shuffle(self.draw_pile)

    def load_clip(self, idx):
        if self.active_cap is not None:
            self.active_cap.release()
        path = self.paths[idx]
        self.active_cap = cv2.VideoCapture(path)
        if self.active_cap.isOpened():
            self.total_frames = int(self.active_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.active_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.active_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            print(f"Error opening: {path}")

    def pick_next_card(self):
        if (time.time() - self.last_switch_time) < 0.5:
            return self.current_idx
        if not self.draw_pile: self.shuffle_deck()
        next_idx = self.draw_pile.pop()
        self.current_idx = next_idx
        self.last_switch_time = time.time()
        self.load_clip(next_idx)
        print(f"Playing Clip {next_idx + 1} | Remaining: {len(self.draw_pile)}")
        return next_idx

    def get_frame(self, frame_idx):
        if self.active_cap is None: return None
        safe_idx = int(frame_idx) % self.total_frames
        self.active_cap.set(cv2.CAP_PROP_POS_FRAMES, safe_idx)
        ret, frame = self.active_cap.read()
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

    @staticmethod
    def luma_pump(frame, intensity):
        if intensity <= 0.1: return frame
        bright_val = int(intensity * 55)
        white = np.full(frame.shape, 255, dtype=np.uint8)
        return cv2.addWeighted(frame, 1.0, white, (intensity * 0.4), 0)

# --- DUAL BRAINS ---
class AnchorBrain:
    def __init__(self):
        self.sub_hist = collections.deque(maxlen=20)
        self.hat_hist = collections.deque(maxlen=20)
        self.chroma_hist = collections.deque(maxlen=10)
        self.hop_length = 1024 
        
    def process(self, y):
        if y.ndim > 1: y = np.mean(y, axis=1)
        if np.max(np.abs(y)) < 0.001: return 0, False, False, 1.0, 0, 0
            
        S = librosa.feature.melspectrogram(y=y, sr=RATE, n_mels=128, hop_length=self.hop_length)
        S_db = librosa.power_to_db(S, ref=np.max)
        cur = S_db[:, -1]
        
        sub = (np.mean(cur[0:5]) + 80) / 80.0
        hats = (np.mean(cur[80:128]) + 80) / 80.0
        
        avg_sub = sum(self.sub_hist) / len(self.sub_hist) if self.sub_hist else 0.01
        avg_hat = sum(self.hat_hist) / len(self.hat_hist) if self.hat_hist else 0.01
        self.sub_hist.append(sub)
        self.hat_hist.append(hats)
        
        is_kick = sub > (avg_sub * 1.1)
        is_hat = hats > (avg_hat * 1.2)
        
        onset = librosa.onset.onset_strength(y=y, sr=RATE, hop_length=self.hop_length)
        onset_val = np.mean(onset) if len(onset) > 0 else 0
        chroma = librosa.feature.chroma_stft(y=y, sr=RATE, hop_length=self.hop_length)
        curr_chroma = np.mean(chroma, axis=1)
        sim = 0.6
        if len(self.chroma_hist) > 0:
            avg = np.mean(self.chroma_hist, axis=0)
            norm_curr = np.linalg.norm(curr_chroma)
            norm_avg = np.linalg.norm(avg)
            if norm_curr > 0 and norm_avg > 0:
                sim = np.dot(curr_chroma, avg) / (norm_curr * norm_avg)
        self.chroma_hist.append(curr_chroma)
        
        return onset_val, is_kick, is_hat, sim, sub, hats

class WildcardBrain:
    def __init__(self):
        self.sub_hist = collections.deque(maxlen=20)
        self.hat_hist = collections.deque(maxlen=20)
        self.chroma_hist = collections.deque(maxlen=10)
        self.hop_length = 1024
        
    def process(self, y):
        if y.ndim > 1: y = np.mean(y, axis=1)
        if np.max(np.abs(y)) < 0.001: return 0, False, False, 1.0, 0, 0
            
        S = librosa.feature.melspectrogram(y=y, sr=RATE, n_mels=128, hop_length=self.hop_length)
        S_db = librosa.power_to_db(S, ref=np.max)
        cur = S_db[:, -1]
        
        sub = (np.mean(cur[0:5]) + 80) / 80.0
        hats = (np.mean(cur[80:128]) + 80) / 180.0
        
        avg_sub = sum(self.sub_hist) / len(self.sub_hist) if self.sub_hist else 0.01
        avg_hat = sum(self.hat_hist) / len(self.hat_hist) if self.hat_hist else 0.01
        self.sub_hist.append(sub)
        self.hat_hist.append(hats)
        
        is_kick = sub > (avg_sub * 1.1)
        is_hat = hats > (avg_hat * 1.2)
        
        onset = librosa.onset.onset_strength(y=y, sr=RATE, hop_length=self.hop_length)
        onset_val = np.mean(onset) if len(onset) > 0 else 0
        chroma = librosa.feature.chroma_stft(y=y, sr=RATE, hop_length=self.hop_length)
        curr_chroma = np.mean(chroma, axis=1)
        sim = 0.6
        if len(self.chroma_hist) > 0:
            avg = np.mean(self.chroma_hist, axis=0)
            norm_curr = np.linalg.norm(curr_chroma)
            norm_avg = np.linalg.norm(avg)
            if norm_curr > 0 and norm_avg > 0:
                sim = np.dot(curr_chroma, avg) / (norm_curr * norm_avg)
        self.chroma_hist.append(curr_chroma)
        
        return onset_val, is_kick, is_hat, sim, sub, hats

class Director:
    def __init__(self):
        self.state = "CHAOS"
        self.last_switch = time.time()
        self.break_start = 0
        
    def get_state(self, role_idx):
        now = time.time()
        
        # DYNAMIC PACING BASED ON ROLE
        # Logic swapped: Role 1 = Anchor (Fixed), Role 0 = Wildcard (Random)
        if role_idx == 1: 
            interval = 11.0
            duration = 3.0
        else: 
            interval = random.uniform(9, 14)
            duration = random.uniform(2.2, 4.2)
            
        if self.state == "CHAOS":
            if now - self.last_switch > interval:
                if random.random() > 0.7:
                    self.state = "FLOW"
                    self.break_start = now
                else:
                    self.last_switch = now
        elif self.state == "FLOW":
            if now - self.break_start > duration:
                self.state = "CHAOS"
                self.last_switch = now
        return self.state

def main():
    p = pyaudio.PyAudio()
    print("\n--- AUDIO SETUP ---")
    bh_idx = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if "BlackHole" in info['name']: bh_idx = i
    idx = bh_idx if bh_idx is not None else int(input("Index: "))
    stream = p.open(format=pyaudio.paFloat32, channels=2, rate=RATE, input=True,
                    input_device_index=idx, frames_per_buffer=CHUNK)

    deck = StrictDeck()
    director = Director()
    
    # INITIALIZE BOTH BRAINS
    anchor_brain = AnchorBrain()
    wildcard_brain = WildcardBrain()
    
    master_clock = 0.0
    frame_counter = 0
    
    song_start_time = None
    audio_started = False
    last_audio_time = time.time()
    
    print("\nWILDCARD VISUALIZER LIVE. (Waiting for Audio...)")

    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            y = np.frombuffer(data, dtype=np.float32)
            
            # --- AUDIO DETECTION ---
            current_time = time.time()
            if not audio_started:
                if np.max(np.abs(y)) > 0.05:
                    song_start_time = current_time 
                    audio_started = True
                    print(">>> SONG START DETECTED (0:00) <<<")
            else:
                if np.max(np.abs(y)) > 0.05:
                    last_audio_time = current_time
                elif (current_time - last_audio_time) > SILENCE_TIMEOUT:
                    print(">>> SILENCE DETECTED: RESETTING CLOCK <<<")
                    audio_started = False
                    song_start_time = None
            
            # --- ROLE SWAP LOGIC ---
            # Default: Role 1 (Wildcard)
            role_idx = 1 
            if audio_started and song_start_time:
                elapsed = current_time - song_start_time
                triggers_passed = 0
                for t in SONG_TRIGGERS:
                    if elapsed >= t: triggers_passed += 1
                
                # INVERTED LOGIC compared to Script 1
                # If Script 1 is 0, Script 2 must be 1.
                if triggers_passed % 2 == 0:
                    role_idx = 1 # Start as Wildcard
                else:
                    role_idx = 0 # Swap to Anchor
            
            # --- HOT SWAP BRAINS ---
            if role_idx == 0:
                # I AM NOW ANCHOR
                onset, is_kick, is_hat, similarity, sub, hats = anchor_brain.process(y)
            else:
                # I AM NOW WILDCARD
                onset, is_kick, is_hat, similarity, sub, hats = wildcard_brain.process(y)
            
            mode = director.get_state(role_idx)
            
            # --- HOT SWAP SPEED LOGIC ---
            speed = 1.0
            if role_idx == 0: # ANCHOR SPEED
                if hats > 0.95: speed = -1.5 
                elif sub > 0.98: speed = 1.3
            else: # WILDCARD SPEED
                if hats > 0.95: speed = -1.1 
                elif sub > 0.98: speed = 1.3
            
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
            
            # --- HOT SWAP TRIGGER LOGIC ---
            if mode == "CHAOS":
                if role_idx == 0: # ANCHOR TRIGGERS
                    if onset > 2.0: should_switch = True; switch_reason = "ANCHOR ONSET"
                    elif similarity < 0.45: should_switch = True; switch_reason = "ANCHOR KEY"
                else: # WILDCARD TRIGGERS
                    if onset > 1.1: should_switch = True; switch_reason = "WILD ONSET"
                    elif similarity < 0.6: should_switch = True; switch_reason = "WILD KEY"

            if frame_counter % 10 == 0:
                current_persona = "ANCHOR" if role_idx == 0 else "WILDCARD"
                print(f"[{current_persona}] {mode} | Onset:{onset:.2f} | Sub:{sub:.2f}")
            frame_counter += 1

            if should_switch:
                print(f"> SWITCH: {switch_reason}")
                deck.pick_next_card()
                master_clock = get_biased_start_frame(deck.total_frames)

            frame = deck.get_frame(master_clock)
            if frame is None: continue

            # --- HOT SWAP FX ---
            if role_idx == 0:
                # ANCHOR FX (RGB HAZE)
                if is_kick: frame = FXEngine.rgb_haze(frame, 0.8)
            else:
                # WILDCARD FX (LUMA PUMP)
                if is_kick:
                    intensity = min(onset / 2.0, 1.0)
                    frame = FXEngine.luma_pump(frame, intensity)
            
            if mode == "CHAOS" and onset > 2.5:
                frame = cv2.bitwise_not(frame)

            cv2.imshow('Script 2: Wildcard', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        except KeyboardInterrupt: break
        except Exception as e: print(f"Error: {e}"); break

    stream.stop_stream()
    p.terminate()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()