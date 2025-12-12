import cv2
import pyaudio
import numpy as np
import librosa
import random
import os
import collections
import time
from collections import defaultdict

# --- CONFIGURATION ---
BASE_PATH = "/Users/cameronbrooks/Developer/VideoDir/clips/grok-video-d481b7fd-0998-4b3b-82cc-c9e2a5c1aade"
EXTENSION = ".mp4"
TOTAL_CLIPS = 62

RATE = 44100
CHUNK = 2048
HOP_LENGTH = 1024

FLOW_INTERVAL = random.uniform(9, 14)
FLOW_DURATION = random.uniform(2.2, 4.2)

print("FLOW_INTERVAL:", FLOW_INTERVAL)
print("FLOW_DURATION:", FLOW_DURATION)

class StrictDeck:
    def __init__(self):
        self.captures = {} 
        self.paths = []
        self.all_indices = [i for i in range(TOTAL_CLIPS) if i != 65]  # exclude Clip 66
        self.draw_pile = []
        self.special_indices = {58, 59}   # Clip 59 & 60 protected full play + once ever
        self.half_clip_index = 23
        self.lap = 1
        self.once_ever_clips = {58, 59}   # ← Clip 59 & 60 play exactly once ever
        self.once_ever_played = set()

        print("\n--- LOADING STRICT DECK (Rc1 - Clip 59 & 60 once ever) ---")
        self.paths.append(f"{BASE_PATH}{EXTENSION}") 
        for i in range(2, TOTAL_CLIPS + 1):
            self.paths.append(f"{BASE_PATH}-{i}{EXTENSION}")
            
        for i, p in enumerate(self.paths):
            if not os.path.exists(p): 
                continue
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
        self.lap += 1
        print(f">> DECK EMPTY. Reshuffling. Lap {self.lap} <<")
        self.draw_pile = self.all_indices.copy()
        random.shuffle(self.draw_pile)

        # Remove once-ever clips if already played
        for idx in self.once_ever_clips:
            if idx in self.once_ever_played and idx in self.draw_pile:
                self.draw_pile.remove(idx)

    def pick_next_card(self):
        if not self.draw_pile: 
            self.shuffle_deck()
        next_idx = self.draw_pile.pop()
        self.current_idx = next_idx

        # Mark once-ever clips as played forever
        if next_idx in self.once_ever_clips:
            self.once_ever_played.add(next_idx)

        remaining = len(self.draw_pile)
        print(f"Playing Clip {next_idx + 1} (Cards remaining: {remaining})")
        return next_idx

    def get_frame(self, frame_idx):
        cap = self.captures[self.current_idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: 
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return frame

class FXEngine:
    @staticmethod
    def rgb_haze(frame, intensity):
        if intensity <= 0.1:
            return frame
        multiplier = random.uniform(43, 93)  
        offset = int(intensity * multiplier)        
        if offset == 0:
            return frame
        b, g, r = cv2.split(frame)
        b = np.roll(b, offset, axis=1) 
        r = np.roll(r, -offset + 4, axis=1) 
        return cv2.addWeighted(frame, 0.5, cv2.merge([b, g, r]), 0.5, 0)

class Director:
    def __init__(self):
        self.state = "CHAOS"
        self.last_switch = time.time()
        self.break_start = 0
        
    def get_state(self, disable_flow=False):
        now = time.time()
        if self.state == "CHAOS":
            if not disable_flow and now - self.last_switch > FLOW_INTERVAL:
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
        sim = 1.0
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
        if "BlackHole" in info['name']: 
            bh_idx = i
    
    idx = bh_idx if bh_idx is not None else int(input("Index: "))

    stream = p.open(format=pyaudio.paFloat32, channels=2, rate=RATE, input=True,
                    input_device_index=idx, frames_per_buffer=CHUNK)

    deck = StrictDeck()
    brain = AudioBrain()
    director = Director()
    
    master_clock = 0
    frame_counter = 0
    onset_hist = collections.deque(maxlen=5)
    song_start_time = None

    max_seen = defaultdict(int)

    print("\nVisualizer Live. Waiting for Audio (or Auto-Pilot)...")

    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            onset, sub, hats, similarity = brain.process(data)
            
            onset_hist.append(onset)
            steady_rhythm = len(onset_hist) == onset_hist.maxlen and all(o > 1.4 for o in onset_hist)
            
            mode = director.get_state(disable_flow=steady_rhythm)
            
            speed = 1
            speed_str = "Normal"
            
            if hats > 0.95:
                speed = -1.1 
                speed_str = "REWIND (Hat > 0.95)"
            elif sub > 1.2:
                speed = 1.3  
                speed_str = "FAST (Sub > 1.2)"

            master_clock += speed
            
            elapsed = (time.time() - song_start_time) if song_start_time else None
            
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
            
            if deck.current_idx == deck.half_clip_index and master_clock >= deck.total_frames // 2:
                should_switch = True
                switch_reason = "End of Clip (front half only)"
            
            if mode == "CHAOS":
                if onset > 1.2:
                    should_switch = True
                    switch_reason = f"ONSET ({onset:.2f} > 1.2)"
                elif similarity < 0.6:
                    should_switch = True
                    switch_reason = "KEY CHANGE"
            
            if deck.current_idx in deck.special_indices or deck.current_idx == deck.half_clip_index:
                if "ONSET" in switch_reason or switch_reason == "KEY CHANGE":
                    should_switch = False
                    switch_reason = ""

            if should_switch and "ONSET" in switch_reason and song_start_time is None:
                song_start_time = time.time()
                elapsed = 0.0

            if frame_counter % 10 == 0:
                time_str = f"[{elapsed:.1f}s] " if elapsed is not None else ""
                print(f"{time_str}[{mode}] Onset:{onset:.2f} | Sub:{sub:.2f} | Hats:{hats:.2f} | Speed:{speed_str}")

            if should_switch:
                time_str = f"[{elapsed:.1f}s] " if elapsed is not None else ""
                print(f"\n{time_str}>>> SWITCHING CLIP: {switch_reason}")
                deck.pick_next_card()

                if deck.lap > 10 and max_seen[deck.current_idx] >= 0:
                    unseen_tail = deck.total_frames - max_seen[deck.current_idx]
                    if unseen_tail > 0:
                        resume_frame = max_seen[deck.current_idx] + 1
                        master_clock = resume_frame
                        print(f"{time_str}   ↳ TAIL-RESUME Clip {deck.current_idx + 1:02d} from frame {resume_frame} ({unseen_tail/deck.total_frames:.1%} left)")

            if song_start_time is not None:
                current_frame = int(master_clock)
                if current_frame < deck.total_frames:
                    max_seen[deck.current_idx] = max(max_seen[deck.current_idx], current_frame)

            frame = deck.get_frame(master_clock)
            
            haze = min(onset / 6.5, 1.0)
            if haze > 0.15:
                frame = FXEngine.rgb_haze(frame, haze)
            
            if mode == "CHAOS" and onset > 2.5:
                frame = cv2.bitwise_not(frame)

            cv2.imshow('Rc1 - Final Perfect Version', frame)
            
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