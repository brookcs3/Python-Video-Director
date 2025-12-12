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

# PACING
FLOW_INTERVAL = 15.0  # Seconds of Chaos before a "Break"
FLOW_DURATION = 5.0   # How long the "Break" lasts

class StrictDeck:
    """
    Ensures EVERY clip is played exactly once before any clip is repeated.
    Tracks usage regardless of whether it was triggered by Audio or Auto-Pilot.
    """
    def __init__(self):
        self.captures = {} 
        self.paths = []
        self.current_path = ""
        
        # The Deck
        self.all_indices = list(range(TOTAL_CLIPS)) # [0, 1, 2 ... 23]
        self.draw_pile = [] # The cards remaining to be drawn
        
        print("\n--- LOADING STRICT DECK ---")
        
        # 1. Generate Paths
        self.paths.append(f"{BASE_PATH}{EXTENSION}") 
        for i in range(2, TOTAL_CLIPS + 1):
            self.paths.append(f"{BASE_PATH}-{i}{EXTENSION}")
            
        # 2. Open Captures
        for i, p in enumerate(self.paths):
            if not os.path.exists(p):
                print(f"MISSING: {p}")
                continue
            
            cap = cv2.VideoCapture(p)
            if cap.isOpened():
                self.captures[i] = cap # Map Index -> Video Object
                
                # Get dimensions from first valid clip
                if not hasattr(self, 'total_frames'):
                    self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 3. Initial Shuffle
        self.shuffle_deck()
        self.pick_next_card()
        print(f"Deck Online. {len(self.captures)} clips loaded.")

    def shuffle_deck(self):
        """Resets the pile when empty"""
        print(f">> DECK EMPTY. Reshuffling all {TOTAL_CLIPS} clips. <<")
        self.draw_pile = self.all_indices.copy()
        random.shuffle(self.draw_pile)

    def pick_next_card(self):
        """Pulls the next unique index from the pile"""
        if not self.draw_pile:
            self.shuffle_deck()
            
        # Draw card
        next_idx = self.draw_pile.pop()
        
        # Determine Path
        self.current_idx = next_idx
        
        # Debug Info
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
        """Your requested RGB Blur Effect"""
        if intensity <= 0.1: return frame
        
        offset = int(intensity * 25) # Max 25px shift
        if offset == 0: return frame
        
        b, g, r = cv2.split(frame)
        b = np.roll(b, offset, axis=1) # Shift Blue Right
        r = np.roll(r, -offset, axis=1) # Shift Red Left
        
        # Add slight transparency/ghosting
        merged = cv2.merge([b, g, r])
        return cv2.addWeighted(frame, 0.5, merged, 0.5, 0)

class Director:
    def __init__(self):
        self.state = "CHAOS"
        self.last_switch = time.time()
        self.break_start = 0
        
    def get_state(self):
        now = time.time()
        if self.state == "CHAOS":
            if now - self.last_switch > FLOW_INTERVAL:
                # 30% chance to enter Flow Mode
                if random.random() > 0.7:
                    self.state = "FLOW"
                    self.break_start = now
                    print(">> DIRECTOR: LET IT FLOW (No cuts) <<")
                else:
                    self.last_switch = now
        
        elif self.state == "FLOW":
            if now - self.break_start > FLOW_DURATION:
                self.state = "CHAOS"
                self.last_switch = now
                print(">> DIRECTOR: CUT. Back to Chaos. <<")
                
        return self.state

class AudioBrain:
    def __init__(self):
        self.chroma_hist = collections.deque(maxlen=10)
        
    def process(self, buffer):
        y = np.frombuffer(buffer, dtype=np.float32)
        
        # Onset
        onset = librosa.onset.onset_strength(y=y, sr=RATE, hop_length=HOP_LENGTH)
        onset_val = np.mean(onset) if len(onset) > 0 else 0
        
        # Bands
        S = librosa.feature.melspectrogram(y=y, sr=RATE, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        cur = S_db[:, -1]
        
        sub = (np.mean(cur[0:5]) + 80) / 80.0
        hats = (np.mean(cur[80:128]) + 80) / 80.0
        
        # Chroma
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
    
    # AUDIO SETUP
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
    print("\nVisualizer Live. Waiting for Audio (or Auto-Pilot)...")

    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            onset, sub, hats, similarity = brain.process(data)
            mode = director.get_state()
            
            # --- 1. PLAYBACK DIRECTION & SPEED ---
            # Default speed is 1 (Standard Playback)
            speed = 1
            
            if hats > 0.85:
                speed = -2 # REWIND on Hi-Hats
            elif sub > 0.8:
                speed = 2  # FAST FWD on Kick

            # Apply Speed
            master_clock += speed
            
            # --- 2. SWITCHING LOGIC (Audio vs Auto-Pilot) ---
            should_switch = False
            
            # A. AUTO-PILOT CHECK (End of Video)
            # If we reached the end (or beginning, if rewinding), switch naturally
            if master_clock >= deck.total_frames:
                should_switch = True
                master_clock = 0 # Reset time for new clip
            elif master_clock < 0:
                should_switch = True
                master_clock = deck.total_frames - 1
            
            # B. AUDIO TRIGGER CHECK (Beat / Key Change)
            # Only if in Chaos Mode
            if mode == "CHAOS":
                if onset > 2.0 or similarity < 0.6:
                    should_switch = True
                    # NOTE: On audio switch, we DO NOT reset master_clock.
                    # This keeps the "morphing" effect alive.
            
            # EXECUTE SWITCH
            if should_switch:
                deck.pick_next_card()

            # --- 3. RENDER ---
            frame = deck.get_frame(master_clock)
            
            # FX: RGB Haze
            haze = min(onset / 2.5, 1.0)
            if haze > 0.15:
                frame = FXEngine.rgb_haze(frame, haze)
            
            # FX: Invert on super hard hits
            if mode == "CHAOS" and onset > 2.5:
                frame = cv2.bitwise_not(frame)

            cv2.imshow('Strict Shuffle', frame)
            
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