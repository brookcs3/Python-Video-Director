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

# AUDIO
RATE = 44100
CHUNK = 2048
HOP_LENGTH = 512

# PACING (The "Director")
FLOW_INTERVAL = 12.0  # Seconds of Chaos before we force a break
FLOW_DURATION = 4.0   # How long the "Long Take" lasts

class VideoDeck:
    """Manages clips to ensure we visit unused clips before repeating"""
    def __init__(self):
        self.captures = []
        self.paths = []
        self.current_idx = 0
        self.total_frames = 0
        
        # Track usage
        self.unused_deck = [] 
        self.history = collections.deque(maxlen=5) # Prevent immediate back-to-back repeats
        
        print("\n--- LOADING CINEMATIC DECK ---")
        
        # 1. Generate Paths (Base + 2..24)
        self.paths.append(f"{BASE_PATH}{EXTENSION}") 
        for i in range(2, TOTAL_CLIPS + 1):
            self.paths.append(f"{BASE_PATH}-{i}{EXTENSION}")
            
        # 2. Open Captures
        for i, path in enumerate(self.paths):
            if not os.path.exists(path):
                print(f"Warning: File not found: {path}")
                continue
            cap = cv2.VideoCapture(path)
            if self.total_frames == 0 and cap.isOpened():
                self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.captures.append(cap)
        
        if not self.captures: raise Exception("No videos loaded!")
        
        # Initialize the deck
        self.shuffle_deck()
        print(f"Deck Loaded. {len(self.captures)} Clips. Fairness Algorithm Active.")

    def shuffle_deck(self):
        """Refills the deck with all clip indices and shuffles them"""
        print(">> RE-SHUFFLING DECK (All clips used, starting fresh) <<")
        self.unused_deck = list(range(len(self.captures)))
        random.shuffle(self.unused_deck)

    def get_next_clip(self):
        """Smart selector: Pops from unused deck to ensure variety"""
        if not self.unused_deck:
            self.shuffle_deck()
            
        # Pop the next unique clip
        next_idx = self.unused_deck.pop()
        
        # Edge Case: If we just reshuffled, try not to play the *exact* same clip 
        # that ended the last cycle (if possible)
        if next_idx == self.current_idx and len(self.unused_deck) > 0:
            self.unused_deck.insert(0, next_idx) # Put it back
            next_idx = self.unused_deck.pop()    # Take a different one
            
        self.current_idx = next_idx
        return self.current_idx

    def get_frame(self, master_frame_idx):
        cap = self.captures[self.current_idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, master_frame_idx)
        ret, frame = cap.read()
        if not ret: return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return frame

class Director:
    """Controls the Pacing (Chaos vs Flow)"""
    def __init__(self):
        self.state = "CHAOS" # or "FLOW"
        self.last_switch_time = time.time()
        self.flow_start_time = 0
        
    def update(self):
        now = time.time()
        
        if self.state == "CHAOS":
            # Have we been chaotic long enough?
            if (now - self.last_switch_time) > FLOW_INTERVAL:
                # 20% Chance to trigger Flow Mode (don't make it too predictable)
                if random.random() > 0.2:
                    self.state = "FLOW"
                    self.flow_start_time = now
                    print(">> DIRECTOR: HOLD THAT SHOT (Flow Mode) <<")
                else:
                    # Reset timer but stay in chaos (skip this cycle)
                    self.last_switch_time = now
                    
        elif self.state == "FLOW":
            # Is the break over?
            if (now - self.flow_start_time) > FLOW_DURATION:
                self.state = "CHAOS"
                self.last_switch_time = now
                print(">> DIRECTOR: CUT. Back to action. <<")
                
        return self.state

class AudioBrain:
    def __init__(self):
        self.chroma_hist = collections.deque(maxlen=10)
        
    def listen(self, buffer):
        y = np.frombuffer(buffer, dtype=np.float32)
        
        # Features
        onset = librosa.onset.onset_strength(y=y, sr=RATE, hop_length=HOP_LENGTH)
        onset_val = np.mean(onset) if len(onset) > 0 else 0
        
        S = librosa.feature.melspectrogram(y=y, sr=RATE, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        cur = S_db[:, -1]
        
        sub = (np.mean(cur[0:5]) + 80) / 80.0
        
        chroma = librosa.feature.chroma_stft(y=y, sr=RATE, hop_length=HOP_LENGTH)
        curr_chroma = np.mean(chroma, axis=1)
        
        sim = 0.6
        if len(self.chroma_hist) > 0:
            avg = np.mean(self.chroma_hist, axis=0)
            sim = np.dot(curr_chroma, avg) / (np.linalg.norm(curr_chroma)*np.linalg.norm(avg))
        
        self.chroma_hist.append(curr_chroma)
        return onset_val, sub, sim

def main():
    p = pyaudio.PyAudio()
    
    # DEVICE SELECTION
    print("\n--- AUDIO DEVICES ---")
    blackhole_idx = None
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if "BlackHole" in dev['name']: blackhole_idx = i
    
    idx = blackhole_idx if blackhole_idx is not None else int(input("Index: "))

    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True,
                    input_device_index=idx, frames_per_buffer=CHUNK)

    # INIT
    deck = VideoDeck()
    brain = AudioBrain()
    director = Director()
    
    master_clock = 0
    print("\nVisualizer Live. Waiting for audio...")

    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            onset, sub, similarity = brain.listen(data)
            
            # Check Director State
            current_mode = director.update()
            
            # --- DIRECTOR LOGIC ---
            
            if current_mode == "CHAOS":
                # ALLOW CUTS
                
                # 1. Hard Beat -> Next Unused Clip
                if onset > 1.8:
                    deck.get_next_clip()
                    
                # 2. Key Change -> Next Unused Clip
                elif similarity < 0.6:
                    deck.get_next_clip()
                    
                # 3. Sub Bass -> Fast Forward
                speed = 1.3 if sub > 0.8 else 1
                
            else:
                # FLOW MODE (NO CUTS ALLOWED)
                # We ignore onset/similarity triggers for switching clips.
                # We only allow speed changes.
                speed = 1 
            
            # --- RENDER ---
            master_clock = (master_clock + speed) % deck.total_frames
            if master_clock < 0: master_clock += deck.total_frames
            
            frame = deck.get_frame(master_clock)
            
            # Apply FX (Only aggressive FX in Chaos mode)
            if current_mode == "CHAOS":
                if onset > 2.0: frame = cv2.bitwise_not(frame)
                if sub > 0.85: # Shake
                    frame = np.roll(frame, random.randint(-10,10), axis=0)
            else:
                # In Flow mode, maybe just a subtle grain or smooth pulse
                pass 

            cv2.imshow('Cinematic Sampler', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)
            break

    stream.stop_stream()
    p.terminate()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()