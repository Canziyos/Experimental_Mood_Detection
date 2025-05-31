import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, Resample
from utils import load_config

# 1. Config
cfg = load_config("config.yaml")

root_dir = cfg["data"]["aud_train_dir"]  # manually change to aud_val_dir or aud_test_dir as needed
out_root = cfg["npys_dir"]["features"]
sr_target = cfg["preprocessing"]["sample_rate"]

# 2. Transforms
mel_spec = MelSpectrogram(
    sample_rate=sr_target,
    n_fft=400,
    win_length=int(cfg["preprocessing"]["win_length"] * sr_target),
    hop_length=int(cfg["preprocessing"]["hop_length"] * sr_target),
    n_mels=cfg["training"]["input_features"]
)
to_db = AmplitudeToDB()

# 3. Augmentation helpers
def add_noise(waveform, noise_level=0.005):
    noise = torch.randn_like(waveform) * noise_level
    return waveform + noise

def pitch_shift_safe(waveform, sr, n_steps=2):
    try:
        return torchaudio.functional.pitch_shift(waveform, sr, n_steps)
    except Exception as e:
        print(f"    Pitch shift failed: {e}")
        return waveform

# 4. Determine split name
split_name = os.path.basename(os.path.normpath(root_dir))
split_root = os.path.join(out_root, split_name)
os.makedirs(split_root, exist_ok=True)  # make features/train, features/val etc.

# 5. Iterate dataset
for emotion in sorted(os.listdir(root_dir)):
    in_emotion_dir = os.path.join(root_dir, emotion)
    out_emotion_dir = os.path.join(split_root, emotion)
    os.makedirs(out_emotion_dir, exist_ok=True)  # make features/train/Angry, etc.

    print(f"\n--- Processing [{split_name}/{emotion}] ---")
    wav_files = sorted([f for f in os.listdir(in_emotion_dir) if f.lower().endswith(".wav")])

    for idx, fname in enumerate(wav_files):
        print(f"  [{idx}] Processing {fname}")
        wav_path = os.path.join(in_emotion_dir, fname)
        waveform, sr_orig = torchaudio.load(wav_path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr_orig != sr_target:
            waveform = Resample(sr_orig, sr_target)(waveform)

        versions = {
            "": waveform,
            "_noise": add_noise(waveform),
            "_pitch": pitch_shift_safe(waveform, sr_target),
        }

        base_prefix = f"{split_name}_{emotion}_{str(idx).zfill(3)}"
        for suffix, wf in versions.items():
            mel = mel_spec(wf)
            mel_db = to_db(mel)
            mel_np = mel_db.squeeze(0).numpy().astype(np.float32)

            out_path = os.path.join(out_emotion_dir, base_prefix + suffix + ".npy")

            if os.path.exists(out_path):
                print(f"    Skipping (already exists): {base_prefix + suffix}.npy")
                continue

            np.save(out_path, mel_np, allow_pickle=False)
            print(f"    Saved: {base_prefix + suffix}.npy")
