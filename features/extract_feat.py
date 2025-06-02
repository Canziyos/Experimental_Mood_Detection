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

root_dir = cfg["data"]["aud_val_dir"]  # change to val or test if needed (only train is augmented)
out_root = cfg["npys_dir"]["features"]
sr_target = cfg["preprocessing"]["sample_rate"]
n_mels = cfg["training"]["input_features"]
target_length = 400  # fixed length.

# normalization.
normalize = "global_mean" in cfg["preprocessing"]
if normalize:
    global_mean = np.array(cfg["preprocessing"]["global_mean"], dtype=np.float32)
    global_std = np.array(cfg["preprocessing"]["global_std"], dtype=np.float32)

# 2. Transforms.
mel_spec = MelSpectrogram(
    sample_rate=sr_target,
    n_fft=400,
    win_length=int(cfg["preprocessing"]["win_length"] * sr_target),
    hop_length=int(cfg["preprocessing"]["hop_length"] * sr_target),
    n_mels=n_mels
)
to_db = AmplitudeToDB()

# 3. Utility functions.
def fix_length(mel_np, target_length):
    cur_len = mel_np.shape[1]
    if cur_len > target_length:
        return mel_np[:, :target_length]
    elif cur_len < target_length:
        return np.pad(mel_np, ((0, 0), (0, target_length - cur_len)), mode='constant')
    return mel_np

def add_noise(waveform, noise_level=0.005):
    return waveform + torch.randn_like(waveform) * noise_level

def pitch_shift_safe(waveform, sr, n_steps=2):
    try:
        return torchaudio.functional.pitch_shift(waveform, sr, n_steps)
    except Exception as e:
        print(f"    Pitch shift failed: {e}")
        return waveform

# 4. Determine split name.
split_name = os.path.basename(os.path.normpath(root_dir))
split_root = os.path.join(out_root, split_name)
os.makedirs(split_root, exist_ok=True)

# 5. Process dataset.
for emotion in sorted(os.listdir(root_dir)):
    in_emotion_dir = os.path.join(root_dir, emotion)
    out_emotion_dir = os.path.join(split_root, emotion)
    os.makedirs(out_emotion_dir, exist_ok=True)

    print(f"\nProcessing {split_name}/{emotion}")
    wav_files = sorted([f for f in os.listdir(in_emotion_dir) if f.lower().endswith(".wav")])

    for idx, fname in enumerate(wav_files):
        print(f"  [{idx}] {fname}")
        wav_path = os.path.join(in_emotion_dir, fname)
        waveform, sr_orig = torchaudio.load(wav_path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr_orig != sr_target:
            waveform = Resample(sr_orig, sr_target)(waveform)

        # Always include original.
        versions = {"": waveform}
        if split_name.lower() == "train":
            versions["_noise"] = add_noise(waveform)
            versions["_pitch"] = pitch_shift_safe(waveform, sr_target)

        base_prefix = f"{split_name}_{emotion}_{str(idx).zfill(3)}"

        for suffix, wf in versions.items():
            mel = mel_spec(wf)
            mel_db = to_db(mel)
            mel_np = mel_db.squeeze(0).numpy().astype(np.float32)
            mel_np = fix_length(mel_np, target_length)

            if normalize:
                mel_np = (mel_np - global_mean[:, None]) / global_std[:, None]

            out_path = os.path.join(out_emotion_dir, base_prefix + suffix + ".npy")
            if os.path.exists(out_path):
                print(f" Skipping: {base_prefix + suffix}.npy")
                continue

            np.save(out_path, mel_np, allow_pickle=False)
            print(f" Saved: {base_prefix + suffix}.npy")
