import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, Resample
from utils import load_config

cfg = load_config("config.yaml")
normalize = True
sr_target = cfg["preprocessing"]["sample_rate"]
n_mels = cfg["training"]["input_features"]
target_length = 400

normalize = "global_mean" in cfg["preprocessing"]
if normalize:
    global_mean = np.array(cfg["preprocessing"]["global_mean"], dtype=np.float32)
    global_std = np.array(cfg["preprocessing"]["global_std"], dtype=np.float32)

mel_spec = MelSpectrogram(
    sample_rate=sr_target,
    n_fft=400,
    win_length=int(cfg["preprocessing"]["win_length"] * sr_target),
    hop_length=int(cfg["preprocessing"]["hop_length"] * sr_target),
    n_mels=n_mels
)
to_db = AmplitudeToDB()

def fix_length(mel_np, target_length):
    cur_len = mel_np.shape[1]
    if cur_len > target_length:
        return mel_np[:, :target_length]
    elif cur_len < target_length:
        return np.pad(mel_np, ((0, 0), (0, target_length - cur_len)), mode='constant')
    return mel_np

def add_noise(wf, noise_level=0.005):
    return wf + torch.randn_like(wf) * noise_level

def pitch_shift_safe(wf, sr, n_steps=2):
    try:
        return torchaudio.functional.pitch_shift(wf, sr, n_steps)
    except Exception as e:
        print("  Pitch shift failed:", e)
        return wf

for split in ["train", "val"]:
    root_dir = cfg["data"][f"{split}_dir"]
    out_root = os.path.join(cfg["npys_dir"]["root"], split)

    print("Extracting", split, "data from:", root_dir)
    for label in sorted(os.listdir(root_dir)):
        in_class = os.path.join(root_dir, label)
        out_class = os.path.join(out_root, label)
        os.makedirs(out_class, exist_ok=True)

        for idx, fname in enumerate(sorted(os.listdir(in_class))):
            if not fname.lower().endswith(".wav"):
                continue
            path = os.path.join(in_class, fname)
            wf, sr = torchaudio.load(path)

            if wf.shape[0] > 1:
                wf = wf.mean(dim=0, keepdim=True)
            if sr != sr_target:
                wf = Resample(sr, sr_target)(wf)

            versions = {"": wf}
            if split == "train":
                versions["_noise"] = add_noise(wf)
                versions["_pitch"] = pitch_shift_safe(wf, sr_target)

            for suffix, waveform in versions.items():
                mel = mel_spec(waveform)
                mel_db = to_db(mel)
                mel_np = mel_db.squeeze(0).numpy().astype(np.float32)
                mel_np = fix_length(mel_np, target_length)

                if normalize:
                    mel_np = (mel_np - global_mean[:, None]) / global_std[:, None]

                out_path = os.path.join(out_class, f"{split}_{label}_{str(idx).zfill(3)}{suffix}.npy")
                np.save(out_path, mel_np, allow_pickle=False)
                print("Saved:", out_path)
