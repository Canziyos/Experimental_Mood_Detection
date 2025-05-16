"""Log-mel feature extractor with single-mode data expansion.
Saves each feature+label pair as soon as it's ready. Zero RAM issues. CUDA-optimized.
Supported modes: "none", "noise", "pitch", "speed"
"""

import numpy as np
import torch
import torchaudio
import random
from pathlib import Path
from mobilenet.audio.log_mel import LogMelSpec
from mobilenet.config import Config
import librosa
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning, message="Lazy modules are a new feature.*")

cfg = Config()
AUG_MODE = "pitch"  # choices: "none", "noise", "pitch", "speed"

paths_file   = cfg.data_dir / "X_logmel_paths.npy"
labels_file  = cfg.data_dir / "y_logmel.npy"
save_dir     = cfg.data_dir / f"logmel_{AUG_MODE}_batches"
save_dir.mkdir(exist_ok=True)

paths  = np.load(paths_file, allow_pickle=True)
labels = np.load(labels_file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logmel = LogMelSpec(img_size=(cfg.logmel_h, cfg.logmel_w)).to(device)
expected = (1, cfg.logmel_h, cfg.logmel_w)

total = len(paths)
max_samples = 240000  # ~15 seconds at 16kHz

print(f"[INFO] Extracting log-mel features from {total} audio files with mode: {AUG_MODE}...")

for i, (path, lbl) in enumerate(zip(paths, labels), 1):
    wav, sr = torchaudio.load(path)
    wav = wav.to(device)

    # Convert stereo to mono
    if wav.shape[0] == 2:
        wav = wav.mean(dim=0, keepdim=True)
    if wav.shape[-1] > max_samples:
        wav = wav[..., :max_samples]

    # --- Original ---
    spec = logmel(wav.squeeze(0))
    if spec.shape == expected:
        np.save(save_dir / f"orig_{i:05d}.npy", spec.cpu().numpy())
        np.save(save_dir / f"label_{i:05d}.npy", np.array(lbl))
    else:
        print(f"[WARN] Skipped original {path} with shape {spec.shape}")

    # --- Augmented ---
    aug_wav = wav.clone()
    if AUG_MODE == "noise":
        noise_level = random.uniform(0.003, 0.01)
        aug_wav = aug_wav + noise_level * torch.randn_like(aug_wav)
    elif AUG_MODE == "pitch":
        aug_np = aug_wav.squeeze(0).cpu().numpy()
        n_steps = random.uniform(-2.0, 2.0)
        shifted_np = librosa.effects.pitch_shift(aug_np, sr=sr, n_steps=n_steps)
        aug_wav = torch.tensor(shifted_np, dtype=torch.float32).unsqueeze(0).to(device)
    elif AUG_MODE == "speed":
        aug_np = aug_wav.squeeze(0).cpu().numpy()
        speed_factor = random.uniform(0.95, 1.05)
        stretched_np = librosa.effects.time_stretch(aug_np, rate=speed_factor)
        aug_wav = torch.tensor(stretched_np, dtype=torch.float32).unsqueeze(0).to(device)
    else:
        pass

    if aug_wav.shape[-1] > max_samples:
        aug_wav = aug_wav[..., :max_samples]

    spec_aug = logmel(aug_wav.squeeze(0))
    if spec_aug.shape == expected:
        np.save(save_dir / f"aug_{i:05d}.npy", spec_aug.cpu().numpy())
        np.save(save_dir / f"label_aug_{i:05d}.npy", np.array(lbl))
    else:
        print(f"[WARN] Skipped aug {path} with shape {spec_aug.shape}")

    print(f"processed {i}/{total}")

print(f"[INFO] All features written per-sample to {save_dir}.")
print(f"[INFO] Use a loader script later to assemble final arrays for training.")

