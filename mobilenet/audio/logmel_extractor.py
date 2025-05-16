import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Lazy modules are a new feature.*")

"""Log-mel feature extractor with single-mode data expansion.
Creates X_logmel.npy and y_logmel_kept.npy by including original and one type of augmented version.
Supported modes: "none", "noise", "pitch", "speed"
"""

import numpy as np
import torch
import torchaudio
import random
from pathlib import Path
from mobilenet.audio.log_mel import LogMelSpec
from mobilenet.config import Config

# --- Config and augmentation mode ---
cfg = Config()
AUG_MODE = "\npitch"  # consequently "none", "noise", "pitch", "speed"

paths_file   = cfg.data_dir / "X_logmel_paths.npy"
labels_file  = cfg.data_dir / "y_logmel.npy"
output_file  = cfg.data_dir / f"X_logmel_{AUG_MODE}.npy"

paths  = np.load(paths_file, allow_pickle=True)
labels = np.load(labels_file)

logmel   = LogMelSpec(img_size=(cfg.logmel_h, cfg.logmel_w))
expected = (1, cfg.logmel_h, cfg.logmel_w)

features, kept_labels, bad_files = [], [], []
total = len(paths)
max_samples = 160000  # Clamp to 10 seconds @ 16kHz

print(f"[INFO] Extracting log-mel features from {total} audio files with mode: {AUG_MODE}...")

for i, (path, lbl) in enumerate(zip(paths, labels), 1):
    wav, sr = torchaudio.load(path)

    # Convert stereo to mono
    if wav.shape[0] == 2:
        wav = wav.mean(dim=0, keepdim=True)

    # --- Original ---
    spec = logmel(wav.squeeze(0))
    if spec.shape == expected:
        features.append(spec.numpy())
        kept_labels.append(lbl)
    else:
        bad_files.append((path, spec.shape))

    # --- Augmented ---
    aug_wav = wav.clone()

    if AUG_MODE == "noise":
        noise_level = random.uniform(0.003, 0.01)
        aug_wav = aug_wav + noise_level * torch.randn_like(aug_wav)

    elif AUG_MODE == "pitch":
        aug_wav = aug_wav[..., :min(aug_wav.shape[-1], max_samples)]
        n_steps = random.uniform(-2.0, 2.0)
        shift = torchaudio.transforms.PitchShift(sample_rate=sr, n_steps=n_steps)
        aug_wav = shift(aug_wav)

    elif AUG_MODE == "speed":
        speed_factor = random.uniform(0.95, 1.05)
        new_sr = int(sr * speed_factor)
        aug_wav = torchaudio.functional.resample(aug_wav, orig_freq=sr, new_freq=new_sr)

    else:
        continue  # Skip augmentation if mode is "none"

    # Clamp after transform (for pitch/speed)
    if aug_wav.shape[-1] > max_samples:
        aug_wav = aug_wav[..., :max_samples]

    spec = logmel(aug_wav.squeeze(0))
    if spec.shape == expected:
        features.append(spec.numpy())
        kept_labels.append(lbl)
    else:
        bad_files.append((path, spec.shape))

    # Progress indicator
    print(f"processed {i}/{total}")

# --- Report skipped files ---
if bad_files:
    print(f"[WARN] Skipped {len(bad_files)} file(s) with unexpected shapes.")
    for p, shp in bad_files[:10]:
        print(f"    {p} -> {shp}")
else:
    print("[INFO] All files produced the expected shape.")

# --- Save outputs ---
X = np.stack(features, axis=0)
np.save(output_file, X)
np.save(cfg.data_dir / "y_logmel_kept.npy", np.array(kept_labels))

print(f"[INFO] Saved {X.shape[0]} samples to {output_file.name} (shape {X.shape}).")
print("[INFO] Corresponding labels saved to y_logmel_kept.npy.")
