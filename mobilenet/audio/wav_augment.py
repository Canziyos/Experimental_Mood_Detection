import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import torch
import torchaudio
import random
import librosa
from mobilenet.audio.log_mel import LogMelSpec
import warnings

# warnings.filterwarnings("ignore", category=UserWarning, message="Lazy modules are a new feature.*")

# === HARDCODED SETTING ===
split = "val"         # "train", "val", or "test"
aug_mode = "none"     # "none", "noise", "pitch", or "speed"

# === Dynamic path construction based on split
paths_file  = f"mobilenet/audio/processed_data/X_{split}_paths.npy"
labels_file = f"mobilenet/audio/processed_data/y_{split}.npy"
save_dir    = f"mobilenet/audio/processed_data/logmel_{split}_{aug_mode}_batches"

os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logmel = LogMelSpec(img_size=(96, 192)).to(device)
expected = (1, 96, 192)
max_samples = 240000

paths = np.load(paths_file, allow_pickle=True)
labels = np.load(labels_file)

print(f"Extracting features for {len(paths)} files with split='{split}' and mode='{aug_mode}'")

for i, (path, lbl) in enumerate(zip(paths, labels), 1):
    wav, sr = torchaudio.load(path)
    wav = wav.to(device)

    if wav.shape[0] == 2:
        wav = wav.mean(dim=0, keepdim=True)
    if wav.shape[-1] > max_samples:
        wav = wav[..., :max_samples]

    # === Save original
    spec = logmel(wav.squeeze(0))
    if spec.shape == expected:
        np.save(f"{save_dir}/orig_{i:05d}.npy", spec.cpu().numpy())
        np.save(f"{save_dir}/label_{i:05d}.npy", np.array(lbl))
    else:
        print(f"Skipped original {path} (bad shape {spec.shape})")

    # === Augmentation (only if aug_mode ≠ "none")
    if aug_mode != "none":
        aug_wav = wav.clone()

        if aug_mode == "noise":
            aug_wav = aug_wav + random.uniform(0.003, 0.01) * torch.randn_like(aug_wav)

        elif aug_mode == "pitch":
            aug_np = aug_wav.squeeze().cpu().numpy()
            n_steps = random.uniform(-2.0, 2.0)
            aug_np = librosa.effects.pitch_shift(aug_np, sr=sr, n_steps=n_steps)
            aug_wav = torch.tensor(aug_np, dtype=torch.float32).unsqueeze(0).to(device)

        elif aug_mode == "speed":
            aug_np = aug_wav.squeeze().cpu().numpy()
            speed_factor = random.uniform(0.95, 1.05)
            aug_np = librosa.effects.time_stretch(aug_np, rate=speed_factor)
            aug_wav = torch.tensor(aug_np, dtype=torch.float32).unsqueeze(0).to(device)

        if aug_wav.shape[-1] > max_samples:
            aug_wav = aug_wav[..., :max_samples]

        spec_aug = logmel(aug_wav.squeeze(0))
        if spec_aug.shape == expected:
            np.save(f"{save_dir}/aug_{aug_mode}_{i:05d}.npy", spec_aug.cpu().numpy())
            np.save(f"{save_dir}/label_aug_{aug_mode}_{i:05d}.npy", np.array(lbl))
        else:
            print(f"Skipped AUG {path} (bad shape {spec_aug.shape})")

    print(f"Processed {i}/{len(paths)}")

print(f"All features written to {save_dir}")
