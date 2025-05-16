"""Offline extractor for log-mel audio. Saves X_logmel_paths.npy, y_logmel.npy, and labelmap_logmel.json."""
import json, torchaudio, numpy as np
from pathlib import Path
from ..config import Config

cfg = Config()

# === Set paths ===
src = cfg.audio_dir
outfile = cfg.data_dir / "X_logmel"
y_outfile = cfg.data_dir / "y_logmel.npy"
labelmap_outfile = cfg.data_dir / "labelmap_logmel.json"

print(f"[INFO] Scanning audio folders at: {src.resolve()}")

# === Build label map from folder names.
label_map = {d.name: i for i, d in enumerate(sorted(src.iterdir())) if d.is_dir()}
print(f"[INFO] Detected classes: {label_map}")

paths, labels = [], []

# === Walk subfolders and collect .wav paths.
for cls, idx in label_map.items():
    class_path = src / cls
    wav_files = list(class_path.rglob("*.wav"))
    print(f"[INFO] Found {len(wav_files)} WAVs in '{cls}'.")
    for wav in wav_files:
        paths.append(str(wav))
        labels.append(idx)

# === Save outputs to .npy and .json files.
paths = np.array(paths)
labels = np.array(labels)

np.save(f"{outfile}_paths.npy", paths)
np.save(y_outfile, labels)

with open(labelmap_outfile, "w") as f:
    json.dump(label_map, f, indent=2)

# === Final status output.
print(f"[INFO] Done. Saved:")
print(f" - {len(labels)} samples.")
print(f" - Paths     - {outfile.name}s.npy")
print(f" - Labels    - {y_outfile.name}")
print(f" - Label map - {labelmap_outfile.name}")
