import os
import json
import numpy as np
from pathlib import Path

# === Hardcoded paths
audio_root = Path("dataset/audio")                              # .wav folders
output_dir = Path("mobilenet/audio/processed_data")             # where .npy + .json go
output_dir.mkdir(parents=True, exist_ok=True)

splits = ["train", "val", "test"]
labelmap_outfile = output_dir / "labelmap_logmel.json"

# === Build label map from training folder names
train_dir = audio_root / "train"
label_map = {cls.name: idx for idx, cls in enumerate(sorted(train_dir.iterdir())) if cls.is_dir()}

# Save label map once
with open(labelmap_outfile, "w") as f:
    json.dump(label_map, f, indent=2)
print(f"Label map saved → {labelmap_outfile}")

# === Process each split
for split in splits:
    split_dir = audio_root / split
    x_out = output_dir / f"X_{split}_paths.npy"
    y_out = output_dir / f"y_{split}.npy"

    paths = []
    labels = []

    for class_name, label_idx in label_map.items():
        class_path = split_dir / class_name
        wavs = list(class_path.glob("*.wav"))
        print(f"[{split}] Found {len(wavs)} in '{class_name}'")
        for wav in wavs:
            paths.append(str(wav))
            labels.append(label_idx)

    np.save(x_out, np.array(paths))
    np.save(y_out, np.array(labels))
    print(f"Saved {len(paths)} samples for {split}")
    print(f" - X: {x_out.name}, shape: {len(paths)}")
    print(f" - y: {y_out.name}, shape: {len(labels)}\n")

print("All metadata saved to: mobilenet/audio/processed_data/")
