import os
import glob
import yaml
import numpy as np
import yaml


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def compute_mel_mean_std(features_root, num_mels=39):
    """
    Compute per-channel (per Mel bin) mean and std for .npy feature files.
    """
    sums = np.zeros(num_mels, dtype=np.float64)
    sq_sums = np.zeros(num_mels, dtype=np.float64)
    total_frames = 0

    npy_files = [
        path for path in glob.glob(os.path.join(features_root, "*", "*.npy"))
        if not any(suffix in os.path.basename(path) for suffix in ["_pitch", "_noise"])
    ]

    print(f"Found {len(npy_files)} clean .npy files for stat computation.")

    for path in npy_files:
        arr = np.load(path)  # shape: (num_mels, T)
        total_frames += arr.shape[1]
        sums += arr.sum(axis=1)
        sq_sums += (arr ** 2).sum(axis=1)

    if total_frames == 0:
        raise ValueError("No frames found across files. Cannot compute statistics.")

    mean = (sums / total_frames).astype(np.float32)
    std = np.sqrt(sq_sums / total_frames - mean**2).astype(np.float32)

    return mean, std

if __name__ == "__main__":
    config_path = "config.yaml"
    print("Using config at:", os.path.abspath(config_path))

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config.setdefault("preprocessing", {})

    train_dir = os.path.join(config["npys_dir"]["features"], "train")
    mean, std = compute_mel_mean_std(train_dir, num_mels=39)

    config["preprocessing"]["global_mean"] = [round(float(m), 4) for m in mean.tolist()]
    config["preprocessing"]["global_std"] = [round(float(s), 4) for s in std.tolist()]

    print("Computed global_mean and global_std for Mel features:")
    print("mean (first 5):", config["preprocessing"]["global_mean"][:5])
    print("std  (first 5):", config["preprocessing"]["global_std"][:5])

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)

    print("config.yaml updated.")
