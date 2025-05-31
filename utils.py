import os
import glob
import yaml
import numpy as np

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def compute_mel_mean_std(features_root, num_mels=39):
    """
    Compute per-channel (per Mel bin) mean and std for .npy feature files.

    Args:
        features_root (str): Path to root of extracted features.
        num_mels (int): Number of Mel bins (default 39).

    Returns:
        mean: np.array of shape (num_mels,)
        std:  np.array of shape (num_mels,)
    """
    sums = np.zeros(num_mels, dtype=np.float64)
    sq_sums = np.zeros(num_mels, dtype=np.float64)
    total_frames = 0

    npy_files = glob.glob(os.path.join(features_root, "*", "*.npy"))

    for path in npy_files:
        arr = np.load(path)  # shape: (num_mels, T)
        total_frames += arr.shape[1]
        sums += arr.sum(axis=1)
        sq_sums += (arr ** 2).sum(axis=1)

    mean = sums / total_frames
    std = np.sqrt(sq_sums / total_frames - mean**2)

    return mean.astype(np.float32), std.astype(np.float32)


if __name__ == "__main__":
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    features_dir = config["npys_dir"]["features"]
    mean, std = compute_mel_mean_std(features_dir, num_mels=39)

    config.setdefault("preprocessing", {})
    config["preprocessing"]["global_mean"] = [round(float(m), 4) for m in mean]
    config["preprocessing"]["global_std"] = [round(float(s), 4) for s in std]

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=None)

    print("Updated config.yaml with global_mean and global_std.")
