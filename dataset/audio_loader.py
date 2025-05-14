import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple
from config import Config

class EmotionDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_np_data(cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    return np.load(cfg.x_aud_path, mmap_mode="r"), np.load(cfg.y_aud_path, mmap_mode="r")

def make_loaders(cfg: Config):
    X, y = load_np_data(cfg)
    print(f"Loaded audio data from: {cfg.x_aud_path.name}, shape: {X.shape}, mode: {cfg.aud_mode}")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=cfg.val_size,
        random_state=cfg.seed,
        stratify=y_trainval,
    )

    def _loader(X_split, y_split, shuffle):
        ds = EmotionDataset(X_split, y_split)
        return ds, DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=2)

    return {
        "train": _loader(X_train, y_train, True),
        "val": _loader(X_val, y_val, False),
        "test": _loader(X_test, y_test, False),
    }
