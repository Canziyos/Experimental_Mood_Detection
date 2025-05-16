import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from config import Config


class PrecomputedLogMelDataset(Dataset):
    """Loads precomputed log-mel spectrograms and labels from .npy files."""
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_audio_loaders(cfg: Config):
    X = np.load(cfg.data_dir / "X_logmel.npy")
    y = np.load(cfg.data_dir / "y_logmel_kept.npy")

    idx = np.arange(len(y))
    tr, te, _, _ = train_test_split(idx, y, test_size=cfg.test_size,
                                    random_state=cfg.seed, stratify=y)
    tr, va = train_test_split(tr, test_size=cfg.val_size, random_state=cfg.seed,
                              stratify=y[tr])

    def _dl(split_idx, train):
        ds = PrecomputedLogMelDataset(X[split_idx], y[split_idx])
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=train,
                          num_workers=cfg.num_workers, pin_memory=True)

    return {"train": _dl(tr, True), "val": _dl(va, False), "test": _dl(te, False)}
