import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from pathlib import Path
from config import Config


class ImageDataset(Dataset):
    """
    Lazily opens X_img.npy / y_img.npy inside *each* worker.
    Nothing huge is pickled during DataLoader-spawn.
    """

    def __init__(self, x_path: Path, y_path: Path):
        self.x_path = x_path
        self.y_path = y_path

        # Only to get length once (tiny memmap, closes immediately).
        self._len = np.load(self.y_path, mmap_mode="r").shape[0]

        # Worker-local fields (set to None in parent; created in worker).
        self.X = None
        self.y = None

    # ------------------------------------------------------------- #
    def _lazy_init(self):
        """Re-open memmaps inside the current worker."""
        if self.X is None:
            self.X = np.load(self.x_path, mmap_mode="r")  # uint8.
            self.y = np.load(self.y_path, mmap_mode="r")  # int64.

    # ------------------------------------------------------------- #
    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        self._lazy_init()  # Executes only once per worker.
        img = torch.tensor(self.X[idx], dtype=torch.float32).div_(255.0)  # (3,224,224) in [0,1].
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return img, label


# ================================================================= #
def make_image_loaders(cfg: Config):
    print(f"Loading image data from: {cfg.x_img_path.name}, mode: {cfg.img_mode}")

    full_ds = ImageDataset(cfg.x_img_path, cfg.y_img_path)
    idxs = np.arange(len(full_ds))
    y_all = np.load(cfg.y_img_path, mmap_mode="r")

    train_idx, test_idx, _, _ = train_test_split(
        idxs, y_all, test_size=cfg.test_size, random_state=cfg.seed, stratify=y_all
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=cfg.val_size, random_state=cfg.seed, stratify=y_all[train_idx]
    )

    def _loader(index_set, shuffle):
        subset = Subset(full_ds, index_set)
        return DataLoader(
            subset,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=True,                # for GPU speed-up.
            persistent_workers=cfg.num_workers > 0,
        )

    return {
        "train": _loader(train_idx, shuffle=True),
        "val": _loader(val_idx, shuffle=False),
        "test": _loader(test_idx, shuffle=False),
    }
