import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pathlib import Path
from mobilenet.audio.log_mel import SpecAugment

# === Constants ===
data_dir = Path("mobilenet/audio/processed_data")
train_folders = [
    data_dir / "logmel_train_pitch_batches",
    data_dir / "logmel_train_noise_batches",
    data_dir / "logmel_train_speed_batches",
    data_dir / "logmel_train_none_batches",
]
val_folder = data_dir / "logmel_val_none_batches"
test_folder = data_dir / "logmel_test_none_batches"

batch_size = 32
num_workers = 2

# Dataset for lazy loading per-file.
class FolderLogMelDataset(Dataset):
    """Streams features from clean + aug folders, loading each pair lazily."""
    def __init__(self, folder: Path):
        self.folder = folder
        self.orig_files = sorted(folder.glob("orig_*.npy"))
        self.aug_files = sorted(folder.glob("aug_*.npy"))
        self.label_orig = sorted(folder.glob("label_*.npy"))
        self.label_aug = sorted(folder.glob("label_aug_*.npy"))

    def __len__(self):
        return min(len(self.orig_files), len(self.label_orig)) + min(len(self.aug_files), len(self.label_aug))

    def __getitem__(self, idx):
        if idx < len(self.orig_files):
            x = np.load(self.orig_files[idx])
            y = np.load(self.label_orig[idx])
        else:
            j = idx - len(self.orig_files)
            x = np.load(self.aug_files[j])
            y = np.load(self.label_aug[j])

        x = (x - x.mean()) / (x.std() + 1e-6)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# in-memory loading from full arrays.
"""
class PrecomputedLogMelDataset(Dataset):
    '''Loads full X and y arrays into memory, optionally applies SpecAugment.'''
    def __init__(self, features: np.ndarray, labels: np.ndarray, apply_aug: bool = False):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
        self.augment = SpecAugment() if apply_aug else torch.nn.Identity()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].unsqueeze(0)  # (1, H, W)
        x = self.augment(x)
        return x, self.y[idx]
"""

# Loader builder (active: FolderLogMelDataset)
def make_audio_loaders():
    print("Using FolderLogMelDataset with hardcoded split loading...")

    train_ds = ConcatDataset([FolderLogMelDataset(f) for f in train_folders])
    val_ds = FolderLogMelDataset(val_folder)
    test_ds = FolderLogMelDataset(test_folder)

    def make_loader(ds, is_train):
        return DataLoader(ds, batch_size=batch_size, shuffle=is_train,
                          num_workers=num_workers, pin_memory=True)

    # return {
    #     "train": make_loader(train_dataset, train=True),
    #     "val":   make_loader(val_dataset, train=False),
    #     "test":  make_loader(test_dataset, train=False),
    # }
    return {
        "train": make_loader(train_ds, True),
        "val":   make_loader(val_ds, False),
        "test":  make_loader(test_ds, False),
    }

# Test entries
if __name__ == "__main__":
    loaders = make_audio_loaders()
    print("Dataloaders ready.")
    for split in loaders:
        print(f"{split}: {len(loaders[split].dataset)} samples")
