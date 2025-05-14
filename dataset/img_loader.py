import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple
from config import Config

# -- Transform pipelines --
_tx_train = T.Compose([
    T.ToPILImage(),                        # expects H×W or H×W×C.
    T.RandomHorizontalFlip(0.5),           # mirror.
    T.RandomRotation(10, fill=0),          # ±10 degrees.
    T.RandomResizedCrop(224, scale=(0.90, 1.05), ratio=(0.9, 1.1)),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),                          # --> 0‑1 float, C×H×W.
    T.RandomErasing(p=0.25, scale=(0.02, 0.06), ratio=(0.3, 3.3)),
    T.Normalize(mean=[0.5], std=[0.5]),
])

_tx_eval = T.Compose([
    T.ToPILImage(),
    T.Resize(224),                         # shortest side.
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])

# -- Dataset class --
class ImageEmotionDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, transform):
        self.X, self.y, self.tf = X, y, transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = self.X[idx]                  # uint8 (1,224,224) or (224,224).
        if img.ndim == 3 and img.shape[0] == 1:
            img = img.squeeze(0)           # → (H,W) for ToPILImage.
        img = self.tf(img)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return img, label

# -- Disk I/O helpers --
def load_img_data(cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.load(cfg.x_img_path, mmap_mode="r"),
        np.load(cfg.y_img_path, mmap_mode="r"),
    )

# -- Public factory --
def make_image_loaders(cfg: Config) -> Dict[str, DataLoader]:
    X, y = load_img_data(cfg)
    print(f"Loaded {X.shape} - mode={cfg.img_mode}")

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tv, y_tv, test_size=cfg.val_size, stratify=y_tv, random_state=cfg.seed
    )

    def _dl(xs, ys, train_flag):
        tf = _tx_train if train_flag else _tx_eval
        ds = ImageEmotionDataset(xs, ys, tf)
        return DataLoader(ds,
                          batch_size=cfg.batch_size,
                          shuffle=train_flag,
                          num_workers=2,
                          pin_memory=True)

    return {
        "train": _dl(X_tr,  y_tr,  True),
        "val"  : _dl(X_val, y_val, False),
        "test" : _dl(X_test,y_test,False),
    }
