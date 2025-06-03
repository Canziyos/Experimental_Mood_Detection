import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class FeatureDataset(Dataset):
    """
    Dataset for loading .npy audio features and integer labels.
    Supports optional on-the-fly augmentation via transform or built-in logic.
    """

    def __init__(self, label_dict, file_list, transform=None, augment=False):
        self.label_dict = label_dict
        self.file_list = file_list
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fpath = self.file_list[idx]
        features = np.load(fpath)
        base = os.path.splitext(os.path.basename(fpath))[0]
        label = self.label_dict[fpath] if fpath in self.label_dict else self.label_dict[base]
        features = torch.tensor(features, dtype=torch.float32)

        if self.transform:
            features = self.transform(features)

        if self.augment:
            is_augmented = any(s in fpath for s in ["_noise", "_pitch", "_stretch"])
            features = self.apply_augment(features, light=is_augmented)

        return features, torch.tensor(label, dtype=torch.long)

    def apply_augment(self, x, light=False):
        x = x.clone()  # break graph connection once

        if light:
            if random.random() < 0.2:
                x = x + torch.randn_like(x) * 0.02
            if random.random() < 0.1:
                t = random.randint(0, x.shape[1] - 10)
                mask = x.clone()
                mask[:, t:t+5] = 0
                x = mask
        else:
            if random.random() < 0.4:
                x = x + torch.randn_like(x) * 0.05
            if random.random() < 0.3:
                t = random.randint(0, x.shape[1] - 20)
                mask = x.clone()
                mask[:, t:t+10] = 0
                x = mask

        return x
