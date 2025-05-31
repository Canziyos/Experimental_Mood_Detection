import os
import numpy as np
import torch
from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    """
    Dataset for loading .npy audio features and integer labels.
    Assumes full paths in file_list and uses label_dict for lookup.
    """

    def __init__(self, label_dict, file_list, transform=None):
        self.label_dict = label_dict
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fpath = self.file_list[idx]
        features = np.load(fpath)

        base = os.path.splitext(os.path.basename(fpath))[0]
        label = self.label_dict[fpath] if fpath in self.label_dict else self.label_dict[base]

        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            features = self.transform(features)

        return features, label
