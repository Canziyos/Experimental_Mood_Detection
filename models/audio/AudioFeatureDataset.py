import os
import numpy as np
import torch
from torch.utils.data import Dataset

class AudioFeatureDataset(Dataset):
    """
    PyTorch Dataset for loading pre-extracted audio features (.npy files) and labels.

    Args:
        features_dir (str): Path to the directory containing .npy feature files.
        label_dict (dict): Mapping from filename (without extension) to integer label.
        file_list (list): List of feature filenames to use (without path).
        transform (callable, optional): Optional transform to apply to features.
    """
    def __init__(self, features_dir, label_dict, file_list, transform=None):
        self.features_dir = features_dir
        self.label_dict = label_dict
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        fpath = os.path.join(self.features_dir, fname)
        features = np.load(fpath)  # shape should match the model input

        # Remove .npy extension to match the label_dict key
        base = os.path.splitext(fname)[0]
        label = self.label_dict[base]

        # Convert to torch tensor.
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform is not None:
            features = self.transform(features)

        return features, label
