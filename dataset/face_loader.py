import numpy as np
import torch
from torch.utils.data import Dataset

class ImageEmotionDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = np.load(x_path)
        self.y = np.load(y_path)

        assert self.X.shape[0] == self.y.shape[0], "Mismatch between images and labels!"

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y
