import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioCNN1D(nn.Module):
    """
    1-D CNN for sequential audio features (e.g. 15*300 SED rows).
    """

    def __init__(self, input_channels: int = 15, input_length: int = 300, n_classes: int = 6):
        super().__init__()

        self.conv1 = nn.Conv1d(input_channels, 512, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(512)

        self.conv2 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(512)

        self.conv3 = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(256)

        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm1d(256)

        self.conv5 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.bn5   = nn.BatchNorm1d(128)

        self.flatten_dim = 128 * input_length      # 128 × 300

        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, n_classes)

    # --------------------------------------------------------------#
    def extract_latent_vector(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.bn5(F.relu(self.conv5(x)))

        x = x.view(x.size(0), -1)          # flatten
        x = self.dropout(F.relu(self.fc1(x)))
        return x                           # (B, 512)

    # -------------------------------------------------#
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_latent_vector(x)
        return self.fc2(x)               # (B, n_classes).
