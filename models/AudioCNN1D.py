import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

cfg = Config()

class AudioCNN1D(nn.Module):
    def __init__(self, cfg.input_channels, cfg.input_length):
        super(AudioCNN1D, self).__init__()

        # === From Srihari. (audio model) ===
        # First conv block: input is (15, 300), i.e., 15 feature rows over 300 time steps.
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(512)

        self.conv2 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(512)

        self.conv3 = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)

        self.conv5 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(128)

        # Flatten: 128 output channels × 300 frames = 38400 features.
        self.flatten_dim = 128 * input_length

        self.fc1 = nn.Linear(self.flatten_dim, 512)

        # === Modified ===
        # Paper claimed 8 classes, but we removed "Surprise", then 6 classes.
        self.fc2 = nn.Linear(512, 6)

        # === Added ===
        # Dropout to reduce overfitting (paper doesn’t mention it).
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # High-level vector from convs + fc1, then classify.
        x = self.extract_latent_vector(x)
        x = self.fc2(x)
        return x

    def extract_latent_vector(self, x):
        # === From paper === (5 conv layers with BatchNorm + ReLU).
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.bn5(F.relu(self.conv5(x)))

        # Flatten the time dimension
        x = x.view(x.size(0), -1)

        # === Added dropout after FC1 (not in original paper) ===
        x = self.dropout(F.relu(self.fc1(x)))

        return x  # this is the vector we can use for fusion later.
