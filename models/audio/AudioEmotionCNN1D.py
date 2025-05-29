import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioEmotionCNN1D(nn.Module):
    """
    1D CNN audio model (5 conv + 2 linear layers) audio emotion recognition,
    adapted from Srihari et al., CVMI 2024.
    """

    def __init__(self, input_features=39, num_classes=8):
        super().__init__()

        # Layer 1: Conv1d, output 512 channels (paper specification)
        self.conv1 = nn.Conv1d(input_features, 512, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(512)

        # Layer 2: Conv1d, 512 channels
        self.conv2 = nn.Conv1d(512, 512, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(512)

        # Layer 3: Conv1d, 256 channels
        self.conv3 = nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        # Layer 4: Conv1d, 256 channels
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(256)

        # Layer 5: Conv1d, 128 channels
        self.conv5 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(128)

        # Paper fixes the flattened output to 23936; 
        # This assumes a fixed input frame count as in their setup.
        self.fc1 = nn.Linear(23936, 512)  # First linear layer after flatten
        self.fc2 = nn.Linear(512, num_classes)  # Final layer to output classes

        # Softmax for inference, not needed for loss computation
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Input x: (batch_size, feature_dim, time_steps), e.g., (B, 39, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        # Output: (batch_size, 128, time_steps)

        # Flatten before FC layers.
        # assumes time_steps are such that 128 * T = 23936 for the fixed paper setup
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # logits for CrossEntropyLoss

    def predict_proba(self, x):
        # Return softmax probabilities for inference.
        logits = self.forward(x)
        return self.softmax(logits)
