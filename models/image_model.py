import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageCNN2D(nn.Module):
    def __init__(self):
        super(ImageCNN2D, self).__init__()

        # First convolutional layer -input is grayscale image (B, 1, 48, 48).
        # Paper specified kernel_size=3, output_channels=32, and pooling after this.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)

        # RGB for later test.
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # (B, 32, 56, 56)

        # Second conv layer -output_channels=64, same kernel size, followed by pooling again.
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # (B, 64, 28, 28).

        # Flattening: 64*12*12 = 9216 matches paper's reported ~48640 neurons??!(unbelievably weird).
        self.flatten_dim = 64 * 12 * 12 

        # FC layers -paper used 128 --> 64 --> 8, we go 128 -> 64 -> 6 (no Surprise class).
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)  # 6 emotion classes.

    def forward(self, x):
        x = self.extract_features(x)  # Get latent feature vector (B, 64).
        x = self.fc3(x)               # Final classification layer (B, 6).
        return x

    def extract_features(self, x):
        # Convolution + ReLU + pooling as defined in the paper.
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Flatten conv output into a single vector per sample.
        x = x.view(x.size(0), -1)

        # Feedforward fully connected stack to compress and prepare for fusion(our intrepretation).
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x  # Final latent feature shape: (B, 64).
