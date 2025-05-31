import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1D(nn.Module):
    def __init__(self, input_features=39, num_classes=6):  # ✅ 6 classes now
        super().__init__()

        # Conv Blocks
        self.conv1 = nn.Conv1d(input_features, 512, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(512)

        self.conv2 = nn.Conv1d(512, 512, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(512)

        self.conv3 = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)

        self.conv5 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(128)

        # Squeeze-and-Excite (SE) attention
        self.se_fc1 = nn.Linear(128, 32)
        self.se_fc2 = nn.Linear(32, 128)

        # Fully connected layers
        self.fc1 = nn.Linear(128, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_classes)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Input: (B, 39, T)
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 512, T)

        res1 = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = x + res1

        x = F.relu(self.bn3(self.conv3(x)))

        res2 = x
        x = F.relu(self.bn4(self.conv4(x)))
        x = x + res2

        x = F.relu(self.bn5(self.conv5(x)))  # (B, 128, T)

        # SE attention
        se = x.mean(dim=2)  # (B, 128)
        se = F.relu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        x = x * se.unsqueeze(-1)  # (B, 128, T)

        # Global pooling
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # (B, 128)

        x = F.relu(self.fc1(x))  # (B, 512)
        latent = x  # <=== captured embedding
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # logits

    def predict_proba(self, x):
        logits = self.forward(x)
        return self.softmax(logits)

    def get_latent(self, x):
        # Forward pass until the fc1 layer
        x = F.relu(self.bn1(self.conv1(x)))
        res1 = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = x + res1

        x = F.relu(self.bn3(self.conv3(x)))
        res2 = x
        x = F.relu(self.bn4(self.conv4(x)))
        x = x + res2

        x = F.relu(self.bn5(self.conv5(x)))
        se = x.mean(dim=2)
        se = F.relu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        x = x * se.unsqueeze(-1)

        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x = F.relu(self.fc1(x))  # (B, 512)
        return x  # 512-dim latent embedding
