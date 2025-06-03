import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1D(nn.Module):
    def __init__(self, input_features, num_classes):
        super().__init__()

        self.conv1 = nn.Conv1d(input_features, 512, 5, padding=2)
        self.bn1   = nn.BatchNorm1d(512)

        self.conv2 = nn.Conv1d(512, 512, 5, padding=2)
        self.bn2   = nn.BatchNorm1d(512)

        self.conv3 = nn.Conv1d(512, 256, 3, padding=1)
        self.bn3   = nn.BatchNorm1d(256)

        self.conv4 = nn.Conv1d(256, 256, 3, padding=1)
        self.bn4   = nn.BatchNorm1d(256)

        self.conv5 = nn.Conv1d(256, 128, 3, padding=1)
        self.bn5   = nn.BatchNorm1d(128)

        self.se_fc1 = nn.Linear(128, 32)
        self.se_fc2 = nn.Linear(32, 128)

        self.fc1     = nn.Linear(128, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc2     = nn.Linear(512, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        res = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = x + res

        x = F.relu(self.bn3(self.conv3(x)))
        res = x
        x = F.relu(self.bn4(self.conv4(x)))
        x = x + res

        x = F.relu(self.bn5(self.conv5(x)))

        gap = x.mean(dim=2)
        se  = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(gap))))
        x   = x * se.unsqueeze(-1)

        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    def predict_proba(self, x):
        return F.softmax(self.forward(x), dim=1)

    def get_latent(self, x):
        with torch.no_grad():
            return F.relu(self.fc1(
                F.adaptive_avg_pool1d(
                    self._se_forward(x), 1).squeeze(-1)))

    def _se_forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        res = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = x + res

        x = F.relu(self.bn3(self.conv3(x)))
        res = x
        x = F.relu(self.bn4(self.conv4(x)))
        x = x + res

        x = F.relu(self.bn5(self.conv5(x)))
        gap = x.mean(dim=2)
        se  = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(gap))))
        return x * se.unsqueeze(-1)
