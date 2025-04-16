import torch
import torch.nn as nn
import torch.nn.functional as F

class MoodMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MoodMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # raw logits, softmax later during loss.
