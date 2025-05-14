import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioCNN1D(nn.Module):
    """1D CNN for speech-emotion recognition.

    5-block convs backbone but replaced the length-dependent flatten with adaptive global pooling.
    That way the model works for any sequence length at inference time and uses far fewer parameters.
    """

    def __init__(self, input_channels: int = 15, num_classes: int = 6):
        super().__init__()

        # --- Convolutional trunk ---
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(input_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        # --- Global average pooling removes time‑axis dependence ---
        self.pool = nn.AdaptiveAvgPool1d(1)  # (B, 128, 1) → (B, 128)

        # --- Fully connected head -
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    # --------------------------
    # Forward / utility methods.
    # --------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        x = self.pool(x).squeeze(-1)  # (B, 128).
        return self.fc(x)

    def extract_latent_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Return the penultimate 512-D vector (handy for fusion or t-SNE)."""
        x = self.conv_blocks(x)
        x = self.pool(x).squeeze(-1)          # (B, 128)
        x = self.fc[0](x)                     # dropout
        x = self.fc[1](x)                     # linear 128->512
        x = self.fc[2](x)                     # ReLU
        return x  # shape (B, 512)

# ---------------------------------------------------------------------
# Any other length threw a shape error.
# Global pooling collapses the time axis to one mean value per
# channel, regardless of how many frames came in, so the FC sees a constant
# 128‑D input.

if __name__ == "__main__":
    dummy = torch.randn(4, 15, 480)  # batch of variable‑length signals
    model = AudioCNN1D()
    out = model(dummy)
    print(out.shape)  # -> (4, 6)
