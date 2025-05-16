"""Log-mel front-end and SpecAugment.

every spectrogram returns with shape (1, H, W) and dtype float32.
"""
from pathlib import Path
import torch
import torchaudio
import torch.nn as nn

class LogMelSpec(nn.Module):
    def __init__(
        self,
        sr:        int = 16_000,
        n_fft:     int = 512,
        hop:       int = 160,
        n_mels:    int = 64,
        img_size:  tuple[int, int] = (96, 192),
    ):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop,
            n_mels=n_mels,
            f_min=50,
            f_max=8_000,
            power=1.0,
        )
        self.resize = nn.Upsample(size=img_size, mode="bilinear", align_corners=False)

    @torch.inference_mode()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        wav : (T) or (1, T) float32 tensor.
        Returns
        -------
        spec : (1, H, W) float32 tensor.
        """
        # Ensure shape (1, T) and dtype float32.
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav = wav.float()

        # Mel → log(1 + mel) → resize → (1, H, W).
        spec = self.mel(wav)                # (1, n_mels, frames)
        spec = torch.log1p(spec)
        spec = self.resize(spec.unsqueeze(1))  # add channel dim → (1, 1, H, W)
        return spec.squeeze(0)                 # final (1, H, W)

class SpecAugment(nn.Module):
    """Simple time- and frequency-masking implementation."""
    def __init__(self, time_mask: int = 20, freq_mask: int = 8, num_masks: int = 2):
        super().__init__()
        self.t_mask = time_mask
        self.f_mask = freq_mask
        self.n      = num_masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 1, F, T)
        if not self.training:
            return x
        B, _, F, T = x.shape
        for _ in range(self.n):
            t0 = torch.randint(0, max(1, T - self.t_mask), ())
            f0 = torch.randint(0, max(1, F - self.f_mask), ())
            x[:, :, f0 : f0 + self.f_mask, :] = 0
            x[:, :, :, t0 : t0 + self.t_mask] = 0
        return x

# Default instance for easy import.
log_mel_spectrogram = LogMelSpec()
