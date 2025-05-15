"""Log-mel front-end and SpecAugment."""
import torch, torchaudio, torch.nn as nn

class LogMelSpec(nn.Module):
    def __init__(self, sr=16_000, n_fft=512, hop=160, n_mels=64, img_size=(96,192)):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop,
            n_mels=n_mels, f_min=50, f_max=8000, power=1.0
        )
        self.resize = nn.Upsample(size=img_size, mode="bilinear", align_corners=False)

    @torch.inference_mode()
    def forward(self, wav: torch.Tensor):  # wav (B,T) in float32.
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        spec = torch.log1p(self.mel(wav))          # (B, n_mels, frames).
        spec = self.resize(spec.unsqueeze(1))      # (B,1,H,W).
        return spec

class SpecAugment(nn.Module):
    def __init__(self, time_mask=20, freq_mask=8, num_masks=2):
        super().__init__()
        self.t_mask, self.f_mask, self.n = time_mask, freq_mask, num_masks
    def forward(self, x):  # (B,1,F,T)
        if not self.training:
            return x
        B, _, F, T = x.shape
        for _ in range(self.n):
            t0 = torch.randint(0, max(1, T-self.t_mask), (1,))
            f0 = torch.randint(0, max(1, F-self.f_mask), (1,))
            x[:,:, f0:f0+self.f_mask, :] = 0
            x[:,:, :, t0:t0+self.t_mask] = 0
        return x

# Factory helper.
log_mel_spectrogram = LogMelSpec()