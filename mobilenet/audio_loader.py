
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from pathlib import Path
from config import Config
from audio_features import log_mel_spectrogram, SpecAugment

class LogMelDataset(Dataset):
    """Streams .wav paths, converts to logmel on the fly (memory safe)."""
    def __init__(self, wav_list: list[Path], labels: np.ndarray, cfg: Config, train: bool):
        self.wavs = wav_list
        self.labels = labels.astype(np.int64)
        self.spec = log_mel_spectrogram
        self.augment = SpecAugment() if train and cfg.aud_mode == "augmented" else nn.Identity()

    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        wav, _ = torchaudio.load(self.wavs[idx])   # (1,T)
        spec = self.spec(wav.squeeze(0))           # (1,H,W)
        spec = self.augment(spec)
        return spec, torch.tensor(self.labels[idx])

# wav_paths.npy & y_aud.npy

def make_audio_loaders(cfg: Config):
    wav_paths = np.load(cfg.wav_list_path)         # array of strings
    labels    = np.load(cfg.y_aud_path)

    idx = np.arange(len(labels))
    tr, te, _, _ = train_test_split(idx, labels, test_size=cfg.test_size,
                                    random_state=cfg.seed, stratify=labels)
    tr, va = train_test_split(tr, test_size=cfg.val_size, random_state=cfg.seed,
                              stratify=labels[tr])

    def _dl(split_idx, train):
        ds = LogMelDataset([Path(p) for p in wav_paths[split_idx]], labels[split_idx], cfg, train)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=train,
                          num_workers=cfg.num_workers, pin_memory=True)
    return {"train": _dl(tr, True), "val": _dl(va, False), "test": _dl(te, False)}
