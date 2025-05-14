#!/usr/bin/env python
"""
audio_infer.py:  Quick inference check for an AudioCNN1D checkpoint.

Example:
    python audio_infer.py --ckpt checkpoints/audio_cnn1d_aug.pth \
                          --x processed_data/X_aug.npy \
                          --y processed_data/y_aug.npy
"""

import argparse, numpy as np, torch
from pathlib import Path
from models.audio_cnn1d import AudioCNN1D
from config import Config

# ---------------------------------------------------------------------
def load_model(ckpt: Path, in_ch: int, in_len: int):
    model = AudioCNN1D(input_channels=in_ch, input_length=in_len)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model

# ---------------------------------------------------------------------
def run_inference(model: torch.nn.Module, X: np.ndarray):
    """
    X shape must be (N, C, L) where C = 15, L = 300 (or squeeze variants).
    """
    X_t = torch.from_numpy(X).float()
    if X_t.ndim == 4 and X_t.shape[1] == 1:        # (N,1,15,300) -> (N,15,300)
        X_t = X_t.squeeze(1)
    assert X_t.ndim == 3, "Expected (N, 15, 300) tensor"

    with torch.no_grad():
        logits = model(X_t)
        probs  = torch.softmax(logits, dim=1)
        preds  = probs.argmax(1)
        latent = model.extract_latent_vector(X_t)
    return logits, probs, preds, latent

# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--x",    required=True, help="Path to X.npy (audio)")
    parser.add_argument("--y",    required=True, help="Path to y.npy")
    args = parser.parse_args()

    cfg = Config()                             # to read default channels/length
    model = load_model(Path(args.ckpt),
                       in_ch=cfg.input_channels,
                       in_len=cfg.input_length)

    X_np = np.load(args.x, mmap_mode="r")      # (N,15,300) or (N,1,15,300)
    y    = torch.from_numpy(np.load(args.y)).long()

    logits, probs, preds, latent = run_inference(model, X_np)

    print("Logits :", logits.shape)
    print("Probs  :", probs.shape)
    print("Preds  :", preds.tolist()[:10])
    print("True   :", y.tolist()[:10])
    print("Latent :", latent.shape)

    acc = (preds == y).float().mean().item()
    print(f"Accuracy: {acc:.3f}")

if __name__ == "__main__":
    main()
