#!/usr/bin/env python3
"""
Generate synthetic logits, probs, and latent vectors to test the 
FusionAV module in isolation (before getting real outputs from teammates).
"""

import numpy as np
import torch
from pathlib import Path
import argparse

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def simulate_fusion_inputs(batch_size=32, num_classes=6,
                           latent_audio_dim=512, latent_image_dim=128,
                           seed=42, out_dir="simulated"):
    np.random.seed(seed)
    torch.manual_seed(seed)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Simulate logits.(pre-sm).
    logits_audio = np.random.randn(batch_size, num_classes).astype(np.float32)
    logits_image = np.random.randn(batch_size, num_classes).astype(np.float32)

    # Apply softmax to get probs.
    probs_audio = softmax(logits_audio)
    probs_image = softmax(logits_image)

    # Latent vectors.
    latent_audio = np.random.randn(batch_size, latent_audio_dim).astype(np.float32)
    latent_image = np.random.randn(batch_size, latent_image_dim).astype(np.float32)

    # Simulate class labels (with some imbalance).
    y_true = np.random.choice(num_classes, size=batch_size, p=[0.1, 0.1, 0.2, 0.3, 0.2, 0.1])

    # Save.
    np.save(out_path / "probs_audio.npy", probs_audio)
    np.save(out_path / "probs_image.npy", probs_image)
    np.save(out_path / "logits_audio.npy", logits_audio)
    np.save(out_path / "logits_image.npy", logits_image)
    np.save(out_path / "latent_audio.npy", latent_audio)
    np.save(out_path / "latent_image.npy", latent_image)
    np.save(out_path / "y_true.npy", y_true)

    print(f"Saved fusion inputs to: {out_path.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--latent_audio_dim", type=int, default=512)
    parser.add_argument("--latent_image_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="simulated")
    args = parser.parse_args()

    simulate_fusion_inputs(**vars(args))
