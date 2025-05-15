#!/usr/bin/env python3
"""
Generate synthetic logits (pre-softmax), probs, and latent vectors to test the 
FusionAV module in isolation (before getting real outputs from teammates).
"""

import numpy as np
import torch
from pathlib import Path
import argparse

#!/usr/bin/env python3
"""
generate synthetic logits, probs, and latent vectors
to test the FusionAV module in isolation (before getting real outputs from teammates).
"""

import numpy as np
import torch
from pathlib import Path
import argparse

# === Fake softmax for numpy arrays ===
# This guy takes a 2D array of raw scores (logits) and pretends to be smart by
# normalizing them into probabilities (along the last axis by default).
# Prevents overflow by subtracting max before exp, then divides by the total.
def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))  # classic trick to not explode values.
    return e / e.sum(axis=axis, keepdims=True)           # normalize to 1.

# === Main fusion input simulator ===
def simulate_fusion_inputs(batch_size=32, num_classes=6,
                           latent_audio_dim=512, latent_image_dim=128,
                           seed=42, out_dir="simulated"):
    
    # Just to make sure the randomness doesn't go on a joyride.
    np.random.seed(seed)
    torch.manual_seed(seed)

    
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # --- Simulate classification logits ---
    # These are the raw brain farts before the model starts its own predictions.
    logits_audio = np.random.randn(batch_size, num_classes).astype(np.float32)
    logits_image = np.random.randn(batch_size, num_classes).astype(np.float32)

    # --- Simulate softmax probs ---
    probs_audio = softmax(logits_audio)
    probs_image = softmax(logits_image)

    # --- latent vectors ---
    # Deep internal thoughts of each model.
    latent_audio = np.random.randn(batch_size, latent_audio_dim).astype(np.float32)
    latent_image = np.random.randn(batch_size, latent_image_dim).astype(np.float32)

    # --- Ground truth---
    # Simulated class labels; skewed on purpose to make things interesting.
    y_true = np.random.choice(num_classes, size=batch_size,
                              p=[0.1, 0.1, 0.2, 0.3, 0.2, 0.1])

    # --- Save the simulation results ---

    np.save(out_path / "probs_audio.npy", probs_audio)
    np.save(out_path / "probs_image.npy", probs_image)
    np.save(out_path / "logits_audio.npy", logits_audio)
    np.save(out_path / "logits_image.npy", logits_image)
    np.save(out_path / "latent_audio.npy", latent_audio)
    np.save(out_path / "latent_image.npy", latent_image)
    np.save(out_path / "y_true.npy", y_true)

    print(f"Saved fusion inputs to: {out_path.resolve()}")

# --- CLI handler ---
# Control batch size, dims, etc. via command-line.
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




# python simulate_fusion_inputs.py --batch_size 64 --out_dir my_folder/
