#!/usr/bin/env python
"""
img_infer.py: Quick inference.

Usage:
    python img_infer.py --ckpt checkpoints/mobilenet_img.pth \
                        --x processed_data/X_img.npy \
                        --y processed_data/y_img.npy
"""

import argparse, numpy as np, torch
from pathlib import Path
import torchvision.transforms as T
from models.mobilenet_v2_embed import MobileNetV2Encap
from config import Config

# -------------------------------------------------#
def load_model(ckpt_path: Path, n_classes: int = 6):
    model = MobileNetV2Encap(pretrained=False, freeze_backbone=False,
                             n_classes=n_classes)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    return model

# same eval transform used in _tx_eval
_eval_tx = T.Compose([
    T.ToPILImage(),
    T.Resize(224),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])

# -----------------------------------------#
def run_inference(model, X_np: np.ndarray):
    """
    X_np shape: (N, 1, 224, 224)  uint8  [0-255].
    Returns logits, probs, preds, latent (128-D features).
    """
    imgs = []
    for img in X_np:
        img = img.squeeze(0)          # (224,224).
        imgs.append(_eval_tx(img))
    X = torch.stack(imgs)             # (N,1,224,224) tensor float32.

    with torch.no_grad():
        logits = model(X)
        probs  = torch.softmax(logits, dim=1)
        preds  = probs.argmax(1)
        latent = model.extract_features(X)  # (N, 128).
    return logits, probs, preds, latent

# -----------------------------------------------------------------------------#
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--x",    required=True, help="Path to X_img.npy")
    parser.add_argument("--y",    required=True, help="Path to y_img.npy")
    args = parser.parse_args()

    model = load_model(Path(args.ckpt))
    X_np  = np.load(args.x, mmap_mode="r")   # (N,1,224,224) uint8.
    y     = torch.from_numpy(np.load(args.y)).long()

    logits, probs, preds, latent = run_inference(model, X_np)

    print("Logits  :", logits.shape)
    print("Probs   :", probs.shape)
    print("Preds   :", preds.tolist()[:10])
    print("True    :", y.tolist()[:10])
    print("Latent  :", latent.shape)

    acc = (preds == y).float().mean().item()
    print(f"Accuracy: {acc:.3f}")

if __name__ == "__main__":
    main()
