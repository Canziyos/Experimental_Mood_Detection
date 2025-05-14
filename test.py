import torch
import numpy as np
import argparse
from pathlib import Path

from models.audio_model_flat import AudioCNN1D  # flatten-versionen


def load_model(ckpt_path: Path) -> torch.nn.Module:
    model = AudioCNN1D(input_channels=15, input_length=300)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    return model


def run_inference(model, X: torch.Tensor):
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(1)
        if hasattr(model, "extract_latent_vector"):
            latent = model.extract_latent_vector(X)
        else:
            latent = None
    return logits, probs, preds, latent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to model .pth")
    parser.add_argument("--x", required=True, help="Path to X.npy")
    parser.add_argument("--y", required=True, help="Path to y.npy")
    args = parser.parse_args()

    # Load model and data
    model = load_model(Path(args.ckpt))
    X = torch.from_numpy(np.load(args.x)).float()  # (N, 15, 300)
    y = torch.from_numpy(np.load(args.y)).long()

    if X.ndim == 3:
        pass
    elif X.ndim == 4:
        X = X.squeeze(1)  # In case shape is (N, 1, 15, 300)
    else:
        raise ValueError("X shape must be (N, 15, 300)")

    # Run inference
    logits, probs, preds, latent = run_inference(model, X)

    # Print
    print("Logits shape :", logits.shape)
    print("Probs shape  :", probs.shape)
    print("Predictions  :", preds.tolist()[:10])
    print("True labels  :", y.tolist()[:10])
    if latent is not None:
        print("Latent shape :", latent.shape)

    acc = (preds == y).float().mean().item()
    print(f"Accuracy     : {acc:.3f}")


if __name__ == "__main__":
    main()
