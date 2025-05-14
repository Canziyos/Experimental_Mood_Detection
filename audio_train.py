import argparse
import torch
from config import Config
from data_loader import DataLoader
from trainer import train
from evaluation import final_report
from models.audio_model import AudioCNN1D


def parse_args():
    p = argparse.ArgumentParser(description="Train 1‑D CNN for emotion recognition")
    p.add_argument("mode", choices=["augmented", "clean"], help="dataset variant to use")
    return p.parse_args()


def set_seed(seed: int):
    import random, numpy as np
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    cfg = Config(mode=args.mode)
    set_seed(cfg.seed)

    loaders = DataLoader(cfg)
    model = AudioCNN1D()

    model, _ = train(model, loaders, cfg)
    final_report(model, loaders["test"], cfg)


if __name__ == "__main__":
    main()