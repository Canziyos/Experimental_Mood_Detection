"""CLI to train the log‑mel MobileNetV2 audio model."""
import argparse, torch
from config import Config
from models.mobilenet_v2_audio import MobileNetV2Audio
from dataset.audio_loader import make_audio_loaders
from train.aud_trainer import train

p = argparse.ArgumentParser(); p.add_argument("--mode", choices=["clean","augmented"], default="augmented")
args = p.parse_args()

cfg = Config(aud_mode=args.mode)
loaders = make_audio_loaders(cfg)
model = MobileNetV2Audio()
print(f"[INFO] lr={cfg.lr}, epochs={cfg.num_epochs}")
train(model, loaders, cfg)
