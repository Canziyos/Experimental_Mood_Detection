import argparse, torch
from config import Config
from dataset.img_loader import make_image_loaders
from img_trainer import train
from evaluation import final_report
from models.mobilenet_v2_embed import MobileNetV2Encap

def parse_args():
    p = argparse.ArgumentParser("Train MobileNetV2 for emotion recognition")
    p.add_argument("mode", choices=["img", "rgb"], help="dataset variant")
    return p.parse_args()

def set_seed(s):
    import random, numpy as np
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def main():
    args = parse_args()
    cfg  = Config(img_mode=args.mode)
    set_seed(cfg.seed)

    loaders = make_image_loaders(cfg)
    model   = MobileNetV2Encap(pretrained=True, freeze_backbone=False)
    print(f"[INFO] lr={cfg.lr}, epochs={cfg.num_epochs}")

    model, _ = train(model, loaders, cfg)
    final_report(model, loaders["test"], cfg)
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
