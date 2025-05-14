# img_train.py – Train the MobileNetV2 image branch
import argparse, torch
from config import Config
from dataset.img_loader import make_image_loaders
from img_trainer import train
from evaluation import final_report
from models.ImageCNN2D import ImageCNN2D as MobileNetV2Encap


# ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser("Train MobileNetV2 on facial‑expression data")
    p.add_argument("mode", choices=["img", "rgb"],
                   help="dataset variant (grayscale vs RGB)")
    return p.parse_args()


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    cfg  = Config(img_mode=args.mode)
    set_seed(cfg.seed)

    # --- data ---------------------------------------------------------------
    loaders = make_image_loaders(cfg)           # train / val / test DataLoaders

    # --- model --------------------------------------------------------------
    model = MobileNetV2Encap(pretrained=True, freeze_backbone=False)
    print(f"[INFO] Starting training – lr={cfg.lr}, mode={cfg.img_mode}, "
          f"epochs={cfg.num_epochs}")

    # --- training -----------------------------------------------------------
    model, _ = train(model, loaders, cfg)

    # --- final evaluation ---------------------------------------------------
    final_report(model, loaders["test"], cfg)

    print("[INFO] Run complete.")


if __name__ == "__main__":
    main()
