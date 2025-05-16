from mobilenet.config import Config
from mobilenet.image.img_loader import make_image_loaders
from mobilenet.image.img_trainer import train
from mobilenet.image.evaluation import final_report
from mobilenet.image.image_model import ImageMobileNetV2
import torch


def set_seed(seed):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():

    cfg = Config()
    set_seed(cfg.seed)

    loaders = make_image_loaders(cfg)
    model = ImageMobileNetV2(pretrained=True, freeze_backbone=False)
    print(f"[INFO] lr={cfg.lr}, epochs={cfg.num_epochs}, workers={cfg.num_workers}")

    model, _ = train(model, loaders, cfg)
    final_report(model, loaders["test"], cfg)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
