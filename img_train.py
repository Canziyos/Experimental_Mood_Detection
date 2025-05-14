import argparse
import torch
from config import Config
from face_loader import ImageEmotionDataset
from img_trainer import train
from evaluation import final_report
from models.ImageCNN2D import ImageCNN2D
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description="Train 2D CNN on facial expression data")
    p.add_argument("mode", choices=["clean", "augmented"], help="dataset variant to use")
    return p.parse_args()

def set_seed(seed: int):
    import random
    import numpy as np
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_dataloaders(cfg: Config):
    # Load raw npy files and split them
    X = np.load(cfg.x_img_path)
    y = np.load(cfg.y_img_path)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=cfg.test_size + cfg.val_size, stratify=y, random_state=cfg.seed)
    val_ratio = cfg.val_size / (cfg.test_size + cfg.val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1 - val_ratio, stratify=y_temp, random_state=cfg.seed)

    train_ds = ImageEmotionDataset(X_train, y_train)
    val_ds = ImageEmotionDataset(X_val, y_val)
    test_ds = ImageEmotionDataset(X_test, y_test)

    loaders = {
        "train": DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True),
        "val":   DataLoader(val_ds, batch_size=cfg.batch_size),
        "test":  DataLoader(test_ds, batch_size=cfg.batch_size),
    }

    return loaders

def main():
    args = parse_args()
    cfg = Config(mode=args.mode)
    set_seed(cfg.seed)

    loaders = build_dataloaders(cfg)
    model = ImageCNN2D()

    model, _ = train(model, loaders, cfg)
    final_report(model, loaders["test"], cfg)

if __name__ == "__main__":
    main()
