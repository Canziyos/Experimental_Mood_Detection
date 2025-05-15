from collections import defaultdict
import time, numpy as np, torch
from typing import Dict
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from config import Config


def train(model: nn.Module, loaders: Dict[str, torch.utils.data.DataLoader], cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ---- class-weighted loss ----
    y_all = np.load(cfg.y_img_path, mmap_mode="r")
    weights = compute_class_weight("balanced",
                                   classes=np.arange(len(cfg.class_names)),
                                   y=y_all)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

    best, hist = 0.0, defaultdict(list)
    ckpt = cfg.checkpoint_dir / f"mobilenet_{cfg.img_mode}.pth"
    ckpt.parent.mkdir(exist_ok=True)

    for epoch in range(cfg.num_epochs):
        model.train()
        tot_loss = tot_correct = tot_count = 0

        for imgs, labels in tqdm(loaders["train"], desc=f"Epoch {epoch+1}/{cfg.num_epochs}", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            tot_loss += loss.item()
            tot_correct += (logits.argmax(1) == labels).sum().item()
            tot_count += labels.size(0)

        tr_acc = 100 * tot_correct / tot_count
        tr_loss = tot_loss / len(loaders["train"])

        # -------- validation --------
        model.eval()
        v_loss = v_correct = v_count = 0
        with torch.no_grad():
            for imgs, labels in loaders["val"]:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(imgs)
                v_loss += criterion(logits, labels).item()
                v_correct += (logits.argmax(1) == labels).sum().item()
                v_count += labels.size(0)
        v_acc = 100 * v_correct / v_count
        v_loss /= len(loaders["val"])
        scheduler.step()

        print(f"Ep {epoch+1:02d} | train {tr_loss:.3f}/{tr_acc:.1f}% | val {v_loss:.3f}/{v_acc:.1f}%")

        hist["train_acc"].append(tr_acc)
        hist["val_acc"].append(v_acc)
        if v_acc > best:
            best = v_acc
            torch.save(model.state_dict(), ckpt)
            print(f"↑ best model saved ({best:.2f}%)")

    model.load_state_dict(torch.load(ckpt))
    return model, hist
