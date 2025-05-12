from collections import defaultdict
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from pathlib import Path
from typing import Dict
from evaluation import evaluate
from config import Config


def train(model: nn.Module, loaders: Dict[str, torch.utils.data.DataLoader], cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

    metrics = defaultdict(list)
    best_val_acc = 0.0
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cfg.checkpoint_dir / f"audio_cnn1d_{cfg.mode}.pth"

    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0
        start = time.time()

        for inputs, labels in tqdm(loaders["train"], desc=f"Epoch {epoch+1}/{cfg.num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        train_loss = epoch_loss / len(loaders["train"])

        val_loss, val_acc = evaluate(model, loaders["val"], criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
            f"{time.time() - start:.1f}s"
        )

        metrics["train_acc"].append(train_acc)
        metrics["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"Model improved → saved to {ckpt_path}")

    # load best model for downstream test/eval
    model.load_state_dict(torch.load(ckpt_path))
    return model, metrics