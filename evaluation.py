import torch
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from config import Config


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_total, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_total += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return loss_total / len(loader), 100 * correct / total


def final_report(model, loader, cfg: Config):
    device = next(model.parameters()).device
    preds, true = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds.extend(outputs.argmax(1).cpu().numpy())
            true.extend(labels.cpu().numpy())

    print("\nClassification report:\n", classification_report(true, preds, digits=2))
    rpt_path = cfg.checkpoint_dir / f"classification_report_{cfg.mode}.txt"
    with open(rpt_path, "w") as f:
        f.write(classification_report(true, preds, digits=2))
    print(f"Rapport sparad → {rpt_path}")

    cm = confusion_matrix(true, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cfg.class_names, yticklabels=cfg.class_names)
    plt.title(f"Confusion Matrix – {cfg.mode.upper()} set")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    fig_path = cfg.checkpoint_dir / f"cm_{cfg.mode}.png"
    plt.savefig(fig_path)
    print(f"Konfusionsmatris sparad → {fig_path}")