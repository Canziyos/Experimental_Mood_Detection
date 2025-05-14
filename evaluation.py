# evaluation.py
"""
Utility functions for loss/accuracy evaluation and a final
classification‑report + confusion‑matrix dump.
"""

import torch
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from config import Config


# ────────────────────────────────────────────────────────────────
def evaluate(model, loader, criterion, device):
    """Return (avg_loss, accuracy %) on the supplied loader."""
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

    return loss_total / len(loader), 100.0 * correct / total


# ────────────────────────────────────────────────────────────────
def final_report(model, loader, cfg: Config, split_name="test"):
    """
    Produces a scikit‑learn classification report and confusion matrix,
    saves them in cfg.checkpoint_dir, and prints the report to console.
    """
    device = next(model.parameters()).device
    preds, true = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds.extend(outputs.argmax(1).cpu().numpy())
            true.extend(labels.cpu().numpy())

    # ─── classification report ──────────────────────────────────
    report_str = classification_report(true, preds,
                                       target_names=cfg.class_names,
                                       digits=2)
    print("\nClassification report:\n", report_str)

    mode = getattr(cfg, "img_mode",
                   getattr(cfg, "aud_mode", "unknown"))

    rpt_path = cfg.checkpoint_dir / f"classification_report_{mode}_{split_name}.txt"
    rpt_path.write_text(report_str, encoding="utf-8")
    print(f"Report saved -> {rpt_path}")

    # ─── confusion matrix plot ──────────────────────────────────
    cm = confusion_matrix(true, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=cfg.class_names,
                yticklabels=cfg.class_names)
    plt.title(f"Confusion Matrix - {mode.upper()} ({split_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    fig_path = cfg.checkpoint_dir / f"cm_{mode}_{split_name}.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Confusion matrix saved → {fig_path}")
