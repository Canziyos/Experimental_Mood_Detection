"""
Unified evaluation utilities for the MobileNetV2 (image) and audio branches.

• evaluate() _ quick pass over a loader, returns loss + accuracy metrics.
• final_report(): saves classification report + confusion‑matrix artefacts.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from config import Config


##########################################################################
def topk_acc(logits: torch.Tensor,
             targets: torch.Tensor,
             ks: Sequence[int] = (1,)) -> list[float]:
    """Compute top‑k accuracies for given ks."""
    max_k = max(ks)
    _, pred = logits.topk(max_k, 1, True, True)   # (B, max_k)
    pred = pred.t()                               # (max_k, B)
    correct = pred.eq(targets.view(1, -1).expand_as(pred))  # (max_k, B)
    res = []
    for k in ks:
        res.append(correct[:k].reshape(-1).float().sum().item() * 100.0 / targets.size(0))
    return res


########################################################################
def evaluate(model: torch.nn.Module,
             loader: torch.utils.data.DataLoader,
             criterion: torch.nn.Module,
             device: torch.device,
             ks: Sequence[int] = (1,)) -> dict[str, float]:
    """Return dict with avg_loss and top-k accuracies."""
    model.eval()
    total, loss_sum = 0, 0.0
    tops = np.zeros(len(ks))
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss_sum += criterion(logits, labels).item()
            total += labels.size(0)
            tops += np.array(topk_acc(logits, labels, ks))
    return {
        "loss": loss_sum / len(loader),
        **{f"top{k}": acc for k, acc in zip(ks, tops / len(loader))}
    }


# ############################################################################
def final_report(model: torch.nn.Module,
                 loader: torch.utils.data.DataLoader,
                 cfg: Config,
                 split_name: str = "test") -> None:
    """
    Print + save classification report and confusion matrix for the supplied loader.
    Artefacts saved to cfg.checkpoint_dir:
      • classification_report_<mode>_<split>.txt
      • cm_<mode>_<split>.png
      • cm_<mode>_<split>.csv
    """
    device = next(model.parameters()).device
    model.eval()

    preds, true = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds.extend(logits.argmax(1).cpu().numpy())
            true.extend(labels.numpy())

    # -- Classification report --
    report = classification_report(true, preds,
                                   target_names=cfg.class_names,
                                   digits=2,
                                   output_dict=True)
    overall_acc = report["accuracy"] * 100
    macro_f1 = report["macro avg"]["f1-score"] * 100

    mode = getattr(cfg, "img_mode", getattr(cfg, "aud_mode", "unknown"))
    ckpt_dir = cfg.checkpoint_dir
    ckpt_dir.mkdir(exist_ok=True)

    rpt_txt = ckpt_dir / f"classification_report_{mode}_{split_name}.txt"
    rpt_txt.write_text(classification_report(true, preds,
                                             target_names=cfg.class_names,
                                             digits=2))
    print("\nClassification report (saved to TXT):")
    print(rpt_txt.read_text())

    # Also dump JSON dict for programmatic use.
    (ckpt_dir / f"classification_report_{mode}_{split_name}.json").write_text(
        json.dumps(report, indent=2))

    # -- Confusion matrix -
    cm = confusion_matrix(true, preds)
    cm_png = ckpt_dir / f"cm_{mode}_{split_name}.png"
    cm_csv = ckpt_dir / f"cm_{mode}_{split_name}.csv"

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=cfg.class_names,
                yticklabels=cfg.class_names)
    plt.title(f"Confusion Matrix – {mode.upper()} ({split_name})\n"
              f"acc {overall_acc:.2f}%  |  macro‑F1 {macro_f1:.2f}%")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(cm_png)
    plt.close()

    # save raw counts as CSV.
    pd.DataFrame(cm,
                 index=cfg.class_names,
                 columns=cfg.class_names).to_csv(cm_csv)

    print(f"Overall accuracy : {overall_acc:.2f}%")
    print(f"Macro F1‑score   : {macro_f1:.2f}%")
    print(f"Confusion matrix saved → {cm_png}")
    print(f"Raw matrix CSV   → {cm_csv}")
    print(f"Report TXT       → {rpt_txt}")
