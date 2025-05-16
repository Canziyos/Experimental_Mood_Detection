import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from pathlib import Path

def final_report(model, loader, cfg, save_prefix=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            preds = out.argmax(dim=1).cpu().numpy()
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Accuracy.
    acc = 100 * np.mean(y_true == y_pred)
    print(f"\nFinal Accuracy: {acc:.2f}%")

    # Classification Report.
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=cfg.class_names))

    # Save report
    ckpt_dir = cfg.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_path = ckpt_dir / f"{save_prefix}classification_report.txt"
    with open(report_path, "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=cfg.class_names))
    print(f"Report saved --> {report_path}")

    # Optional: Save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=cfg.class_names, yticklabels=cfg.class_names, cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("True")
        cm_path = ckpt_dir / f"{save_prefix}confusion_matrix.png"
        plt.savefig(cm_path)
        print(f"Confusion matrix saved -> {cm_path}")
    except ImportError:
        print(" matplotlib or seaborn not installed, skipping confusion matrix.")
