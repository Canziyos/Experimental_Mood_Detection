import os
import time
import glob
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from typing import Dict
from torch.utils.data import DataLoader

# === hyperparameters and paths ===
n_classes = 6
lr = 1e-3
batch_size = 32
num_epochs = 15
step_size = 5
gamma = 0.5

checkpoint_dir = "checkpoints"
best_model_path = os.path.join(checkpoint_dir, "mobilenet_aud.pth")
history_path = os.path.join(checkpoint_dir, "train_history.json")

os.makedirs(checkpoint_dir, exist_ok=True)

def train(model: nn.Module, loaders: Dict[str, DataLoader]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    label_files = glob.glob("mobilenet/audio/processed_data/logmel_train_none_batches/label_*.npy")
    y_all = np.array([np.load(f) for f in label_files])

    weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_all)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    opt = optim.Adam(model.parameters(), lr=lr)

    # To avoid sudden drops like StepLR, CosineAnnealing will be tested:
    # It fades the learning rate smoothly from the initial value to near-zero over the course of training.
    #sch = CosineAnnealingLR(opt, T_max=num_epochs)

    sch = StepLR(opt, step_size=step_size, gamma=gamma)

    best, hist = 0.0, defaultdict(list)

    for ep in range(num_epochs):
        print(f"\n[Epoch {ep+1}/{num_epochs}] Training started...")
        model.train()
        tot_l = tot_c = tot_n = 0

        for batch_idx, (spec, y) in enumerate(loaders["train"], 1):
            start = time.time()
            spec, y = spec.to(device, non_blocking=True), y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(spec)
            loss = criterion(out, y)
            loss.backward()
            opt.step()

            tot_l += loss.item()
            tot_c += (out.argmax(1) == y).sum().item()
            tot_n += y.size(0)

            elapsed = (time.time() - start) * 1000
            batch_acc = 100 * (out.argmax(1) == y).sum().item() / y.size(0)
            print(f"  Batch {batch_idx:03d}/{len(loaders['train'])} | "
                  f"Loss: {loss.item():.4f} | Acc: {batch_acc:.2f}% | Time: {elapsed:.1f} ms")

        tr_acc = 100 * tot_c / tot_n
        tr_loss = tot_l / len(loaders["train"])

        # ---- Validation ----
        model.eval()
        v_c = v_n = v_l = 0
        print("[Validation] Running...")
        with torch.no_grad():
            for spec, y in loaders["val"]:
                o = model(spec.to(device))
                v_l += criterion(o, y.to(device)).item()
                v_c += (o.argmax(1) == y.to(device)).sum().item()
                v_n += y.size(0)

        v_acc = 100 * v_c / v_n
        v_loss = v_l / len(loaders["val"])
        sch.step()

        print(f"[Summary] Ep {ep+1:02d} | Train: {tr_loss:.3f} / {tr_acc:.1f}% | Val: {v_loss:.3f} / {v_acc:.1f}%")
        hist["train_acc"].append(tr_acc)
        hist["val_acc"].append(v_acc)

        if v_acc > best:
            best = v_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best val accuracy: {best:.2f}%")

    # Save training history
    hist_clean = {k: [float(x) for x in v] for k, v in hist.items()}
    with open(history_path, "w") as f:
        json.dump(hist_clean, f, indent=2)
    print(f"Training history saved to {history_path}")

    model.load_state_dict(torch.load(best_model_path))
    return model, hist


# if __name__ == "__main__":
#     import sys
#     import os
#     sys.path.insert(0, os.path.abspath('.'))
#     from mobilenet.audio.audio_loader import make_audio_loaders
#     from mobilenet.audio.audio_model import AudioMobileNetV2

#     print("[TEST] Creating data loaders...")
#     loaders = make_audio_loaders()

#     print("[TEST] Fetching one batch from training loader...")
#     batch = next(iter(loaders["train"]))
#     specs, labels = batch

#     print(f"[TEST] Batch specs shape: {specs.shape}")   # Expected: (batch_size, 1, 96, 192)
#     print(f"[TEST] Batch labels shape: {labels.shape}") # Expected: (batch_size,)

#     print(f"[TEST] First 5 labels: {labels[:5].tolist()}")

#     print("[TEST] Model initialization...")
#     model = AudioMobileNetV2(pretrained=False, freeze_backbone=False)
#     print(model)

#     print("[TEST] Forward pass with one batch...")
#     out = model(specs)
#     print(f"[TEST] Output shape: {out.shape}")  # Expected: (batch_size, num_classes)

#     print("[TEST] Sanity check complete.")
