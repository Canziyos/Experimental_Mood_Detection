# full_train.py - minimal version (clean + _pitch + _noise files)

import os, random, glob
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

from utils import load_config
from models.CNN1D          import CNN1D
from models.FeatureDataset import FeatureDataset

# 1. determinism.
seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# 2. config/device.
cfg = load_config("config.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = cfg["training"]["batch_size"]
epochs = cfg["training"]["epochs"]
lr = cfg["training"]["lr"]
input_features = cfg["training"]["input_features"]
num_classes = len(cfg["classes"])
features_dir = cfg["npys_dir"]["root"]

# 3. collect all files.
all_files = glob.glob(os.path.join(features_dir, "train", "*", "*.npy"))

# 4. label mapping/split.
label_names  = sorted({os.path.basename(os.path.dirname(f)) for f in all_files})
label2idx = {name: i for i, name in enumerate(label_names)}
label_dict = {f: label2idx[os.path.basename(os.path.dirname(f))] for f in all_files}
labels = [label_dict[f] for f in all_files]

train_f, val_f, _, _ = train_test_split(
    all_files, labels, test_size=0.20, stratify=labels, random_state=seed
)

print("Train class distribution:", Counter(label_dict[f] for f in train_f))
print("Val   class distribution:", Counter(label_dict[f] for f in val_f))

# 5. datasets / loaders.
train_ds = FeatureDataset(label_dict, train_f)
val_ds   = FeatureDataset(label_dict, val_f)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size)

# 6. model stack.
model = CNN1D(input_features=input_features, num_classes=num_classes).to(device)
criterion  = nn.CrossEntropyLoss()
optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=2)

# 7. early-stopping setup.
best_val_loss = float('inf')
patience = 8
patience_ctr = 0
ckpt_path = cfg["ckpt"]["cnn1d"]
os.makedirs(cfg["ckpt"]["root"], exist_ok=True)

# 8. training loop .
for epoch in range(epochs):

    model.train()
    tot_loss, preds, gts = 0.0, [], []
    print(f"\n[Epoch {epoch+1}/{epochs}] Training…")

    for i, (xb, yb) in enumerate(train_loader, 1):
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        out   = model(xb)
        loss  = criterion(out, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tot_loss += loss.item()
        preds.extend(out.argmax(1).cpu().tolist())
        gts.extend(yb.cpu().tolist())

        if i % 10 == 0 or i == len(train_loader):
            print(f"batch {i}/{len(train_loader)}  loss {loss.item():.4f}")

    train_loss = tot_loss / len(train_loader)
    train_acc = accuracy_score(gts, preds)
    print(f"==> epoch {epoch+1}: train loss {train_loss:.4f}  acc {train_acc:.4f}")

    # validation.
    model.eval(); val_loss, preds, gts = 0.0, [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            val_loss += criterion(out, yb).item()
            preds.extend(out.argmax(1).cpu().tolist())
            gts.extend(yb.cpu().tolist())

    val_loss /= len(val_loader)
    val_acc = accuracy_score(gts, preds)
    print(f"==> val loss {val_loss:.4f}  acc {val_acc:.4f}")

    scheduler.step(val_loss)

    # early-stopping/checkpoint.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_ctr  = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': best_val_loss
        }, ckpt_path)
        print(f"loss improved – checkpoint saved to {ckpt_path}")
    else:
        patience_ctr += 1
        print(f"no improve ({patience_ctr}/{patience})")
        if patience_ctr >= patience:
            print("Early stopping triggered.")
            break

print("Training complete.")
