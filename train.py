import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.CNN1D import CNN1D
from utils import load_config
from models.FeatureDataset import FeatureDataset
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

# 1. Deterministic Setup.
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 2. Config and Device.
config = load_config("config.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_s = config["training"]["batch_size"]
epochs = config["training"]["epochs"]
lr = config["training"]["lr"]
input_features = config["training"]["input_features"]
num_classes = len(config["classes"])
features_dir = config["npys_dir"]["root"]

# 3. Collect Clean Files Only.
all_files = glob.glob(os.path.join(features_dir, "train", "*", "*.npy"))
# all_files = [
#     f for f in glob.glob(os.path.join(features_dir, "train", "*", "*.npy"))
#     if not any(suffix in os.path.basename(f) for suffix in ["_pitch", "_noise"])
# ]

label_names = sorted({os.path.basename(os.path.dirname(f)) for f in all_files})
label_to_idx = {name: idx for idx, name in enumerate(label_names)}
label_dict = {f: label_to_idx[os.path.basename(os.path.dirname(f))] for f in all_files}
labels = [label_dict[f] for f in all_files]

# 4. Stratified Split.
train_f, val_f, _, _ = train_test_split(
    all_files, labels, test_size=0.2, stratify=labels, random_state=seed
)

print("Train class distribution:", Counter([label_dict[f] for f in train_f]))
print("Val class distribution:", Counter([label_dict[f] for f in val_f]))

# 5. Dataset and Dataloaders.
train_dataset = FeatureDataset(label_dict, train_f, augment=True)
val_dataset = FeatureDataset(label_dict, val_f, augment=False)

train_loader = DataLoader(train_dataset, batch_size=batch_s, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_s)

# 6. Model, Loss, Optimizer, Scheduler.
model = CNN1D(input_features=input_features, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)


# 7. Early Stopping and Checkpoint.
best_val_loss = float("inf")
patience = 5
patience_counter = 0
ckpt_path = config["ckpt"]["cnn1d"]
os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)


# 8. Training Loop.
for epoch in range(epochs):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    print(f"\n[Epoch {epoch+1}/{epochs}] Training...")

    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()

        # Gradient clipping.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y_batch.cpu().tolist())

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
            print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

    avg_train_loss = total_loss / len(train_loader)
    train_acc = accuracy_score(all_labels, all_preds)
    print(f"==> Epoch {epoch+1} Summary: Train Loss = {avg_train_loss:.4f}, Accuracy = {train_acc:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    val_preds, val_labels = [], []
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            logits = model(x_val)
            loss = criterion(logits, y_val)
            val_loss += loss.item()
            val_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            val_labels.extend(y_val.cpu().tolist())

    avg_val_loss = val_loss / len(val_loader)
    val_acc = accuracy_score(val_labels, val_preds)
    print(f"==> Validation Loss = {avg_val_loss:.4f}, Accuracy = {val_acc:.4f}")

    # Step LR scheduler.
    scheduler.step(avg_val_loss)

    # Early stopping logic.
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': best_val_loss,
        }, ckpt_path)
        print(f"==> Validation loss improved. Checkpoint saved to {ckpt_path}")
    else:
        patience_counter += 1
        print(f"==> No improvement. Patience = {patience_counter}/{patience}")
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

print("Training complete.")
