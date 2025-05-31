import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.CNN1D import CNN1D
from utils import load_config
from models.FeatureDataset import FeatureDataset

from sklearn.metrics import accuracy_score


# Load config
config = load_config("config.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Params
batch_size = config["training"]["batch_size"]
epochs = config["training"]["epochs"]
lr = config["training"]["lr"]
input_features = config["training"]["input_features"]
num_classes = len(config["classes"])

# Load datasets
features_dir = config["npys_dir"]["features"]
all_files = [...]  # your train_files list
label_dict = {...} # your label_dict

train_dataset = FeatureDataset(label_dict, train_files)
val_dataset = FeatureDataset(label_dict, val_files)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Model
model = CNN1D(input_features=input_features, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
for epoch in range(epochs):
    model.train()
    running_loss = 0
    all_preds, all_labels = [], []

    print(f"\n[Epoch {epoch+1}/{epochs}] Training...")

    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y_batch.cpu().tolist())

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
            print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

    acc = accuracy_score(all_labels, all_preds)
    print(f"==> Epoch {epoch+1} Summary: Train Loss = {running_loss:.4f}, Accuracy = {acc:.4f}")

    # Validation
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            logits = model(x_val)
            val_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            val_labels.extend(y_val.cpu().tolist())

    val_acc = accuracy_score(val_labels, val_preds)
    print(f"==> Validation Accuracy: {val_acc:.4f}")
