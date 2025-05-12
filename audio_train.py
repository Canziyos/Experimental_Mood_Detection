"""
Train and evaluate 1D CNN on either clean or augmented dataset.

Usage:
    python train.py aug to use X_aug.npy / y_aug.npy
    python train.py aud  to use X_aud.npy / y_aud.npy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from models.audio_model import AudioCNN1D
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import time

class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("aug", "aud"):
        print("Usage: python train.py [aug|aud]")
        sys.exit(1)

    mode = sys.argv[1]
    name = "augmented" if mode == "aug" else "clean"

    print(f"STARTING TRAINING on {name.upper()} data...")

    data_dir = os.path.abspath(os.path.join("processed_data"))
    x_path = os.path.join(data_dir, f"X_{mode}.npy")
    y_path = os.path.join(data_dir, f"y_{mode}.npy")

    X = np.load(x_path)
    y = np.load(y_path)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.10, random_state=42, stratify=y_trainval
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(EmotionDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True, num_workers=2)
    val_loader   = DataLoader(EmotionDataset(X_val_tensor, y_val_tensor), batch_size=32, shuffle=False, num_workers=2)
    test_loader  = DataLoader(EmotionDataset(X_test_tensor, y_test_tensor), batch_size=32, shuffle=False, num_workers=2)

    model = AudioCNN1D().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    checkpoint_dir = os.path.abspath(os.path.join("checkpoints"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, f"audio_cnn1d_{name}.pth")

    num_epochs = 10
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        start_time = time.time()

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()

            if i % 10 == 0:
                print(f"  Batch {i}/{len(train_loader)} — Loss: {loss.item():.4f}")

        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        duration = time.time() - start_time

        print(f"Epoch {epoch+1} completed in {duration:.2f}s | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"Model improved. Saved to '{model_path}'.")

    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_correct = 0
    test_total = 0
    class_preds = []
    class_true = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            test_total += labels.size(0)
            test_correct += (preds == labels).sum().item()
            class_preds.extend(preds.cpu().numpy())
            class_true.extend(labels.cpu().numpy())

    test_acc = 100 * test_correct / test_total
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")

    print("\nClassification Report:")
    print(classification_report(class_true, class_preds, digits=2))

    cm = confusion_matrix(class_true, class_preds)
    class_names = ["Angery", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix on {name.upper()} Test Set")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
