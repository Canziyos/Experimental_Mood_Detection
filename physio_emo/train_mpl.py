import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from models import MoodMLP
import torch.nn as nn
import numpy as np

# Load and prepare dataset
data = pd.read_csv("physio_labeled.csv")
shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Encode labels
encoder = LabelEncoder()
y_np = encoder.fit_transform(shuffled["label"])
y = torch.tensor(y_np, dtype=torch.long)

# Compute class weights using NumPy
class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1, 2]), y=y_np)
weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

# Prepare features
X = torch.tensor(shuffled[["heart_rate", "gsr"]].values, dtype=torch.float32)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate model
model = MoodMLP(input_dim=2, hidden_dim=16, output_dim=3).to(device)

# Weighted loss function
criterion = nn.CrossEntropyLoss(weight=weight_tensor.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1:02d}, Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "mlp_model.pth")
print("Model saved to mlp_model.pth")
