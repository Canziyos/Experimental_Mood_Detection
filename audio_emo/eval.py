import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from models import MoodMLP

# Load and prepare data
data = pd.read_csv("physio_labeled.csv")
shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(shuffled["label"])

# Convert to tensors
X = torch.tensor(shuffled[["heart_rate", "gsr"]].values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Split same way as training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
test_ds = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MoodMLP(input_dim=2, hidden_dim=16, output_dim=3).to(device)
model.load_state_dict(torch.load("mlp_model.pth"))
model.eval()

# Evaluate
correct = 0
total = 0

with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

acc = correct / total
print(f"MLP Test Accuracy: {acc:.2f}")
