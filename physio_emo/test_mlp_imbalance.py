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
y_all = encoder.fit_transform(shuffled["label"])
X_all = shuffled[["heart_rate", "gsr"]]

# Split once to get consistent test set
_, X_test, _, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
df_test = X_test.copy()
df_test["label"] = y_test

# Create imbalance: full 'calm', 30% 'agitated', 10% 'depressed'
imbalanced_df = pd.concat([
    df_test[df_test["label"] == 0].sample(frac=0.3, random_state=42),
    df_test[df_test["label"] == 1],  # calm
    df_test[df_test["label"] == 2].sample(frac=0.1, random_state=42)
])

imbalanced_df = imbalanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Convert to tensors
X_imb = torch.tensor(imbalanced_df[["heart_rate", "gsr"]].values, dtype=torch.float32)
y_imb = torch.tensor(imbalanced_df["label"].values, dtype=torch.long)

test_ds = TensorDataset(X_imb, y_imb)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

# Load trained MLP
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
print(f"[MLP Imbalance Test] Accuracy: {acc:.2f}")
