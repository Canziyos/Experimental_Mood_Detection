# Evaluate the trained CNN1D on the held-out test subset.
import os, glob, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils import load_config
from models.CNN1D import CNN1D
from models.FeatureDataset import FeatureDataset


SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Config / Paths.
cfg       = load_config("config.yaml")
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_mel    = cfg["training"]["input_features"]
n_classes = len(cfg["classes"])

test_dir  = cfg["npys_dir"]["npy_test"]            # "features/test"
ckpt_path = cfg["ckpt"]["audio_model"]             # best model path.
batch     = cfg["training"]["batch_size"]

# Gather test *.npy.
test_files = glob.glob(os.path.join(test_dir, "*", "*.npy"))
assert test_files, f"No .npy files found under {test_dir}"

get_class     = lambda p: os.path.basename(os.path.dirname(p))
class_to_idx  = {c: i for i, c in enumerate(sorted({get_class(f) for f in test_files}))}
file_to_label = {f: class_to_idx[get_class(f)] for f in test_files}

# Dataset / Loader.
test_ds = FeatureDataset(file_to_label, test_files)
test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False,
                         num_workers=0,     # keep 0 if we are on Windows / want simple debug.
                         pin_memory=(device.type == "cuda"))

# Model & Checkpoint.
model = CNN1D(input_features=n_mel, num_classes=n_classes).to(device)
ckpt  = torch.load(ckpt_path, map_location=device)

if "model_state_dict" in ckpt:         # training script saved under this key.
    model.load_state_dict(ckpt["model_state_dict"])
elif "model" in ckpt:                  # my example script.
    model.load_state_dict(ckpt["model"])
else:                                  # plain state-dict
    model.load_state_dict(ckpt)

model.eval()

# evaluation loop.
all_pred, all_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:          # xb shape (B, 39, T)
        xb = xb.to(device)
        logits = model(xb)
        all_pred.extend(logits.argmax(1).cpu().tolist())
        all_true.extend(yb.tolist())

acc = accuracy_score(all_true, all_pred)
print(f"\nTEST accuracy: {acc:.4f}\n")

print("Classification report:")
print(classification_report(all_true, all_pred, target_names=cfg["classes"], digits=4))

print("Confusion matrix:")
print(confusion_matrix(all_true, all_pred))
