import os
import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils import load_config
from AudioFeatureDataset import AudioFeatureDataset

config = load_config("config.yaml")
features_dir = config["npys_dir"]["features"]

# 1. Find all .npy files.
all_files = glob.glob(os.path.join(features_dir, "*", "*.npy")) 

# 2. Map each file to a class label based on the parent folder name.
label_names = sorted({os.path.basename(os.path.dirname(f)) for f in all_files})
label_to_idx = {name: idx for idx, name in enumerate(label_names)}

label_dict = {}
for full_path in all_files:
    folder = os.path.basename(os.path.dirname(full_path))  # parent folder
    base = os.path.splitext(os.path.basename(full_path))[0]
    label_dict[full_path] = label_to_idx[folder]

# 3. Stratified split.
labels = [label_dict[f] for f in all_files]
train_files, temp_files, _, temp_labels = train_test_split(all_files, labels, test_size=0.2, stratify=labels, random_state=42)
val_files, test_files = train_test_split(temp_files, test_size=0.5, stratify=temp_labels, random_state=42)

# 4. Instantiate datasets (AudioFeatureDataset loads npy by path and looks up label in label_dict)
train_dataset = AudioFeatureDataset(label_dict, train_files)
val_dataset = AudioFeatureDataset(label_dict, val_files)
test_dataset = AudioFeatureDataset(label_dict, test_files)

# 5. Create data loaders.
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)
