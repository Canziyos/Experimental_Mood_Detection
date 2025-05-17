

# import numpy as np
# from pathlib import Path

# meta_dir = Path("mobilenet/audio/processed_data")

# def load_paths(split):
#     paths = np.load(meta_dir / f"X_{split}_paths.npy", allow_pickle=True)
#     # Normalize: lower case, forward slashes
#     return set(Path(p).as_posix().lower() for p in paths)

# train_paths = load_paths("train")
# val_paths   = load_paths("val")
# test_paths  = load_paths("test")

# train_val_overlap = train_paths & val_paths
# train_test_overlap = train_paths & test_paths
# val_test_overlap = val_paths & test_paths

# def report(overlap_set, name):
#     print(f"{name} overlap: {len(overlap_set)}")
#     if overlap_set:
#         print("  Example(s):")
#         for p in list(overlap_set)[:5]:
#             print("   ", p)

# report(train_val_overlap, "Train/Val")
# report(train_test_overlap, "Train/Test")
# report(val_test_overlap, "Val/Test")



# import numpy as np
# import torchaudio
# from pathlib import Path
# import json

# # === Hardcoded path to processed metadata
# meta_dir = Path("mobilenet/audio/processed_data")
# split = "train"  # ← change to "val" or "test" as needed

# x_path = meta_dir / f"X_{split}_paths.npy"
# y_path = meta_dir / f"y_{split}.npy"
# label_map_path = meta_dir / "labelmap_logmel.json"

# # === Load metadata
# paths = np.load(x_path, allow_pickle=True)
# labels = np.load(y_path)

# print(f"\nLoaded {len(paths)} paths and {len(labels)} labels for split: '{split}'")

# # === Consistency check
# if len(paths) != len(labels):
#     print("Mismatch between number of paths and labels.")
# else:
#     print("Path and label count match.")

# # === Load label map to print class names
# with open(label_map_path) as f:
#     label_map = json.load(f)
# inv_label_map = {v: k for k, v in label_map.items()}

# # === Print first 10 samples
# print("\nSample check (first 10):")
# for i in range(min(10, len(paths))):
#     path = paths[i]
#     label = labels[i]
#     class_name = inv_label_map.get(int(label), "UNKNOWN")
#     print(f"[{i:02}] {path}  →  {label} ({class_name})")

#     # Optional: check file exists and is loadable
#     try:
#         torchaudio.load(path)
#     except Exception as e:
#         print(f"    Error loading file: {e}")
# import numpy as np
# from pathlib import Path

# meta_dir = Path("mobilenet/audio/processed_data")

# def load_paths(split):
#     paths = np.load(meta_dir / f"X_{split}_paths.npy", allow_pickle=True)
#     # Normalize: lower case, forward slashes
#     return set(Path(p).as_posix().lower() for p in paths)

# train_paths = load_paths("train")
# val_paths   = load_paths("val")
# test_paths  = load_paths("test")

# train_val_overlap = train_paths & val_paths
# train_test_overlap = train_paths & test_paths
# val_test_overlap = val_paths & test_paths

# def report(overlap_set, name):
#     print(f"{name} overlap: {len(overlap_set)}")
#     if overlap_set:
#         print("  Example(s):")
#         for p in list(overlap_set)[:5]:
#             print("   ", p)

# report(train_val_overlap, "Train/Val")
# report(train_test_overlap, "Train/Test")
# report(val_test_overlap, "Val/Test")
# import numpy as np
# from pathlib import Path

# splits = ["train", "val", "test"]
# base_dir = Path("mobilenet/audio/processed_data")
# aug_mode = "none"

# for split in splits:
#     folder = base_dir / f"logmel_{split}_{aug_mode}_batches"

#     orig_files = sorted(folder.glob("orig_*.npy"))
#     label_files = sorted(folder.glob("label_*.npy"))

#     print(f"\n {folder.name}")
#     print(f" - orig_*.npy   : {len(orig_files)}")
#     print(f" - label_*.npy  : {len(label_files)}")

#     # Check first file shape and label
#     if orig_files and label_files:
#         x = np.load(orig_files[0])
#         y = np.load(label_files[0])
#         print(f"   sample shape: {x.shape}, label: {y.item()}")

#     # Show first 5 labels
#     print("   First 5 labels:")
#     for i in range(min(5, len(label_files))):
#         y = np.load(label_files[i])
#         print(f"    {label_files[i].name} → {y.item()}")

#     # Unique label classes
#     unique_labels = {int(np.load(f)) for f in label_files}
#     print(f"   Unique labels found: {sorted(unique_labels)}")
import os
import numpy as np

aug_dir = "mobilenet/audio/processed_data/logmel_train_pitch_batches"
aug_files = sorted([f for f in os.listdir(aug_dir) if f.startswith("aug_")])
lbl_files = sorted([f for f in os.listdir(aug_dir) if f.startswith("label_aug")])

for aug, lbl in zip(aug_files, lbl_files):
    print(f"{aug}  <->  {lbl}")