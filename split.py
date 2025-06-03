# import os
# import shutil
# import random

# src_root = "dataset/animals"                # original folder with all classes
# dst_root = "dataset/animals_split"          # output folder
# train_ratio = 0.8
# random.seed(42)

# for class_name in os.listdir(src_root):
#     class_path = os.path.join(src_root, class_name)
#     if not os.path.isdir(class_path):
#         continue

#     wav_files = [f for f in os.listdir(class_path) if f.lower().endswith(".wav")]
#     random.shuffle(wav_files)

#     split_idx = int(len(wav_files) * train_ratio)
#     train_files = wav_files[:split_idx]
#     val_files = wav_files[split_idx:]

#     for split, file_list in [("train", train_files), ("val", val_files)]:
#         out_dir = os.path.join(dst_root, split, class_name)
#         os.makedirs(out_dir, exist_ok=True)
#         for fname in file_list:
#             src_path = os.path.join(class_path, fname)
#             dst_path = os.path.join(out_dir, fname)
#             shutil.copy2(src_path, dst_path)

# print("Split complete. Output written to:", dst_root)
import os
from collections import defaultdict

root_dir = "dataset/animals"  # or "features" if counting .npy
ext = ".wav"  # or ".npy"

split_counts = defaultdict(lambda: defaultdict(int))  # split -> class -> count

for split in os.listdir(root_dir):  # e.g., "train", "val"
    split_path = os.path.join(root_dir, split)
    if not os.path.isdir(split_path):
        continue

    for class_name in os.listdir(split_path):
        class_path = os.path.join(split_path, class_name)
        if not os.path.isdir(class_path):
            continue

        count = len([f for f in os.listdir(class_path) if f.lower().endswith(ext)])
        split_counts[split][class_name] = count

# Print results
for split in sorted(split_counts):
    print(f"\n{split.upper()} split:")
    for class_name, count in sorted(split_counts[split].items()):
        print(f"  {class_name}: {count} samples")
