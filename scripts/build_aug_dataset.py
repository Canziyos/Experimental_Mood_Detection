"""
Builds audio feature dataset (X, y) using original and augmented features.

Includes:
- Original samples
- Pitch-shifted versions
- Gaussian noise versions

Assumes features are already extracted (.npy files),
stored under emotion folders in the augmented_features directory.

Expected filenames per sample:
- filename.npy
- filename_pitch.npy
- filename_noise.npy

Only includes classes listed in label_map. All others get ignored.
"""

import os
import numpy as np

# Input: location of extracted features (organized by emotion)
feature_dir = os.path.abspath(os.path.join("..", "augmented_features"))

# Output: where to save the final training-ready data
out_dir = os.path.abspath(os.path.join("..", "processed_data"))
os.makedirs(out_dir, exist_ok=True)

# Expected variations for each base sample
aug_suffix = ["", "_pitch", "_noise"]

# Output filenames
X_OUT = "X_aug.npy"
Y_OUT = "y_aug.npy"

# Class label mapping (Surprise removed)
label_map = {
    "Angery": 0,
    "Disgust": 1,
    "Fear": 2,
    "Happy": 3,
    "Neutral": 4,
    "Sad": 5
}

X = []
y = []

print("Starting to build augmented feature dataset...")

# Loop through emotion folders
for label_name, label_idx in label_map.items():
    class_dir = os.path.join(feature_dir, label_name)
    if not os.path.isdir(class_dir):
        print(f"Skipping missing folder: {label_name}.")
        continue

    print(f"\nProcessing class '{label_name}' (label {label_idx})...")

    # Get base files (not pitch or noise variants)
    base_files = [
        f for f in os.listdir(class_dir)
        if f.endswith(".npy") and not any(suffix in f for suffix in ["_pitch", "_noise"])
    ]

    print(f"Found {len(base_files)} base files.")

    added = 0
    skipped = 0

    for base in base_files:
        base_path = os.path.splitext(base)[0]

        for suffix in aug_suffix:
            filename = f"{base_path}{suffix}.npy"
            full_path = os.path.join(class_dir, filename)

            if not os.path.isfile(full_path):
                skipped += 1
                continue

            try:
                features = np.load(full_path)
                if features.shape != (15, 300):
                    print(f"Skipping invalid shape: {filename}, shape={features.shape}.")
                    skipped += 1
                    continue
                X.append(features)
                y.append(label_idx)
                added += 1
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                skipped += 1

    print(f"Added: {added} files | Skipped: {skipped}.")

# Final output
print("\nFinalizing...")

X = np.stack(X)
y = np.array(y)

np.save(os.path.join(out_dir, X_OUT), X)
np.save(os.path.join(out_dir, Y_OUT), y)

print(f"Saved X to '{os.path.join(out_dir, X_OUT)}' with shape {X.shape}.")
print(f"Saved y to '{os.path.join(out_dir, Y_OUT)}' with shape {y.shape}.")
print("Done.")
