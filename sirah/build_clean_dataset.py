"""
Builds X and y from clean (non-augmented) features.

Each emotion is a subfolder in clean_features/.
Each .npy file must be shape (15, 300) or it's skipped.

This version matches our label_map with 6 emotion classes (no Surprise).
Saves final X and y to processed_data/ as .npy files.
"""

import os
import numpy as np

# clean_features folder.
features_dir = os.path.abspath(os.path.join("..", "clean_features"))

# Where to save the final dataset.
out_dir = os.path.abspath(os.path.join("..", "processed_data"))


os.makedirs(out_dir, exist_ok=True)

# Output file names.
output_x = "X_aud.npy"
output_y = "y_aud.npy"

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

print("Building dataset from clean features...")

for emotion_name, label_idx in label_map.items():
    emotion_dir = os.path.join(features_dir, emotion_name)
    if not os.path.isdir(emotion_dir):
        print(f"Folder not found: {emotion_dir}")
        continue

    print(f"Processing label '{emotion_name}' → {label_idx}")
    file_count = 0

    for fname in os.listdir(emotion_dir):
        if not fname.endswith(".npy"):
            continue

        fpath = os.path.join(emotion_dir, fname)
        try:
            features = np.load(fpath)
            if features.shape == (15, 300):
                X.append(features)
                y.append(label_idx)
                file_count += 1
            else:
                print(f"Skipped invalid shape {features.shape}: {fpath}")
        except Exception as e:
            print(f"Error loading file {fpath}: {e}")

    print(f"Added {file_count} samples for '{emotion_name}'.")

X = np.stack(X)
y = np.array(y)

np.save(os.path.join(out_dir, output_x), X)
np.save(os.path.join(out_dir, output_y), y)

print(f"\nSaved X to '{os.path.join(out_dir, output_x)}' with shape {X.shape}.")
print(f"Saved y to '{os.path.join(out_dir, output_y)}' with shape {y.shape}.")
print("Done.")
