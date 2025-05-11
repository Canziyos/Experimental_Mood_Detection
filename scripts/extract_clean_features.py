"""
- Extracts audio features (ZCR, RMS, MFCC) from .wav files,
- normalizes them, pads/trims to (15, 300), and saves them
as .npy files into a target directory.
"""

import os
import numpy as np
import librosa

aud_dir = os.path.abspath(os.path.join("..", "data", "audio"))
output_dir = os.path.abspath(os.path.join("..", "clean_features"))

SAMPLE_RATE = 44100
N_MFCC = 13
FIXED_FRAMES = 300

os.makedirs(output_dir, exist_ok=True)

def extract_features(filepath):
    try:
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        rms = librosa.feature.rms(y=y)[0]
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        return np.vstack([zcr, rms, mfcc])  # shape: (15, T)
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def pad_or_trim(features, target_frames=FIXED_FRAMES):
    current_frames = features.shape[1]
    if current_frames < target_frames:
        return np.pad(features, ((0, 0), (0, target_frames - current_frames)), mode='constant')
    return features[:, :target_frames]

def normalize_matrix(matrix):
    normed = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        row = matrix[i]
        mean = np.mean(row)
        std = np.std(row) if np.std(row) > 1e-6 else 1.0
        normed[i] = (row - mean) / std
    return normed

print("Starting audio feature extraction...")

# Loop through each emotion folder
for emotion_label in os.listdir(aud_dir):
    emotion_dir = os.path.join(aud_dir, emotion_label)
    if not os.path.isdir(emotion_dir):
        print(f"Skipped non-folder: {emotion_label}")
        continue

    print(f"\nProcessing emotion class: {emotion_label}")
    save_dir = os.path.join(output_dir, emotion_label)
    os.makedirs(save_dir, exist_ok=True)

    file_count = 0

    for fname in os.listdir(emotion_dir):
        if not fname.lower().endswith(".wav"):
            print(f"  Skipped non-wav file: {fname}")
            continue

        fpath = os.path.join(emotion_dir, fname)
        features = extract_features(fpath)

        if features is not None:
            padded = pad_or_trim(features)
            normed = normalize_matrix(padded)
            save_path = os.path.join(save_dir, fname.replace(".wav", ".npy"))
            np.save(save_path, normed)
            print(f"  Saved: {save_path}")
            file_count += 1

    print(f"Finished class '{emotion_label}' with {file_count} files.")

print("Audio feature extraction completed.")
