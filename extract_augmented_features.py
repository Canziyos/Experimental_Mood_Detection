"""
Extraction, augmentation and normalization.

This script processes all .wav files under data/audio by:
- Extracting ZCR, RMS, and MFCC features for each file.
- Normalizing the features:
- Padding or trimming them to fixed length (15, 300).
- Saving them as .npy files into augmented_features dir.
- Augmenting each audio with:
    - Pitch shift (+2 semitones).
    - Gaussian noise.
So you end up with 3 .npy files per input .wav
"""

import os
import numpy as np
import librosa


aud_dir = os.path.abspath(os.path.join("..", "dataset", "Audio"))
out_dir = os.path.abspath(os.path.join("..", "augmented_features"))


sample_rate = 44100  # Match audio extracted via moviepy which uses fmpeg under hood and i used it.
n_mfcc = 13          # Standard for speech/emotion tasks; captures core spectral features.
frames = 300         # fixed-length inputs; fits our speech clips (2–4 sec).


os.makedirs(out_dir, exist_ok=True)

def extract_features(y, sr):
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.vstack([zcr, rms, mfcc])

def pad_or_trim(features):
    current = features.shape[1]
    if current < frames:
        return np.pad(features, ((0, 0), (0, frames - current)), mode='constant')
    return features[:, :frames]

def normalize_sample(features):
    normed = np.zeros_like(features)
    for i in range(features.shape[0]):
        row = features[i]
        mean = np.mean(row)
        std = np.std(row) if np.std(row) > 1e-6 else 1.0
        normed[i] = (row - mean) / std
    return normed

def save_feature(y, sr, save_path):
    feat = extract_features(y, sr)
    feat = pad_or_trim(feat)
    feat = normalize_sample(feat)
    if feat.shape == (15, frames):
        np.save(save_path, feat)
        print(f"  Saved: {save_path}")

print("Starting augmented audio feature extraction...")

for label in os.listdir(aud_dir):
    class_path = os.path.join(aud_dir, label)
    if not os.path.isdir(class_path):
        continue

    save_class_path = os.path.join(out_dir, label)
    os.makedirs(save_class_path, exist_ok=True)

    for fname in os.listdir(class_path):
        if not fname.lower().endswith(".wav"):
            continue

        stem = os.path.splitext(fname)[0]
        fpath = os.path.join(class_path, fname)

        try:
            y, sr = librosa.load(fpath, sr=sample_rate)

            # Save original feature.
            save_feature(y, sr, os.path.join(save_class_path, f"{stem}.npy"))

            # Pitch-shift.
            y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
            save_feature(y_pitch, sr, os.path.join(save_class_path, f"{stem}_pitch.npy"))

            # Noise-add.
            noise = np.random.normal(0, 0.005, y.shape)
            y_noise = y + noise
            save_feature(y_noise, sr, os.path.join(save_class_path, f"{stem}_noise.npy"))

        except Exception as e:
            print(f"Failed on {fpath}: {e}")

print("Augmented feature extraction completed.")
