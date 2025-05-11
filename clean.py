import os

# Directories
audio_dir = os.path.abspath(os.path.join(".", "data", "audio", "Angery"))
clean_dir = os.path.abspath(os.path.join(".", "clean_features", "Angery"))

# Collect valid base names from clean_features (without .npy)
valid_basenames = {
    os.path.splitext(f)[0]
    for f in os.listdir(clean_dir)
    if f.endswith(".npy")
}

removed = 0
for fname in os.listdir(audio_dir):
    if not fname.endswith(".wav"):
        continue

    stem = os.path.splitext(fname)[0]

    if stem not in valid_basenames:
        fpath = os.path.join(audio_dir, fname)
        try:
            os.remove(fpath)
            print(f"Removed: {fpath}")
            removed += 1
        except Exception as e:
            print(f"Failed to remove {fpath}: {e}")

print(f"\nTotal .wav files removed: {removed}")
