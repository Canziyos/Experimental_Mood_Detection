## augmented_features/

This folder contains audio features extracted from **augmented versions** of the original `.wav` files.

For each input file, the script generates:
- The original features
- A pitch-shifted version (+2 semitones)
- A noise-injected version (Gaussian noise added)

So each original audio file results in **three** `.npy` feature files:
- `sample01.npy`
- `sample01_pitch.npy`
- `sample01_noise.npy`

All features:
- Are normalized per row
- Have shape `(15, 300)`
- Are stored in emotion-class subfolders (e.g., `Happy/`, `Sad/`)

---

### How to generate these files:

1. Locate your audio files under `data/audio/`, with one folder per emotion class (e.g., `Angery/`, `Happy/`, etc.).

2. Run the script `extract_features_augmented.py` from the `scripts` folder.

---

### Example:

For this input:

- data/audio/Sad/sample27.wav


You'll get:

- augmented\_features/Sad/sample27.npy
- augmented\_features/Sad/sample27\_pitch.npy
- augmented\_features/Sad/sample27\_noise.npy


These files are later used to construct the full training dataset (`X_aug.npy`, `y_aug.npy`).