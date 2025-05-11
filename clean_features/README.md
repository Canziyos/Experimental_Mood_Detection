## clean_features/

This folder contains extracted audio features from the original `.wav` files in `data/audio/`.

Each file here is saved as a `.npy` array with shape `(15, 300)`, representing:
- ZCR (Zero-Crossing Rate)
- RMS (Root Mean Square energy)
- 13 MFCC coefficients

The features are:
- Normalized per row (mean = 0, std = 1)
- Padded or trimmed to 300 frames
- Saved under the same emotion subfolder as the original audio

---

### How to generate these files:

1. Make sure your original `.wav` files are under `data/audio/`, with one folder per emotion class (e.g., `Angery/`, `Happy/`, etc.).

2. Run the script `extract_clean_features.py` in the `scripts` folder


### Output example:

For this input:

data/audio/Happy/sample01.wav


You'll get:

```
clean_features/Happy/sample01.npy
```

These files are used to construct the full dataset (`X.npy` and `y.npy`) before training.