# Experimental_Mood_Detection
Personal sandbox for experimenting with MER models, data strategies, and uncertainty handling in mood detection for elderly care.


### Reference

This repo implements key ideas from:

**Srihari M., Bhaskaran V.**  
*A Multimodal Fusion Approach: Emotion Identification from Audio and Video Using Convolutional Neural Network*.  
IEEE CVMI 2024. [DOI: 10.1109/CVMI61877.2024.10782687](https://doi.org/10.1109/CVMI61877.2024.10782687)

## What has been done so far

- Audio model is based on Srihari et al., with slight modifications.

### Dataset Preparation

- Please review the `README` files in both `clean_features` and `augmented_features` to locate your dataset and understand the folder structure.
- To build the datasets, run the following scripts (found in the `scripts` folder):
  - `build_clean_dataset.py`
  - `build_augmented_dataset.py`
- Once the datasets are prepared, you can proceed to run `audio_train.py`.  
  (Note: other components will be added gradually.)



# 1‑D Audio Emotion Classifier

A minimal, modular PyTorch pipeline for training and evaluating a 1‑D CNN (`AudioCNN1D`) on speech‑emotion features.

---

## Contents

* `config.py`  Global hyper‑parameters and paths.
* `data_loader.py`  Data loading + DataLoader builders.
* `trainer.py` Training loop + scheduler.
* `evaluation.py` Validation/Test helpers & plots.
* `train.py`  Command‑line entry point.
* `models/audio_model.py` CNN architecture.

---

## Requirements

Install everything:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Data Preparation

The following NumPy arrays under can be placed, generated, in `processed_data/`:

* `X_aug.npy`, `y_aug.npy` Augmented dataset
* `X_aud.npy`, `y_aud.npy` Raw/clean dataset

Each `X` array is shaped `(N, C, L)` where **C** = feature channels and **L** = sequence length.  Label vectors `y` use integer‑encoded class IDs.

---

## Quick Start

Train on the augmented set:

```bash
python train.py augmented
```

Train on the clean set:

```bash
python train.py clean
```

All runtime options live in **`config.py`** – edit values or subclass `Config` for experiments.

---

## Outputs

* Best checkpoint → `checkpoints/audio_cnn1d_<mode>.pth`
* Classification report → `checkpoints/classification_report_<mode>.txt`
* Confusion‑matrix PNG → `checkpoints/cm_<mode>.png`

---

## Train From Scratch on Your Own Data

1. Extract your features → `X_custom.npy`, labels → `y_custom.npy` and drop them in `processed_data/`.
2. Add a `"custom"` case to `data.load_np_data()` pointing at those filenames.
3. In `config.py`, set `mode = "custom"` and tweak any hyper‑params.
4. Run `python train.py custom`.

---

## Configuration Tips

* **Batch size, epochs, LR** etc. are plain dataclass fields – change & commit.
* Scheduler uses `StepLR`; swap for cosine or ReduceLROnPlateau in `trainer.py`.
* Handle class imbalance via `WeightedRandomSampler` or `CrossEntropyLoss(weight=...)`.

---

## License

Apache‑2.0 – see the LICENSE file for details.

## Support / Questions

Open an issue on GitHub or email canziyos1@gmail.com






