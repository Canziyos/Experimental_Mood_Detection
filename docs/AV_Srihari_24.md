# A Multimodal Fusion Approach for Emotion Recognition  
*Srihari M. & Bhaskaran V., 2024*

A complete pipeline that ingests **speech (audio)** and **facial‑expression video**, then predicts one of **eight emotions**.  
The two streams are combined with a **late‑fusion averaging step**.

---

## Key ideas

* Keep the models **lightweight** so they fit on modest hardware.  
* Classify eight discrete emotions (not just “positive vs. negative”).  
* All experiments use the **RAVDESS** audio‑video dataset.

---

## Video branch – Facial Expression Recognition (FER)

| Step | Method |
|------|--------|
|Frame extraction|Split each clip into **22 frames** with OpenCV.|
|Face detection|Haar‑cascade; crop the face region.|
|Resize|Scale faces to **48 × 48 px**.|
|Feature embedder|Pre‑trained **Vision Transformer**: `vit‑base‑patch16‑224‑in21k`.|
|Classifier|**2‑layer 2‑D CNN** (32 & 64 filters) → flatten (48 640 units) → **3 dense layers** (128 → 64 → 8) → soft‑max.|

**Video‑only performance**

* Accuracy **84.6 %**  
* F1 **0.805**

---

## Audio branch – Speech Emotion Recognition (SER)

| Step | Method |
|------|--------|
|Acoustic features|**Zero‑Crossing Rate**, **RMS energy**, **MFCCs**.|
|Augmentation|Add Gaussian noise, pitch‑shift, or both.|
|Classifier|**1‑D CNN** with 5 conv layers (512 → 512 → 256 → 256 → 128 filters) → flatten (23 936 features) → **2 dense layers** (512 → 8) → soft‑max.|

**Audio‑only performance**

* Accuracy **78.8 %**  
* F1 **0.785**

---

## Late fusion

1. Run FER and SER models independently.  
2. **Average** their soft‑max probability vectors.  
3. Choose the emotion with the highest averaged score.

**Fusion results** (1 440 RAVDESS clips)

* Accuracy **97.2 %**  
* F1 **0.972** (precision and recall ≈ 97 %).

> The high score is helped by RAVDESS being clean and well balanced.

---

## RAVDESS in one line

* 24 professional actors.  
* 1 440 clips labelled: anger, calm, disgust, fear, happy, neutral, sad, surprise.

---

## Author observations

* Careful **pre‑processing** and **ViT embeddings** mattered more than huge networks.  
* **Late fusion** outperformed their attempts at early‑ and attention‑based fusion.

---
**Training settings**
- Optimizer: Adam | Learning rate: 0.0001
- Batch size: 32 | Epochs: 100
- Frameworks: Python, Keras, OpenCV

### Take‑aways

* **Simple CNNs** plus strong features (ViT for video, MFCC for audio) deliver solid accuracy.  
* Averaging probabilities makes the system robust when one modality is noisy.  
* Suitable for chat‑bots, virtual agents, and other emotion‑aware interfaces—as long as data resemble studio conditions.

---

### Why ViT embeddings help

A pre‑trained *Vision Transformer* converts each face crop into a compact vector that captures:

* facial geometry  
* local expression changes (e.g., smile curvature, eyebrow raise)  
* global context (pose, lighting)

These vectors hold more emotion‑relevant information than raw pixels or hand‑crafted HOG/LBP features, allowing even a small classifier to learn quickly.

