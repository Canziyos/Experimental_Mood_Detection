
## **A Multimodal Fusion Approach: Emotion Identification from Audio and Video Using Convolutional Neural Network**  
*(Srihari M., Bhaskaran V., 2024)*

A complete pipeline for **emotion recognition** using both **speech (audio)** and **facial expression (video)** data, fused using a **late fusion** strategy with deep learning models.

---
- **audio and visual inputs**.
- Predict complex emotions beyond simple sentiment analysis (not just “happy” or “sad”).
- Focus on **simple, low-compute architectures** that still give **high accuracy**.
- Evaluate on the **RAVDESS dataset** (Ryerson Audio-Visual Database of Emotional Speech and Song).

---

## **Video Modality: Facial Expression Recognition (FER)**

### Preprocessing
- Videos are **split into 22 static frames** using OpenCV.
- Faces are cropped using **Haar Cascade face detection**.
- Images are resized to **48x48** to reduce memory and speed up training.
- **Image embeddings** are generated using a pre-trained **Vision Transformer (ViT)**:
  - Specifically: `google/vit-base-patch16-224-in21k`.

### Video Model Architectures
Several CNN-based models were tested. The best-performing one was:

- **2D 2CNN + 3 Linear Layers**:
  - 2 Conv layers with 32 and 64 filters.
  - Flattened to 48,640 neurons.
  - Fully connected layers: 128 → 64 → 8 (emotion classes).
  - Softmax for output probabilities.

**Best video-only model performance**:
- Accuracy: **84.57%**
- F1-score: **0.8053**

---

## **Audio Modality: Speech Emotion Recognition (SER)**

### Preprocessing and Feature Extraction
- Extracted features:
  - **Zero Crossing Rate (ZCR)**: measures frequency sign changes.
  - **Root Mean Square (RMS)**: average energy/loudness.
  - **Mel Frequency Cepstral Coefficients (MFCCs)**: audio spectral shape.

- Data augmentation techniques:
  - **Gaussian noise**, **pitch shifting**, and combined **noise + pitch** variations.

### Audio Model Architectures
Several CNN and CNN+Transformer models were tested. The top performer:

- **1D 5CNN + 2 Linear Layers**:
  - 5 convolutional layers: 512 → 512 → 256 → 256 → 128 filters.
  - Flattened to 23,936 features.
  - Linear layers: 512 → 8 classes (Softmax output).

**Best audio-only model performance**:
- Accuracy: **78.76%**
- F1-score: **0.7851**

---

### Late Fusion
- Predictions are made separately.
- The **probability distributions** (Softmax outputs) are **averaged**.
- Final emotion prediction is taken as the **argmax** of the averaged vector.

**Fusion performance** (on 1440 RAVDESS samples):
- Accuracy: **97.22%**
- F1-score: **0.9722**
- Precision/Recall: ~97% each

> **Extremely high accuracy**, likely due to the **clean and balanced dataset (RAVDESS)** and careful preprocessing.

---

## Dataset: RAVDESS
- Contains **1440 audio-visual samples** from **24 professional actors**.
- Each actor expresses 8 emotions:
  - Anger, Calm, Disgust, Fear, Happy, Neutral, Sad, Surprise.

---

## Observations (from the Authors)

- Complex deep networks **did not always outperform** simpler ones.
- Preprocessing (e.g., facial cropping and feature-rich embeddings) had major impact.
- The **Vision Transformer (ViT)** helped improve FER performance substantially.
- **Late fusion** proved simpler and more effective than earlier attempts with **early fusion** or attention-based models.

---

## Conclusion
- Even simple CNN-based architectures can be highly effective when:
  - Preprocessing is solid.
  - Feature extraction is rich (e.g., using pretrained ViT).
- Fusion improves robustness significantly, especially in noisy or ambiguous scenarios.
- This pipeline is suitable for Chatbots, intelligent assistants, emotion-aware interfaces etc.

---

# Note:
## Feature Embedding
means:
Embeddings (i.e., vector representations of data) that capture a lot of meaningful and discriminative information about the input, making it easier for the model to learn and classify.

In this paper:
They used pretrained Vision Transformer (ViT) models to extract features from face images.

Instead of raw pixels or simple filters, ViT outputs high-dimensional vectors that capture facial structure, local expression changes (mouth curve, eyebrow raise), Global context of the image (e.g., head orientation, lighting).

These embeddings are considered “feature-rich” because they hold more emotion-relevant information than basic hand-crafted features like HOG or LBP.

Because raw pixels are redundant, noisy, and unstructured. Embeddings compress the image into something compact. numerically stable, and semantically informative.

