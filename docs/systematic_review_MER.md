## **EMOTION, FEELING, MOODS, AFFECT, SENTIMENTS, AND EMOTIONAL DIMENSION**



## FER in the Context of Multimodal Emotion Recognition (MER)

- They are **non-intrusive**, rich in **affective information**, and suitable for **real-time emotion recognition systems**.

### Psychological Basis
- FER is rooted in the **discrete emotion theory** (e.g., Ekman's six basic emotions) and **multidimensional models** (Valence-Arousal, Valence-Arousal-Dominance).
- Ekman’s research is highlighted for establishing facial expressions as **biologically grounded and cross-culturally recognized**.

### Feature Extraction Techniques
- Common preprocessing steps: **face detection**, alignment, **normalization**, and **temporal smoothing**.
- Techniques include:
  - **Facial Action Coding System (FACS)**: maps expressions to muscle movements.
  - **Convolutional Neural Networks (CNN)** and **Recurrent Neural Networks (RNN)** for spatial-temporal modeling.
  - **Multiview attention mechanisms** and **cross-attention models** are now used to emphasize critical emotional regions in the face.


## Audio-Visual Emotion Recognition (A-V MER)

### Audio + Visual?
- Provide **complementary information**.
- Facial expressions can be **faked or masked**, but vocal tone often reveals the true emotion.
- Audio helps where facial data is occluded; visual helps when audio is noisy or unavailable.

### Fusion Techs

#### 1. **Early Fusion**
- Merges raw or low-level features from both audio and visual streams.
- Captures **cross-modal correlations** early but can be sensitive to noise and misalignment.

#### 2. **Late Fusion**
- Each modality is **processed independently**, and predictions are combined (e.g., via voting or averaging).
- More **robust to missing or weak modalities**, but less integrated.

#### 3. **Hybrid Fusion**
- Combines both early and late strategies.
- Often uses attention-based or transformer models for **dynamic weighting** of modalities.

#### 4. **Cross-Attention and Transformers**
- Techniques like **Multimodal Transformers** align and fuse A-V data at multiple levels.
- These are now standard in **state-of-the-art** MER systems.

## Example Models and Results

| Model / Study | Modalities | Fusion | Notable Accuracy / Dataset |
|---------------|------------|--------|-----------------------------|
| Le et al. (2023) | Video + Audio + Text | Transformer-based fusion | 78.98% (IEMOCAP), 79.63% (CMU-MOSEI) |
| Mocanu et al. (2023) | Audio + Facial Expressions | Cross-Attention | 89.25% (RAVDESS), 84.57% (CREMA-D) |
| Zhang et al. (2022) | Facial + Speech | Encoder-decoder + attention | +2.8% F1 score boost |
| Feng et al. (2022) | Facial + Speech + Text | Multi-view attention | Notable gains on IEMOCAP, MSP-IMPROV |

---

## Datasets Frequently Used
- **IEMOCAP**: multimodal with speech, facial expressions, and text.
- **CMU-MOSEI**: includes over 23,000 annotated A-V samples.
- **CREMA-D**, **RAVDESS**, **AFEW**, **SFEW**: widely used for A-V FER.
- Many datasets now use **real-world clips**, e.g., movies or interviews, increasing diversity but also noise and occlusion challenges.

## Challenges Highlighted
- **Synchronization** of audio and video streams is essential for good fusion.
- **Occlusion and lighting** issues in facial video still persist.
- **Computational cost** increases significantly with multimodal systems.
- Lack of **standardization in annotations and emotion taxonomies** across datasets.
- Real-time systems face trade-offs between **latency and accuracy**.
