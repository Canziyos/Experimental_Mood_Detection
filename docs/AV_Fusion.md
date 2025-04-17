## Feature and Decision Level Audio-visual Data Fusion in Emotion Recognition Problem

### Introduction

This paper addresses the challenge of emotion recognition in human-computer interaction (HCI) by combining **audio and visual signals**. While unimodal systems have shown decent performance, their limitations in real-world environments motivate the use of **multimodal fusion**.

The study evaluates two primary fusion strategies:
1. audio and visual features are combined before classification.
2. predictions from separate classifiers are integrated.

Combining these modalities enhances emotion classification accuracy?

---

### Methodology

#### Feature Extraction:
- **Audio features**: Derived from speech signals.
- **Visual features**: Extracted using three different algorithms:
  - **Local Binary Patterns (LBP)**: Classical texture-based method for still images.
  - **Quantized Local Zernike Moments (QLZM)**: Advanced method for extracting facial features.
  - **Local Binary Patterns on Three Orthogonal Planes (LBP-TOP)**: Captures spatial-temporal information from video sequences.

#### Dimensionality Reduction:
- **(PCA)** is applied to both audio and visual datasets to reduce dimensionality and noise.

#### Classification Models:
- **(SVC)** trained using Sequential Minimal Optimization (SMO).
- **(NN)**.

#### Experimental Setup:
- Each classifier is tested on:
  - Audio-only features.
  - Each of the three visual feature sets.
  - Feature-level fused datasets (audio + visual).
- Decision-level fusion is applied by selecting the best-performing classifier-modality combination for each emotion and combining their outputs.

---

### Results

#### Unimodal Systems:
- Different classifier-modality combinations perform better for specific emotions.
- Some combinations are better at recognizing “anger” while others excel at detecting “fear” or “neutral”.

#### Feature-Level Fusion:
- Combining audio and visual features improved the accuracy by **4%** compared to the best unimodal system, (complementary strengths of each modality!)?.

#### Decision-Level Fusion:
- Further improvement of **3%** over the feature-level fusion.
- This approach uses emotion-specific best classifiers and integrates their decisions through strategies like averaging or voting.

---

### Related Work

The paper reviews several influential studies:
- **Rashid et al. (2012)**: Applied decision-level fusion using Bayes sum rule and observed a notable increase in performance.
- **Kahou et al. (2013)**: Used deep neural networks (CNN, DBN, autoencoders) in a competition setting, achieving 41% accuracy.
- **Cruz et al. (2012)**: Modeled temporal changes in features and improved classification using derivatives and HMMs.
- **Soleymani et al. (2012)**: Incorporated EEG and eye-tracking for arousal/valence prediction.
- **Busso et al. (2004)**: Demonstrated that combining facial expression and acoustic data can boost accuracy to 90% using both feature- and decision-level fusion.

---

### Conclusions

- The study demonstrates that **audio-visual fusion**, particularly at the **decision level**, offers measurable improvements in emotion recognition accuracy.
- It also emphasizes that no single modality or model is universally superior — rather, **emotion-specific combinations** perform best.
- Future work: refining fusion strategies and exploring temporal modeling.
