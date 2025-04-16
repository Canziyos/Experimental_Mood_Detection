
### Title  
**Multimodal Emotion Recognition Using Visual, Vocal and Physiological Signals: A Review**  
*Udahemuka et al., Applied Sciences, 2024*

---

This review investigates current methods for recognizing emotions and focuses on subtle and dynamic emotional expressions and the application of deep learning for fine-grained emotion recognition.

- Emphasizes **subtle emotion recognition** and the need to detect **micro-expressions** across modalities.  
- Highlights the **importance of dynamic expressions** over static displays for capturing authentic emotional changes.  
- Discusses **deep learning** approaches and their superiority over handcrafted features in recognizing subtle signs.

---

### Modalities  
- **Visual**: Facial expressions, gestures, micro-expressions.  
- **Vocal**: Prosodic and spectral features like pitch, MFCC, energy.  
- **Physiological**: EEG, ECG, EMG, GSR, respiration, etc.  

Each modality provides unique indications, and multimodal approaches aim to combine their strengths while compensating for their limitations.

---

### Dataset Discussion  
- Categorizes datasets as **acted**, **induced**, or **natural**.  
- Stresses the need for **spontaneous**, **diverse**, and **balanced** datasets.  
- Lists many benchmarks for visual and vocal signals, including **FER2013**, **JAFFE**, **CK+**, **EMO-DB**, and **CASME II**.

---

### Methods

**Handcrafted features**  
The article reviews traditional, manually engineered features used for recognizing emotion, especially in visual data:
**Handcrafted features** refer to features that are **manually designed by humans** based on domain knowledge, rather than learned automatically by a model.

These are:
- Extracted using **predefined algorithms**
- Often based on **mathematical or statistical rules**
- Not adapted to the specific dataset during training

---

- A **handcrafted approach** would be:  
  - Apply **Local Binary Pattern (LBP)** to capture skin texture.  
  - Use **HOG** to extract edge directions around eyes and mouth.  
  - Combine these features into a vector and feed into a traditional classifier like **(SVM)**.

- A **deep learning approach** would skip manual feature engineering. We give the image to a *(CNN)**, and it *learns* which patterns (edges, curves, wrinkles, etc.) are useful for emotion recognition — directly from data.

---

- **Local Binary Pattern (LBP)** and **LBP-TOP (Three Orthogonal Planes)**: Common for facial micro-expression recognition, capturing texture and motion over time.
- **Histogram of Oriented Gradients (HOG)** and **SIFT**: Used to describe facial shape and edge information, mainly in macro-expressions.
- **Optical flow** and **optical strain**: Capture subtle motion between frames, important for detecting micro-expressions.
- **Limitations**: These methods rely heavily on prior knowledge, struggle with noisy or imbalanced data, and perform poorly on partial faces or datasets with large variation.

---

**Deep learning architectures**  
The article outlines how deep learning overcomes many of the shortcomings of handcrafted features:

- Convolutional Neural Networks **(CNNs)**: Used to extract spatial features from static facial images or motion-encoded inputs (e.g., optical flow fields).
- **3D-CNNs**: Capture both spatial and temporal features simultaneously from video data. Two-stream variants are also described for different input modalities.
- Recurrent models like **(LSTM)**: Learn the evolution of emotion over time by modeling sequences of spatial features extracted by CNNs.
- Capsule Networks **(CapsuleNet)** and attention-based models: Capture complex spatial relationships, especially for subtle or brief expressions.

The paper details several studies using combinations of CNNs and LSTMs to handle **macro-** and **micro-expressions**, noting their performance on datasets like **CASME II**, **SAMM**, and **SMIC**.

---

**Temporal modeling**  
Temporal structure is central to the recognition of **dynamic** emotions, especially subtle micro-expressions that unfold over very short durations (1/25 to 1/5 of a second).

- The paper emphasizes using **LSTM**, **GRU** (Gated Recurrent Unit), and **3D-CNNs** to capture this time-dependent nature.
- For micro-expressions, motion between **onset**, **apex**, and **offset** phases must be modeled explicitly.
- Dynamic expression recognition outperforms static frame-based methods in both accuracy and realism.

---

### Fusion Approaches  
- Late fusion (decision-level) and early fusion (feature-level).  
- Combining modalities improves robustness, but introduces complexity and challenges with synchronization and noise.

---

### Challenges Identified  
- Limited data for training, especially for subtle or spontaneous expressions.  
- Poor generalization across demographics, languages, and conditions.  
- Real-world constraints such as noisy or missing modalities.  
- Lack of standardization in evaluation protocols and benchmarks.

