
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
**Handcrafted features** refer to features that are **manually designed by us** based on domain knowledge (NOT learned automatically by a model.

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


- **Local Binary Pattern (LBP)** and **LBP-TOP (Three Orthogonal Planes)**: Common for facial micro-expression recognition, capturing texture and motion over time.
- **Histogram of Oriented Gradients (HOG)** and **SIFT**: Used to describe facial shape and edge information, mainly in macro-expressions.
- **Optical flow** and **optical strain**: Capture subtle motion between frames, important for detecting micro-expressions.
- **Limitations**: These methods rely heavily on prior knowledge, struggle with noisy or imbalanced data, and perform poorly on partial faces or datasets with large variation.


**Deep learning architectures**  
The article outlines how deep learning overcomes many of the shortcomings of handcrafted features:

- Convolutional Neural Networks **(CNNs)**: Used to extract spatial features from static facial images or motion-encoded inputs (e.g., optical flow fields).
- **3D-CNNs**: Capture both spatial and temporal features simultaneously from video data. Two-stream variants are also described for different input modalities.
- Recurrent models like **(LSTM)**: Learn the evolution of emotion over time by modeling sequences of spatial features extracted by CNNs.
- Capsule Networks **(CapsuleNet)** and attention-based models: Capture complex spatial relationships, especially for subtle or brief expressions.

The paper details several studies using combinations of CNNs and LSTMs to handle **macro-** and **micro-expressions** and notes their performance on datasets like **CASME II**, **SAMM**, and **SMIC**.

---

**Temporal modeling**  
Temporal structure is central to the recognition of **dynamic** emotions, especially subtle micro-expressions that unfold over very short durations (1/25 to 1/5 of a second).

- The paper emphasizes using **LSTM**, **GRU** (Gated Recurrent Unit), and **3D-CNNs** to capture this time-dependent nature.
- For micro-expressions, motion between **onset**, **apex**, and **offset** phases must be modeled explicitly.
- Dynamic expression recognition outperforms static frame-based methods in both accuracy and realism.

---

### Fusion Approaches  
- Late fusion (decision-level) and early fusion (feature-level).  


### Challenges Identified  
- Limited data for training, especially for subtle or spontaneous expressions.  
- Poor generalization across demographics, languages, and conditions.  
- Real-world constraints such as noisy or missing modalities.  
- Lack of standardization in evaluation protocols and benchmarks.

## Notes:

- A **micro-expression** is a very brief, involuntary facial expression that reveals a person's true emotion, even when they are trying to hide or suppress it.
- Duration: typically **1/25 to 1/5 of a second** — so fast that most people miss them in real-time.
- They often occur under high emotional pressure or when someone is being deceptive.
- Detecting them requires **high-frame-rate video** and often special algorithms to capture the tiny movements.


- THe Onset phase is when the facial muscles **start to move** from a neutral state toward forming the expression. ("build-up" or initiation).

- **Apex** is the **peak** of the expression — the point at which the emotional expression is **most intense or clearly visible**.
- It’s the split-second where the emotion is “fully exposed.”

- The **offset** phase is when the facial muscles **relax** and return to the neutral state.
- Essentially the “cool-down” or ending of the expression.


Micro-expressions are so short-lived, Thus modeling the **entire motion trajectory** (onset → apex → offset) gives a fuller picture than analyzing just a static frame. It helps capture the **temporal dynamics** of facial movement, which is critical for high-accuracy recognition — especially under subtle or ambiguous conditions.

Good question. Micro-expressions are tricky — they’re subtle, fast, and easy to miss. So, the models and methods used to detect them have to be *both sensitive* and *temporally aware*. Here's a breakdown of what's typically used:

---

### Classical Approaches (Pre-DL)
These often rely on **optical flow** or **LBP-TOP**:

- **Optical Flow**  
  Captures motion between video frames. For micro-expressions, you track tiny muscle movements frame-by-frame.
  - Common method: **TV-L1 Optical Flow**
  - Pro: interpretable, works with small data  
  - Con: struggles with noise, low discriminative power  

- **LBP-TOP (Local Binary Patterns - Three Orthogonal Planes)**  
  Hand-crafted spatiotemporal features from facial regions in X-Y, X-T, Y-T slices.
  - Pro: lightweight, classic baseline
  - Con: needs precise alignment, loses subtle dynamics


### Deep Learning Models

#### 1. **3D CNNs (Convolutional Neural Networks)**
- Instead of processing frame-by-frame (2D), 3D CNNs learn spatial and temporal features jointly.
- Can directly learn patterns across onset → apex → offset.
- Example: **C3D**, **I3D** (Inflated 3D)

#### 2. **Two-stream Networks**
- One stream gets the raw video; the other gets optical flow.
- Outputs are fused to capture both appearance and motion.
- Example: Similar setup to what’s used in video action recognition.

#### 3. **Recurrent Models**
- Use **RNNs** or **LSTM**s to capture the time sequence of expressions.
- Often used after CNN feature extraction.
- Especially useful when expressions unfold over time (like micro-expressions).

#### 4. **Temporal Attention Models**
- These learn to focus more on important frames (like the apex).
- Example: **STSTNet** (SpatioTemporal Synergistic Network)

#### 5. **Facial Action Unit (AU)-based Methods**
- Detect specific muscle movements (e.g., AU12 = lip corner puller).
- Tools like **OpenFace** can extract AU activations over time, and classifiers are trained on their dynamics.


- Datasets commonly used **CASME II**, **SAMM**, **SMIC** — all recorded with **high frame rate cameras** (100–200 fps), with precise onset/apex/offset annotation.
