### Summary: Comparison of Emotion Recognition Sensors (Expanded)

Sensors used in emotion recognition are categorized into five main groups:

---

#### **1. Visual Sensors**
Primarily **cameras** used for:
- **Facial Expression Recognition (FER)**
- **Remote Photoplethysmography (rPPG)** — (using RGB cameras to detect subtle changes in skin color due to blood flow).

**Pros:**
- Low cost.
- Easy and non-invasive data collection.
- Widely supported by existing computer vision frameworks.

**Cons:**
- Strongly affected by lighting conditions.
- Privacy concerns (video data).
- Faces can be manipulated.
- High inter-subject variability (skin color, facial structure).

---

#### **2. Audio Sensors**
These involve **microphones** used in **Speech Emotion Recognition (SER)**.

**Pros:**
- Low cost and widely available.
- Can capture content (what is said) and paralinguistic features (how it's said — tone, pitch, energy).

**Cons:**
- Performance drops in noisy or multi-speaker environments.
- Language- and culture-dependent.
- Emotions like "neutral" or "bored" can be hard to distinguish from others.

---

#### **3. Radar Sensors**
These use **radio-frequency (RF) signals** to detect physiological changes like:
- **Heart rate**.
- **Respiratory patterns**.
- Micro-movements of the chest (non-contact).

**Pros:**
- Not affected by lighting.
- Allows for remote and passive monitoring (no need for wearables).

**Cons:**
- Susceptible to Doppler distortions (especially if the person moves).
- Noise interference in cluttered or dynamic environments.
- Less explored compared to visual/audio options.

---

#### **4. Other Physiological Sensors**
wearables or direct-contact sensors such as:
- **Electroencephalography (EEG)** -brain activity.  
- **Electrocardiography (ECG)** -heart activity.  
- **Electromyography (EMG)** -muscle activation. 
- **Galvanic Skin Response (GSR)** -sweat level  
- **Blood Volume Pulse (BVP)**  
- **Electrooculography (EOG)** -eye movements  

**Pros:**
- Capture involuntary, physiological signals less likely to be consciously controlled.
- Strong correlation with internal emotional states.

**Cons:**
- Invasive and uncomfortable (need to be worn or attached to skin).
- Might introduce bias (wearing sensors may affect the subject’s emotional state).
- Often less acceptable for use in daily-life applications.

---

#### **5. Multi-Sensor Fusion**
Combining two or more of the above sensor types.

**Pros:**
- Increases robustness and accuracy.
- Helps compensate for the limitations of single modalities.

**Cons:**
- Synchronization challenges.
- Increased computational load and system complexity.
- Requires more complex dataset design and annotation.

---



**“Multimodal Sentiment Analysis Using Hierarchical Fusion with Context Modeling”** (Majumder et al., 2018).

#### Problem:
- Sentiment analysis using **just text**, **just audio**, or **just video** often misses the full picture.
- Simple **feature concatenation** across modalities leads to *noisy*, hard-to-train models with redundant or misaligned information.

#### Proposed Solution:
- Use a **hierarchical fusion strategy**:
  - First combine modalities **pairwise** (bimodal fusion).
  - Then fuse the **bimodal representations** into a final trimodal one.
- Also introduce **context modeling**:
  - Each utterance is not analyzed in isolation.
  - Use RNNs (specifically **GRUs**) to understand how utterances relate to one another in a sequence (e.g., in a video or dialogue).


- Multimodal sentiment analysis on **utterances in videos**, particularly using datasets where each spoken segment is labeled for sentiment.
---
