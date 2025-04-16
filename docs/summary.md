### Emotion Models

Emotion models structure how we represent, detect, and classify emotions. The reviewed article compares two main families of models.

---

#### 1. Discrete Emotion Models

These models assume that emotions can be broken into a finite set of basic categories. This view is rooted in evolutionary psychology.

**Examples:**
- Ekman’s model: Proposes six or seven universal emotions.  
  Happiness, sadness, anger, fear, surprise, disgust, and sometimes contempt.
- Plutchik’s wheel of emotions: Defines eight basic emotions and their intensities, arranged in opposing pairs.  
  Joy vs. sadness, anger vs. fear, trust vs. disgust, surprise vs. anticipation.

**Advantages:**
- Easy to interpret and label.
- Matches common-sense emotional categories.
- Compatible with many labeled datasets (e.g., FER2013, Emo-DB).

**Disadvantages:**
- Too rigid for complex emotional states.
- Poor at capturing subtle or mixed emotions (e.g., bittersweet, envy, nostalgia).
- Culture-dependent to some degree; not all facial expressions are interpreted the same globally.

Very useful for classification tasks such as facial expression recognition, but oversimplified in human psychology terms.

**As presented in the article:**
- Based on evolutionary theory (Darwin, Ekman).
- Emotions are treated as primitive reactions with fixed categories.
- Emphasizes Ekman's criteria (e.g., rapid onset, short duration, universal expression).
- Highlights Plutchik’s model for its layered emotional structure and intensity levels.
- Favored by the authors in applications involving facial expressions or speech, especially when using labeled datasets.

---

#### 2. Dimensional Emotion Models

These models represent emotions as coordinates in a continuous emotional space, usually with two or three dimensions.

**Main examples:**
- Valence-Arousal model:
  - Valence refers to the pleasantness or unpleasantness of the emotion.
  - Arousal refers to the level of activation or intensity.
- PAD model (Pleasure–Arousal–Dominance):
  - Adds a third axis, dominance, which reflects the degree of control or power a person feels.

**Advantages:**
- Captures emotional intensity and subtleties.
- Better suited for continuous or blended emotional states.
- Useful for regression-based approaches and real-time tracking.

**Disadvantages:**
- Harder to label and interpret (e.g., what does [0.3, 0.7, −0.2] represent?).
- Less intuitive for users without technical background.
- Mapping between dimensional points and categorical labels is often ambiguous.
 
Ideal for modeling emotion dynamics in video, audio, or physiological data streams.

**As presented in the article:**
- Described as more flexible than discrete models.
- Focuses on Valence-Arousal (2D) and PAD (3D) models.
- Notes that anger and fear can share similar coordinates in 2D, which make them difficult to distinguish.
- States that the dominance dimension in PAD helps reduce such ambiguity.
- Favored in contexts involving physiological data and continuous monitoring.

---

#### Critical Point in the Article

"Dimensional emotion models can accurately identify the core emotion. However, for some complex emotions, they will lose some details."

This statement appears inconsistent. It is arguably the discrete models that are more prone to oversimplification, whereas dimensional models are usually better suited to capturing emotional nuance..


---

### Comparison of Emotion Recognition Sensors 

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
- Electroencephalography **(EEG)** -brain activity.  
- Electrocardiography **(ECG)** -heart activity.  
- Electromyography **(EMG)** -muscle activation. 
- Galvanic Skin Response **(GSR)** -sweat level  
- Blood Volume Pulse **(BVP)**  
- Electrooculography **(EOG)** -eye movements  

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

### Fusion Strategies in Emotion Recognition

The reviewed paper identifies three common levels of fusion.

---

#### **1. Pixel-level fusion**

This strategy merges raw input data from different sensors before any feature extraction.

**Example:** Combining EEG and GSR signals directly into a multichannel array.

**Advantages:**
- Preserves complete information from all modalities.
- Allows early-stage correlation learning.

**Disadvantages:**
- Sensitive to noise and signal misalignment.
- Requires identical sampling rates.
- Computationally expensive.

Rarely used in practice except for naturally aligned signals (e.g., RGB-D video).

---

#### **2. Feature-level fusion**

Features are first extracted separately from each sensor, then concatenated into one vector before classification.

**Example:** Concatenating Mel-Frequency Cepstral Coefficient (MFCC) features from audio with Histogram of Oriented Gradients (HOG) features from video.

**Advantages:**
- Balances richness and efficiency.
- Retains important traits while reducing raw noise.
- Works well with asynchronous data (after alignment).

**Disadvantages:**
- Requires careful timing alignment between modalities
- Vulnerable to degraded or missing features from any modality.

Most popular in academic research due to its flexibility and performance.

---

#### **3. Decision-level fusion**

Each sensor is processed independently up to the final prediction. The individual decisions are then merged.

**Example:** Using majority voting between visual, audio, and EEG emotion classifiers.

**Advantages:**
- Robust to failure or noise in individual modalities.
- Modular and easy to implement.
- Can use specialized models per sensor type.

**Disadvantages:**
- May miss cross-modal sign/indications.
- Risk of contradictory outputs between modalities.

Common in real-world systems, especially where sensors operate independently.

---

Great. Let's look at **point 5: Datasets** as described in the reviewed article.

---

### Point 5: Datasets for Emotion Recognition

The article categorizes datasets based on the **sensor modality** they support. It covers a wide selection, which is crucial because emotion recognition systems are heavily dependent on **data diversity**, **labeling quality**, and **sensor synchronization**.

---

#### **1. Visual Datasets**

Used primarily for **Facial Expression Recognition (FER)**.

**Examples:**
- **CK+ (Extended Cohn-Kanade)** — posed expressions with temporal sequences.
- **JAFFE (Japanese Female Facial Expression)** — grayscale images with discrete emotion labels.
- **RaFD (Radboud Faces Database)** — high-resolution faces with eye-gaze variations.
- **FER2013** — in-the-wild dataset from Kaggle with noisy and spontaneous expressions.

> The article emphasizes that visual datasets are often **limited by demographic bias**, expression subtlety, and controlled environments.

---

#### **2. Audio Datasets**

Used for **Speech Emotion Recognition (SER)**.

**Examples:**
- **Emo-DB** — acted German speech with 7 emotion labels.
- **RAVDESS** — multimodal audiovisual dataset with professional actors.
- **IEMOCAP (Interactive Emotional Dyadic Motion Capture)** — rich set with audio, video, motion capture, and natural dialog.

> The paper highlights issues like **language-dependence** and **acting bias**, which reduce generalization across cultures or real-world speech.

---

#### **3. Physiological Signal Datasets**

Often used in health-related or stress/mood studies.

**Examples:**
- **DEAP (Dataset for Emotion Analysis using EEG, Physiological Signals, and Video Stimuli)** — EEG, GSR, and facial videos with arousal–valence ratings.
- **SEED (SJTU Emotion EEG Dataset)** — Chinese subjects watching emotional film clips, used for EEG-based emotion classification.
- **DREAMER** — audio-visual elicited emotions with EEG and ECG signals.

> Most physiological datasets are **small** in subject count and suffer from **inter-subject variability**.

---

#### **4. Multimodal Datasets**

Designed for **sensor fusion** research.

**Examples:**
- **MAHNOB-HCI** — combines EEG, ECG, GSR, and video with emotion annotations.
- **AMIGOS** — group and individual affect recognition from EEG, ECG, and face/body video.
- **SAVEE** — British male speakers with audiovisual and lip movement data.

> These datasets are rare, expensive to collect, and often not perfectly synchronized. But they are **essential** for testing real-world, robust systems.

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
