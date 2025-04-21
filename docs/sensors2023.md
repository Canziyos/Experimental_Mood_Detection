## Emotion Recognition Using Different Sensors, Emotion Models, Methods and Datasets: A Comprehensive Review

### 0. Introduction
- Advances in sensors and information technology (hot topic).

- Application areas: HCI, healthcare, education, security, driving, psychology, and helping people who can't express emotions clearly.

### 0.1 Input Types
- Behavioral cues: facial expressions, speech, gestures – external and visible.

- Physiological signals: HR, respiration, EEG, ECG – internal emotional indicators.

- Multimodal fusion: combining several signals to improve recognition accuracy.

### 0.2 Purpose of the paper
- Compare and review 200+ papers on emotion recognition.

- Focus on sensors, models, methods, and datasets.

- Help researchers choose the right tools for their needs.


## 1. Emotion Models

Before a machine can recognize emotions, it needs a model — a framework to define what emotions are and how they are structured or categorized.

Two Main Types of Emotion Models

### 1.1 Discrete Model

- Emotions are viewed as distinct categories  
   (like a list: happiness, sadness, anger, fear, disgust, surprise, etc).

- Based on Darwinian evolution: emotions are basic, hardwired responses.

- Famous theories:
   - **Ekman’s Basic Emotions** (6 or 7 basic categories)
   - **Plutchik’s Wheel** — expands the emotion list and visualizes intensity and relationships.

- Easy to label and classify.

- Doesn’t capture mixed or complex emotions.

### 1.2 Dimensional Model

- Emotions are not fixed categories but **positions in a continuous space** (vectors).

- Two common dimensional frameworks:
   - **Valence–Arousal Model**
     - *Valence*: how positive or negative the emotion is.
     - *Arousal*: how intense or activating the emotion is.
     - *Example*: anger and fear both have high arousal and low valence — yet feel different.
   - **PAD Model (Pleasure–Arousal–Dominance)**
     - Adds a third axis: *Dominance* — how controlling or submissive the emotion feels.

- Good for expressing complex or blended emotional states.

- May fail to distinguish between some basic or distinct emotions.
---

## 3. Sensors for Emotion Recognition

Emotion recognition uses different types of sensors to capture either external behavior or internal physiological signals.

Each has strengths and weaknesses depending on signal type, user comfort, and environmental conditions.


### 3.1 Sensor Categories and Characteristics

#### 3.1.1 Visual Sensors

- Used for Facial Expression Recognition (FER) and remote photoplethysmography (rPPG).
- rPPG tracks heart rate via skin color changes using RGB cameras.
- FER pipeline: face detection → feature extraction → emotion classification.
- Notes:
  - Affected by lighting conditions, occlusions (e.g. masks), and camera angles.
  - Facial expressions can be faked or suppressed.
  - Individual differences in skin tone and expression intensity reduce classification accuracy.

#### 3.1.2 Audio Sensors

- Basis for Speech Emotion Recognition (SER).
- Uses acoustic features and semantic language features.
- Common applications: call centers, virtual assistants, autism support.
- Notes:
  - Sensitive to speaker variability, accent, and cultural factors.
  - One sentence can contain multiple or ambiguous emotional signals.

#### 3.1.3 Radar Sensors

- Enables non-contact monitoring of vital signs (heart rate, respiration).
- Captures micro-movements of the chest using radar echo signals.
- Works well in low-light or privacy-sensitive settings.
- Notes:
  - Prone to noise from movement along the radar axis (toward/away).
  - Motion interference affects sentiment classification accuracy.

#### 3.1.4 Other Physiological Sensors

- Includes EEG, ECG, EMG, GSR, BVP, EOG.
- These detect involuntary physiological signals linked to emotions.
- Notes:
  - Require skin contact or wearables — possibly uncomfortable or invasive.
  - Devices may increase stress or awareness, which can bias signals.

#### 3.1.5 Multi-Sensor Fusion

- Combines multiple sensors.
- Three main fusion strategies:
  1. Pixel-level fusion – merges raw data (but may combine noise).
  2. Feature-level fusion – combines extracted features before classification.
  3. Decision-level fusion – each sensor outputs a decision, then all are fused.

- Notes:
  - Requires synchronization across sensors.
  - Increases computational complexity.
  - Needs larger, well-structured multimodal datasets.


## 4. Emotion Recognition Method

This section outlines the main pipeline for recognizing emotions from sensor data. The process is split into several stages:

### 4.1 Signal Preprocessing
- Improves signal quality.
- Reduces noise and irrelevant artifacts.

**Preprocessing Techniques:**

#### For Visual Signals:
- Cropping, rotating, scaling, grayscaling.
- Used to normalize image inputs and improve robustness.

#### For Audio Signals:
- Silent frame removal
- Pre-emphasis (boosting high-frequency components)
- Normalization
- Windowing (to preserve time-domain continuity)
- Noise reduction (e.g., MMSE – Minimum Mean Square Error)

#### For Radar/Physiological Signals:
- Filtering (removes baseline drift, crosstalk)
- Wavelet transforms (captures both time and frequency info)
- Nonlinear dynamics (e.g., entropy measures to handle irregular fluctuations)

### 4.2 Feature Extraction
- Transforms raw signals into compact, informative features.
- Reduces computation and improves classification accuracy.

#### For Visual Signals:
- PCA (Principal Component Analysis): dimensionality reduction.
- 2DPCA and Bidirectional PCA: better at handling image matrices.
- HOG (Histogram of Oriented Gradients): edge-based, robust to lighting changes.
- LBP, LDA: other common options.

#### For Speech Signals:
- LPC (Linear Prediction Coefficients): models vocal tract.
- TEO (Teager Energy Operator): measures energy for stressed speech.
- MFCC, FFT, LDA: also common.

#### For Physiological/Radar Signals:
- FFT: transforms signals to frequency domain.
- mRMR: selects features with high relevance and low redundancy.
- EMD, LDA, Relief-F: other feature selectors.

### 4.3 Feature Selection
- Optional step to pick only the most relevant features for classification.
- Helps prevent overfitting.

### 4.4 Classification
- Applies machine learning (ML) or deep learning (DL) algorithms to assign an emotion label.

### 4.5 Validation
- Tests the model using evaluation metrics to measure accuracy or generalization.
---

## 5. Classification

Emotion classification assigns a label to the input signal using either classical machine learning (ML) or deep learning (DL) models. This section is divided into:

### 5.1 Machine Learning Methods

Classical ML methods work with manually extracted features and are computationally efficient.

#### 5.1.1 Support Vector Machine (SVM)
- Finds the optimal hyperplane that maximizes the margin between emotion classes.
- Handles non-linear data using kernel tricks.
- Prevents overfitting with soft margins and regularization.
- **Reported performance**:
  - 93.75% on Berlin Emotion speech dataset.
  - 91.95% FER accuracy on CK+ dataset.

#### 5.1.2 Gaussian Mixture Model (GMM)
- Models data as a mixture of Gaussian distributions.
- Learns parameters using the Expectation-Maximization (EM) algorithm:
  - E-step: estimate latent variables.
  - M-step: maximize likelihood.
- Applied in speech emotion recognition and privacy-preserving facial analysis.

#### 5.1.3 Hidden Markov Model (HMM)
- Probabilistic model for temporal/sequential data.
- Uses:
  - State transition probabilities `a_ij`
  - Emission probabilities `b_ij`
  - Initial state probabilities `π_i`
- Models the dynamics of emotional transitions over time.
- Applied in SER, FER, audiovisual models, and sentiment detection.

#### 5.1.4 Random Forest (RF)
- Ensemble of decision trees using Bagging and random feature selection.
- Robust to noise, missing data, and overfitting.
- Uses majority voting for prediction.
- Applied in:
  - SER with fuzzy logic
  - Physiological ER (HR + GSR)
  - Anxiety intensity classification (up to 80.83% accuracy)

---

### 5.2 Deep Learning Methods

DL models learn both features and classifiers jointly. Suitable for complex, high-dimensional emotion data.

#### 5.2.1 Convolutional Neural Network (CNN)
- Learns spatial features using convolution and pooling layers.
- Network structure: Convolution → Pooling → Fully Connected → Output.
- Used in:
  - Facial expression recognition (FER)
  - EEG-based ER
  - Multimodal fusion (face video + EEG)

#### 5.2.2 Long Short-Term Memory (LSTM)
- A recurrent network designed for sequential data.
- Contains input, forget, and output gates controlling:
  - Cell state `c_t`
  - Hidden state `h_t`
- Captures long-range dependencies in speech and physiological signals.
- **Often combined with CNN or attention mechanisms**:
  - CNN-LSTM
  - Bi-LSTM with Directional Self-Attention

#### 5.2.3 Deep Belief Network (DBN)
- Built from stacked Restricted Boltzmann Machines (RBMs).
- Trained via unsupervised pre-training followed by fine-tuning.
- Learns high-order feature representations layer-wise.
- Applied in:
  - EEG-based emotion recognition
  - Multimodal ER with joint feature learning

---
- An **RBM** is shallow (just one visible + one hidden layer).

### Can RBM or DBN Be Used Alone for Emotion Recognition?
- **Single RBM**: can be used for **unsupervised feature extraction**, but not ideal for classification.
- **DBN**: can be used as a **standalone ER model**, especially in **biosignal-based systems (e.g., EEG)**, **where data is scarce or noisy**.

- Often, DBNs are combined with other classifiers (e.g., **Softmax, SVM**) or used as **pretraining blocks**.

- RBMs are shallow, but stacking them forms a DBN. A DBN can work as a full ER model in some cases, but it's often more effective as part of a hybrid or layered system.
---

## 6. Datasets

This section reviews main datasets used in emotion recognition (ER), categorized by sensor type. A major challenge is the lack of large, diverse, and standardized datasets, especially for multimodal ER.

### 6.1 Visual Datasets

Used primarily for FER.

- **CK+**: Extended Cohn-Kanade; posed expressions; 123 subjects.
- **JAFFE**: Japanese female facial expressions; 213 grayscale images.
- **FER2013**: In-the-wild dataset; 35,887 images; 7 emotion labels.
- **AffectNet**: Over 1 million images; labeled for valence and arousal.
- **Affectiva-MIT**: Naturalistic video dataset of facial reactions.
- **EmotiW**: Annual competition dataset focused on real-world FER.

Notes:
- Many datasets contain posed expressions.
- Diversity in age, ethnicity, and spontaneous behavior is limited.


### 6.2 Audio Datasets

Used in SER.

- **Berlin EMO-DB**: Acted speech in German; 10 speakers; 7 emotions.
- **RAVDESS**: English; speech and singing; 24 actors.
- **IEMOCAP**: Audio + video + transcripts; 10 actors; scripted + improvised speech.
- **CREMA-D**: 91 actors; validated crowd-sourced emotion labels.

Notes:
- Most datasets involve acted emotions rather than natural speech.
- Cultural and linguistic bias is common.


### 6.3 Radar-Based Datasets

Used for non-contact sensing of emotional state via micro-movements.

- use Doppler or FMCW radar.
- Datasets in this domain are still limited and often private or unpublished.

Notes:
- Lack of public radar-based datasets.
- Small sample sizes and poor standardization.

### 6.4 Physiological-Based Datasets

Focus on signals like EEG, ECG, EMG, GSR, BVP, etc.

- **DEAP**: 32 participants; EEG + peripheral signals; valence/arousal ratings.
- **SEED**: 15 participants; EEG during emotional video clips.
- **MAHNOB-HCI**: EEG, ECG, GSR, facial videos, and audio; 27 subjects.
- **DREAMER**: Portable EEG + ECG; 23 participants; emotional film stimuli.

Notes:
- Limited number of participants.
- Often involve invasive or inconvenient sensors.
- No universal labeling standards.


### 6.5 Multimodal Datasets

Combine multiple sensor modalities.

- **IEMOCAP**: Audio + video + text data.
- **MAHNOB-HCI**: EEG + facial video + gaze + physiological data.
- **DEAP**: Multimodal; EEG + physiological + video stimuli.

Notes:
- Offer rich data but are complex to use.
- Require time-aligned fusion and heavy preprocessing.
- High storage and computational needs.

## 7. Discussion and Challenges

Key limitations and open challenges in deploying emotion recognition (ER) in real-world settings.

### 7.1 Current Limitations

- **Datasets**: Mostly lab-based; need spontaneous, diverse data.
- **Labeling**: Emotions are subjective; annotation is inconsistent.
- **Sensors**: Physiological sensors are intrusive; visual/audio are noise-sensitive.
- **Fusion**: Requires sync, alignment, and increases system complexity.
- **Generalization**: Models often overfit; poor cross-dataset performance.


### 7.2 Research Challenges

- Build robust, real-time, and cross-cultural ER systems.
- Use non-intrusive sensors suitable for daily life.
- Develop adaptive, lightweight fusion strategies.
- Tackle ethical issues: privacy, bias, and transparency.

**Note**: No single method works for all. Future ER needs hybrid, personalized approaches trained on realistic data.

---
## 8. Conclusion

Sensor-based emotion recognition (ER) holds great potential across domains like healthcare, education, and HCI.

Key points:
- Different sensors offer trade-offs: visual/audio are non-invasive but noisy, physiological are accurate but intrusive.
- Multimodal fusion is promising but complex to implement.
- Many current systems lack real-world robustness.

Future directions:
- Develop real-time, unobtrusive, and reliable ER systems.
- Address ethical issues: privacy, fairness, and consent.
- Use cross-disciplinary approaches combining ML, signal processing, and psychology.
