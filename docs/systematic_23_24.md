## Emotion recognition and artificial intelligence: A systematic review (2014–2023)

## 1. Introduction

### Motivation
Emotion recognition (ER) is essential for improving interactions in:
- Human–machine communication.
- Healthcare and well-being monitoring.
- Affective computing.
- Education and marketing.

Understanding emotion enables systems to adapt to human needs more effectively.

### Terminology Clarification

The paper distinguishes between:

| Term                 | Description                                              |
|----------------------|----------------------------------------------------------|
| **Affect**           | Basic, short-lived physiological responses (e.g., arousal) |
| **Emotion**          | Context-specific, relatively brief episodes (e.g., fear)  |
| **Feeling**          | Subjective awareness of emotion ("I feel anxious")        |
| **Mood**             | Longer-lasting, diffuse emotional state (e.g., irritable) |
| **Sentiment**        | Linguistic or text-based attitude (e.g., positive review) |
| **Emotion Dimensions** | Continuous axes (valence, arousal, dominance)            |


- Clarifies foundational definitions in the field.
- Reviews methods for:
  - Emotion elicitation and data collection
  - Modeling and recognition algorithms
  - Modalities (visual, audio, physiological)
  - Multimodal fusion strategies
- Highlights trends like:
  - Transformer-based architectures
  - Self-supervised learning
  - Multimodal learning challenges


---
## 2. Emotion Models and Dimensions

This section revisits emotion modeling but introduces several extensions beyond what was covered in the [paper summary](sensors2023.md).

### Basic vs. Complex Emotions

Basic emotions (e.g., Ekman’s model) are considered biologically driven and universal.  
Complex emotions (e.g., guilt, pride, jealousy) are shaped by social and cultural context.  
These are harder to classify but may be important in advanced ER systems.

- Cultural Variability

Emotion expression and recognition vary across cultures.  
Some expressions may be misinterpreted or underrepresented in datasets that are culture-biased.  
This highlights the importance of culturally diverse training data and adaptable models.

- Hybrid Emotion Models
Some systems combine discrete emotion labels (like Ekman’s) with dimensional scores (valence, arousal).  
This hybrid labeling helps to capture more nuanced or mixed emotional states.  

- Extended Dimensional Models
Beyond PAD (Pleasure–Arousal–Dominance), other models like the Geneva Emotion Wheel use 14 or more dimensions.  
These are rarely used in AI due to annotation difficulty but may offer richer emotional granularity.  

---

## 3. Emotion Elicitation and Labeling

How emotional states are provoked and labeled during dataset collection.

### 3.1 Emotion Elicitation Methods

1. **Passive Elicitation**
   - Emotion evoked via external stimuli (videos, images, music).
   - Easy to implement but may trigger weaker reactions.

2. **Active Elicitation**
   - Requires subject interaction (e.g., storytelling, role-play, VR, games).
   - Yields stronger and more realistic emotional expressions.

3. **Naturalistic Capture**
   - Records emotions during real-life, spontaneous activity.
   - Highest realism but lowest experimental control and hardest to annotate.


### 3.2 Labeling Strategies

- **Self-Annotation**
  - Subjects rate their own emotional state.
  - Pros: subjective insight; Cons: delay, memory distortion.

- **External Annotation**
  - Observers label based on visible/audio cues or coded systems (e.g., FACS).
  - Pros: scalable; Cons: observer bias, limited internal access.

**Note:**  
Regarding "coded systems (e.g., FACS)," [see AUs in foundations file](Foundations.md).
- FACS was originally designed for trained human coders to label facial muscle movements using Action Units (AUs).

- Today, many ER systems use automated FACS tools (e.g., OpenFace, Affectiva SDK, iMotions, FaceReader) to detect facial landmarks, extract AUs, and sometimes map them to emotions (mapping is a separate step).


- **Crowdsourced Labeling**
  - Multiple non-expert raters label samples.
  - Used in large-scale datasets like AffectNet.

- **Physiological Grounding**
  - Labels inferred from biosignals (e.g., HRV, EEG).
  - Still experimental and not widely adopted.

### Notes

- No universal labeling standard exists.
- Trade-off: ecological validity ↔ experimental control.
---
## 4. Emotion Modalities

Input signals are grouped into physical, physiological, and multimodal categories.

### 4.1 Physical Modalities

These are observable behaviors.

#### Facial Expression
- Most widely used in visual emotion recognition.
- Analyzed using landmarks, FACS, or deep learning.
- Can be faked or masked; sensitive to lighting, occlusion, and angle.

#### Speech and Voice
- Uses acoustic features (pitch, tone, speed) and sometimes language content.
- Rich in emotion, but affected by noise, language, and speaker traits.

#### Body Gesture and Posture
- Captures emotion from pose, motion, or kinetic patterns.
- Less studied but useful in embodied systems like robots or games.

#### Textual Input
- Extracts emotion from written language (e.g., sentiment analysis).
- Used in chatbots, social media, and NLP.
- Lacks physical cues; needs contextual understanding.

### 4.2 Physiological Modalities

Measure internal, involuntary bodily responses.

- EEG (Electroencephalography): brainwave signals; high temporal resolution but noisy and sensor-intensive.

- ECG (Electrocardiography): Tracks heart electrical activity; reflects arousal and stress levels.

- GSR/EDA (Galvanic Skin Response): Measures skin conductivity linked to emotional arousal. Simple and low-cost.

- EMG (Electromyography): Detects muscle activity (e.g., subtle facial twitches). High accuracy but requires direct contact.

- Other Signals: Blood Volume Pulse (BVP), respiration, eye tracking, pupil dilation, etc.

### 4.3 Multimodal Input

- Combines two or more modalities.
- Common combinations: face + voice, EEG + video, text + audio.
- Challenges: Time synchronization, feature alignment, fusion strategy.

---

## 5. Emotion Datasets

This section reviews commonly used datasets for emotion recognition, categorized by modality.

---

### 5.1 Visual and Audio-Visual Datasets

- **CK+**: Posed facial expressions; 123 participants.
- **JAFFE**: Japanese female facial expressions; grayscale images.
- **FER2013**: Large-scale, in-the-wild facial dataset; 35k+ images.
- **AffectNet**: 1M+ images; labeled with valence/arousal.
- **IEMOCAP**: Audio, video, and text; acted and improvised dialogue.
- **RAVDESS**: Speech and singing samples; 24 actors.
- **CREMA-D**: Speech recordings; emotion labels from crowdsourcing.
- **MELD**: Multispeaker conversations from TV show; multimodal labels.
- **EmotiW (AFEW)**: Video clips from movies; used in annual challenge.

### 5.2 Physiological Datasets

- **DEAP**: EEG + peripheral signals; 32 subjects; music video stimuli.
- **DREAMER**: EEG + ECG; 23 participants; video-based elicitation.
- **SEED**: EEG during emotional film viewing; session-wise data.
- **MAHNOB-HCI**: EEG, ECG, GSR, video, and audio; 27 participants.
- **ASCERTAIN**: EEG + personality and emotion ratings.

### 5.3 Multimodal Datasets

- **IEMOCAP**: Audio, video, text, motion capture; detailed labels.
- **SEMAINE**: Audiovisual emotional interaction recordings.
- **RECOLA**: Continuous emotion annotations; multimodal input.
- **MOSI / MOSEI**: Text, audio, video; sentiment and emotion tagging.
- **SEED-IV**: EEG-based with more emotion categories than SEED.

### Dataset Challenges

- Most datasets contain **acted emotions**, not spontaneous.
- **Cultural/language bias** limits generalization.
- **Small sample sizes** in physiological studies.
- **Labeling inconsistency** across datasets.
- **Limited diversity** in gender, age, and ethnicity.
----

## 6. AI Techniques for Emotion Recognition

This section presents the AI methods used to detect and classify emotions across various input modalities.

### 6.1 Traditional ML Techniques

Used mostly with hand-crafted features, especially in physiological and audio-based ER.

#### Algorithms:
- Support Vector Machines (SVM).
- k-Nearest Neighbors (k-NN).
- Naive Bayes.
- Decision Trees (DT).
- Random Forests (RF).
- AdaBoost.
- Linear Discriminant Analysis (LDA).
- Gaussian Mixture Models (GMM).
- Hidden Markov Models (HMM).
- Multilayer Perceptrons (MLPs).

**Pros**:
- Efficient on small datasets.
- Interpretable and fast to train.


### 6.2 DL Techniques

Used for larger datasets and automatic feature learning.

#### Architectures:
- **Convolutional Neural Networks (CNN)**
  - Effective for image-based and EEG spectrogram data.

- **Recurrent Neural Networks (RNN)**
  - Suitable for time-series data (e.g., speech, biosignals).

- **Long Short-Term Memory (LSTM)**
  - Captures long-range dependencies in sequential data.

- **Gated Recurrent Units (GRU)**
  - Lightweight alternative to LSTM; used in audio/video ER.

- **Deep Belief Networks (DBN)**
  - Composed of stacked Restricted Boltzmann Machines; used in EEG-based ER.

- **Autoencoders / Variational Autoencoders (VAE)**
  - Learn compressed representations in unsupervised settings.

- **Transformers**
  - Emerging model; excels in capturing global dependencies.
  - Applied to video, audio, text, and multimodal ER tasks.

### 6.3 Emerging and Hybrid Approaches

- **Self-Supervised Learning (SSL)**
  - Learns useful features without labeled data (e.g., contrastive learning).

- **Transfer Learning**
  - Fine-tuning pretrained models like ResNet, VGG, or BERT on ER datasets.

- **Multimodal Fusion Networks**
  - Combine multiple input modalities using:
    - Early fusion (feature-level)
    - Late fusion (decision-level)
    - Hybrid fusion
  - Often include attention mechanisms to weight inputs dynamically.

- **Graph Neural Networks (GNNs)**
  - Used for spatial structure modeling (e.g., facial landmark graphs).

- **Capsule Networks (CapsNets)**
  - Experimental models for preserving spatial hierarchies in image-based FER.

----
## 7. Multimodal Fusion Techniques

### 7.1 Fusion Strategies

#### Early, Feature-Level, Fusion
- Features from each modality are extracted and concatenated before classification.
- Simple to implement and allows modeling of cross-modal interactions early.
- Drawbacks:
  - Sensitive to dimensional imbalance and noise.
  - Requires modalities to be temporally aligned and preprocessed consistently.

#### Late, Decision-Level, Fusion
- Each modality is processed separately with its own classifier.
- Individual predictions are combined via:
  - Majority voting
  - Weighted averaging
  - Stacked meta-classifier (e.g., SVM, MLP)

#### Hybrid Fusion
- Combines early and late fusion in multi-stage architectures.
- Maintains separate processing streams, followed by intermediate fusion layers or decision refinements.


### 7.2 Fusion Architectures and Tools

#### Attention Mechanisms
- Dynamically weight feature importance across modalities or time steps.
- Examples:
  - Cross-modal attention
  - Self-attention for intra-modal refinement

#### Multimodal Transformers
- Use self-attention and cross-attention to model interdependencies between modalities.
- Key models:
  - **MulT**: Multimodal Transformer with modality-specific and cross-modal attention.
  - **TFN (Tensor Fusion Network)**: Models all unimodal, bimodal, and trimodal interactions explicitly using tensor outer product.
  - **MISA**: Models intra- and inter-modality dynamics using shared-private architecture.
  - **MAG-BERT**: Multimodal adaptation gate integrated with BERT for fine-grained fusion of text + non-text signals.

#### Other Fusion Tools/Frameworks
- **CMU Multimodal SDK**: Used in MOSI/MOSEI; supports multimodal fusion workflows.
- **OpenFace + OpenSMILE**: For facial and audio feature extraction before fusion.
- **Fusion with Graph Neural Networks (GNNs)**: Emerging use in spatial–temporal feature fusion.

### Fusion Challenges

- **Temporal synchronization**: Misaligned modality timestamps degrade performance.
- **Feature heterogeneity**: Different dimensionality and sampling rates require careful preprocessing.
- **Missing modality handling**:
  - Dropout simulation.
  - Imputation methods.
  - Modality-invariant representations.
- **Compute cost**: Multimodal transformers and hybrid systems are resource-intensive.
----


## 8. Evaluation Metrics

Depends on the task type and emotion representation (categorical or dimensional).

### 8.1 Classification Metrics (Discrete Emotions)

- **Accuracy**: 
  - Proportion of correctly predicted samples.
  - Can be misleading in imbalanced datasets.

- **Precision**: 
  - TP / (TP + FP)
  - Measures correctness among predicted positives.

- **Recall**: 
  - TP / (TP + FN)
  - Measures how many actual positives were correctly identified.

- **F1 Score**: 
  - Harmonic mean of precision and recall.

- **Confusion Matrix**:
  - Visual summary of prediction vs. true labels across all classes.

### 8.2 Regression Metrics (Continuous Dimensions)

Used for valence, arousal, or dominance scores.

- **Mean Squared Error (MSE)**:
  - Penalizes large errors more heavily.

- **Mean Absolute Error (MAE)**:
  - Measures average magnitude of errors.

- **Pearson Correlation Coefficient (PCC)**:
  - Measures linear correlation between predicted and actual values.

- **Concordance Correlation Coefficient (CCC)**:
  - Combines correlation and agreement; widely used in continuous ER tasks like RECOLA and SEWA.


### 8.3 Task-Specific Considerations

- **Binary vs. Multiclass**:
  - Binary tasks predict presence/absence; multiclass predicts full emotion label set.

- **Imbalanced Data**:
  - Use macro-F1 or weighted-F1 to correct skewed class distributions.

- **Real-Time Systems**:
  - May also evaluate latency, processing time, or system responsiveness.

## 9. Discussion and Future Challenges

### 9.1 Challenges

#### Data Limitations
- Most datasets are small, acted, and lack real-world complexity.
- Limited demographic diversity.
- Poor labeling of subtle or mixed emotions.

#### Fusion Complexity
- Requires:
  - Temporal synchronization
  - Cross-modal alignment
  - Handling of noisy or missing modalities

#### Label Ambiguity
- Emotions are subjective and culturally dependent.
- Annotator disagreement is common.
- Distinctions between emotion, mood, and feeling are blurred.

#### Real-Time Constraints
- Need for low-latency, lightweight models.
- Adaptation to mobile, embedded, or edge-computing environments.
- Support for continuous tracking, not just classification.

#### Ethical Concerns
- Privacy risks, especially with facial and voice data.
- Bias in datasets and models.
- Lack of transparency and explainability.

### 9.2 Promising Research Directions

- Self-supervised learning to reduce dependency on labeled data.
- Domain adaptation for better cross-dataset generalization.
- Robust fusion techniques for incomplete or noisy multimodal input.
- Culturally adaptive and context-aware emotion models.
- Explainable AI (XAI) to build trust and interpretability into ER systems.
