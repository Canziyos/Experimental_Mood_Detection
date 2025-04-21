## Survey on audiovisual emotion recognition: databases, features, and data fusion strategies

## 1. Introduction (New Contributions Only)

- Focus is specifically on audiovisual emotion recognition (AER) — facial expressions and vocal cues.

- Model-level fusion is emphasized as more flexible for **asynchronous data**.

- Introduces and analyzes **benchmark challenges**:
  - **AVEC**, **EmotiW**, **INTERSPEECH** — used to evaluate AER systems under real-world constraints.

- Highlights AER-specific issues:
  - Modality dominance (e.g., visual vs. audio depending on emotion type).
  - Temporal alignment problems in bimodal fusion.

> Reference overlapping content on terminology, emotion models, and general modalities to the previous summaries see [Emotion recognition and AI – systematic review](systematic_23_24.md) and [Sensors 2023](sensors2023.md).


## 2. Databases (New Contributions Only)

### Key Additions Compared to Previous Reviews

- Three-way classification of corpora:
  1. **Posed**: acted emotions (e.g., RAVDESS, eNTERFACE).
  2. Induced: emotions elicited through stimuli (e.g., MAHNOB-HCI).
  3. Spontaneous: natural emotion during interaction (e.g., SEMAINE, RECOLA, SEWA).

- Highlights **spontaneous corpora** as more realistic but harder to annotate and standardize.

- Details **benchmark challenge datasets** specific to audiovisual ER:
  - **AVEC (Audio/Visual Emotion Challenge)**: focuses on continuous emotion prediction (valence/arousal).
  - **EmotiW (Emotion Recognition in the Wild)**: uses real-world movie clips for classification tasks.
  - **INTERSPEECH ComParE**: challenges on paralinguistic affect (e.g., arousal, depression, sleepiness).

- Mentions **SEWA** dataset (SEntiment analysis in the Wild):
  - Spontaneous audiovisual interactions.
  - Uses both categorical and dimensional annotations.

- Discusses **annotation formats** more deeply:
  - Time-continuous emotion annotations (e.g., RECOLA, AVEC).
  - Multi-rater aggregation strategies.

- Observes that many corpora suffer from:
  - Lack of diversity (age, language, culture).
  - Poor audio–video synchronization.

> For basic dataset coverage like IEMOCAP, CREMA-D, AffectNet, DEAP, refer to earlier reviews.
---
## 3. Feature Extraction and Representation

### Visual Features (New Points)
- Emphasizes **dynamic features** (not just static frame-based ones):
  - **LBPTOP** (Local Binary Patterns from Three Orthogonal Planes) – captures spatiotemporal texture changes across XY, XT, YT planes.
  - **Optical Flow–based motion features** for facial movement.
  - **3D CNNs** – extract both spatial and short-term temporal cues from video.

- Mentions **appearance + geometry fusion**:
  - Combines facial textures with landmark-based distances or angles.

### Audio Features (New Points)
Wu et al. emphasize the role of classic low-level descriptors (LLDs) in capturing expressive cues from speech. These frame-level features reflect both **acoustic** and **physiological** properties of the speaker under different emotional states.

| Feature    | Description                                   | Emotional Relevance                           |
|------------|-----------------------------------------------|------------------------------------------------|
| **MFCCs** (Mel-Frequency Cepstral Coefficients) | Represent spectral envelope based on human hearing scales. | Encodes voice timbre; useful across all emotions. |
| **Pitch** (F0)           | Fundamental frequency of vocal fold vibration. | High in anger, surprise; low in sadness. |
| **Intensity**            | Loudness or energy of speech.                | Higher in fear, anger; lower in sadness. |
| **Formants** (F1, F2, ...) | Resonant frequencies tied to articulation.   | Shifts in formants may indicate tension or articulation changes due to emotion. |
| **Jitter**               | Microvariations in pitch over time.         | Increased in stress, fear, or fatigue. |
| **Shimmer**              | Microvariations in amplitude (loudness).    | Elevated in nervousness or emotional instability. |

#### Toolkits and Standards
- **openSMILE**: Widely used toolkit for extracting LLDs.
- **INTERSPEECH ComParE**: Provides standardized feature sets (e.g., eGeMAPS) for benchmarking.


- Introduces **prosodic-emotion correlation**:
  - Mapping prosodic patterns to valence/arousal (e.g., rising pitch = high arousal).

- Highlights feature sets from:
  - **INTERSPEECH ComParE Challenges** – standardized LLD configurations.
  - **openSMILE** toolkit – widely used for acoustic feature extraction.

---

### Crossmodal Feature Issues
- Notes that:
  - Visual and audio features are often **sampled at different rates**, causing alignment challenges.
  - **Asynchronous onset** of emotion in face vs. voice is a nontrivial problem.

- Suggests using:
  - **Sequence alignment** or **attention-based weighting** to deal with temporal misalignment.

> For general audio/visual features like CNNs, HOG, LBP, MFCCs, see earlier summaries, [1](sensors2023.md) and [2](systematic_23_24.md).


## 4. Data Fusion Strategies (New Contributions Only)

Wu et al. provide a more structured breakdown of fusion in audiovisual ER, classifying it into four levels:

### 4.1 Feature-Level Fusion (Early Fusion)

- Concatenation of audio and visual features into a single vector.
- Requires **precise temporal alignment** and **feature normalization** across modalities.
- Challenges:
  - Dimensionality mismatch.
  - High risk of overfitting if modalities are weakly correlated.
  - No flexibility to handle missing modalities.

### 4.2 Decision-Level Fusion (Late Fusion)

- Each modality is processed by an independent classifier; decisions are combined via:
  - Majority voting
  - Confidence-weighted averaging
  - Ensemble meta-learning (e.g., stacking)

- Pros:
  - Modality independence.
  - Robust to modality failure.
- Cons:
  - Ignores deep intermodal relationships.

---

### 4.3 Model-Level Fusion (Intermediate Fusion)

- Introduced as a **more flexible and powerful alternative** to early/late fusion.
- Combines modalities during **hidden layers**, not at input or output.
- Examples:
  - Parallel subnetworks for each modality with shared layers or fusion gates.
  - Attention mechanisms for **modality-aware feature integration**.

- Handles:
  - **Asynchrony** between modalities.
  - **Modal imbalance** (e.g., one modality dominating the other).

### 4.4 Hybrid Fusion

- Combines multiple fusion strategies in multi-stage architectures.
  - E.g., Feature fusion → model fusion → late ensemble.

- Often seen in SOTA systems:
  - Combines benefits of robust modality-specific processing with joint representations.


### 4.5 Key Insights and Trends

- **Model-level fusion** is most promising for AER under real-world constraints (noise, misalignment, missing data).
- **Attention-based fusion** is growing rapidly:
  - Learns to dynamically weight modalities.
  - Often integrated into transformer-based architectures.

> This section goes deeper than previous reviews, especially in its emphasis on **model-level fusion**, its architecture patterns, and the handling of asynchrony.

---
## 5. Benchmarking and Performance Evaluation

### 5.1 Performance Trends

- Visual-only models tend to outperform audio-only models for most **categorical emotion classification** tasks.
- Audio tends to perform better in **arousal prediction** (prosodic cues), while **valence** is better predicted from visual signals.

### 5.2 Audiovisual Systems Comparison

- Summarizes results from **benchmark datasets** like:
  - **AVEC**
  - **EmotiW**
  - **RECOLA**
  - **SEWA**
  - **RAVDESS** (less commonly used for fusion tasks)

- Presents results as a function of:
  - Fusion strategy (feature-level vs. model-level vs. hybrid)
  - Input representation (raw vs. hand-crafted vs. learned)
  - Annotation type (categorical vs. dimensional)

### 5.3 Metrics Emphasis

- For **classification**:
  - Accuracy, F1-score, confusion matrix — standard.
- For **regression (continuous)**:
  - Strong emphasis on **CCC (Concordance Correlation Coefficient)** as a primary metric.
  - Pearson correlation (PCC) and RMSE used as complementary.


### 5.4 Noted Gaps and Observations

- Few studies report **cross-database performance** or **subject-independent generalization**.
- Lack of **standardized evaluation protocols** across papers limits comparison.
- Many fusion methods are still **tuned on specific datasets** — poor generalizability.
---

## 6. Challenges and Future Directions (Key Points Only)

### 6.1 Technical Challenges

- **Temporal Misalignment**:
  - Audio and visual signals often differ in timing (onset, duration).
  - Requires smarter fusion strategies (e.g., attention, alignment layers).

- **Modality Dominance & Redundancy**:
  - One modality may dominate or duplicate information from another.
  - Risk of overfitting if not handled during fusion.

- **Data Imbalance & Sparsity**:
  - Scarcity of spontaneous, multimodal emotion data.
  - Most datasets are limited in speaker, language, and setting diversity.

- **Label Ambiguity**:
  - Emotion categories often overlap or change over time.
  - Continuous and categorical labels are hard to align.

### 6.2 Future Directions

- **Modality-Adaptive Fusion**:
  - Dynamically adjust weights or strategies based on modality reliability or context.

- **Pretraining and Transfer Learning**:
  - Leverage large-scale emotional or general-purpose corpora (e.g., using transformers, contrastive learning).

- **Multitask Learning**:
  - Combine emotion recognition with related tasks (e.g., sentiment, speaker traits) for more robust models.

- **Cross-Domain and Cross-Cultural Generalization**:
  - Address model bias across language, culture, and demographics.

- **Explainable Multimodal Emotion Models**:
  - Improve interpretability and transparency of fused models — especially important for HCI and health domains.

> Compared to earlier reviews, Wu et al. emphasize **alignment and redundancy handling**, and highlight **modality-adaptive and interpretable fusion** as key future research areas.
