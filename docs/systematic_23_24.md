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
