Nice — Prezi is a great choice for this kind of project since it helps show connections visually (which is perfect for multimodal systems). Since you’re already using it, here’s a step-by-step guide on **how to modify and improve your “Introduction” section** in Prezi.

---

## 🔧 Your Goal
Let’s build a **clear, flowing introduction section** in Prezi that matches the structure:

1. **Introduce the reader**
2. **Describe the subject**
3. **Explain the motivation**
4. **State the aim**

---

## Suggested Content Structure

### Frame 1: *What are emotions and how do we sense them?*
**Title:** *Understanding Emotions and Their Signals*  
**Text:**
- Emotions such as fear, joy, sadness reflect in facial expressions, voice, and physiology.
- These signals can be captured by **non-invasive sensors** like cameras and microphones.

📸 Use a relevant image (e.g., emotion faces, facial landmarks, waveform for voice).

---

### Frame 2: *Why does this matter in dementia care?*  
**Title:** *Challenges in Detecting Emotions in Dementia Patients*  
**Text:**
- Dementia often affects verbal communication.
- **Behavioral and Psychological Symptoms of Dementia (BPSD)** include agitation, depression, and aggression.
- Early detection of emotional changes can help prevent episodes.

---

### Frame 3: *Why multimodal fusion?*  
**Title:** *Limitations of Single-Sensor Systems*  
**Text:**
- **Facial Expression Recognition (FER)** struggles in poor lighting or with occlusions.
- **Speech Emotion Recognition (SER)** is vulnerable to background noise.
- No single sensor is reliable in all situations — we need fusion.

 Maybe show a split image of "camera problem in dark" + "mic in noisy room".

---

### Frame 4: *What is our approach?*  
**Title:** *Multimodal AI for Non-Invasive Mood Detection*  
**Text:**
- We combine facial expressions (FER) and speech signals (SER).
- Use **AI models (CNN, RNN)** to improve emotion recognition.
- Goal: Support **non-pharmacological BPSD care** through proactive mood alerts.

---

## 🧭 How to Modify Prezi (step by step)

1. **Click “+ Add Frame”** → choose “Zoom to Frame” for each sub-section.
2. For each frame:
   - Double-click the title to change it.
   - Double-click the text box to paste the suggested content.
   - Use “Insert → Image” to add visuals from your computer or search online.
3. Connect the frames logically in your overview.
   - Drag arrows or arrange zoom order using the left-side frame panel.
4. Use **consistent style**:
   - Fonts: Use the same font across all slides.
   - Colors: Keep background light, text dark (for accessibility).

---

Want me to now write **Prezi-optimized text** for each of these frames? (Brevity + clarity for each bubble.)


## *Emotion Recognition Using Different Sensors, Emotion Models, Methods and Datasets* (Sensors 2023)

- **Speech features** (such as pitch, energy, MFCCs - Mel Frequency Cepstral Coefficients) are **widely used** to recognize emotions.
- Highlights that **SER complements FER** because vocal tone often reflects **emotions not visible** in the face.
- Discusses fusion with audio as a way to **overcome occlusion or visual distortion** issues.
- Lists key datasets like **IEMOCAP** and **CREMA-D** that include annotated speech emotion data .

---

## *A Systematic Review on Multimodal Emotion Recognition* (2024)

- States that **audio signals** convey **prosodic** (rhythm, pitch) and **spectral** (tone, frequency) cues linked to emotion.
- Discusses how emotions like **anger, sadness, fear** change the acoustic profile of speech.
- Explains that **fusing audio and visual** signals improves robustness, especially in noisy or poorly lit settings.
- Emphasizes the use of **cross-modal transformers** and **attention mechanisms** to balance audio and visual input.

---

## *A Multimodal Fusion Approach: Emotion Identification from Audio and Video Using CNNs* (2024)

- Audio input is processed using:
  - **Zero Crossing Rate (ZCR)**.
  - **Root Mean Square (RMS)** energy.
  - **MFCCs** — commonly used in SER.
- Highlights how CNN-based models can **learn emotional cues from raw audio**, especially when enhanced with data augmentation.
- Audio-only model achieved **~79% accuracy**, confirming speech’s strong emotional signal .

---

## *Feature and Decision Level Audio-Visual Data Fusion* (IEEE)

- Audio features extracted using **openSMILE**, a standard tool for SER, focusing on prosodic and spectral cues.
- Demonstrates that **audio-only classification** (using W-SMO or NN) achieved decent accuracy, especially for emotions like **fear** and **anger**.
- When **fused with visual features**, the overall accuracy **improved significantly** (up to +7%) — showing speech contains **distinctive emotion signals**.
- Confirms that **some emotions are better detected in speech** than in facial expressions, and vice versa.

