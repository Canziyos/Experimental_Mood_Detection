# Emotion‑Recognition Cheatsheet (detailed)

## 1. Emotion Models

### 1.1  Discrete (categorical)

| Model | Basic set | Extra notes |
|-------|-----------|-------------|
|Ekman (1970s)|happiness, sadness, anger, fear, disgust, surprise (+ contempt)|Backed by cross‑culture facial studies; rapid, short‑lived reactions|
|Plutchik wheel|8 opposites in a circle (joy↔sadness, trust↔disgust, etc.)|Adds intensity by “petal” distance from centre|

**Pros** • Straightforward labels • Matches most public datasets  
**Cons** • Can’t show blends (“bittersweet”) • Some culture‑specific differences

---

### 1.2  Dimensional

| Model | Axes | Typical use |
|-------|------|-------------|
|Valence‑Arousal|pleasantness, activation|Real‑time sliders for games, affective HCI|
|PAD|pleasure, arousal, dominance|Adds sense of control vs. submission|

**Pros** • Captures intensity & nuance • Great for continuous tracking  
**Cons** • Harder to annotate • Coordinates need post‑mapping to words

---

## 2. Sensor Options

| Class | Hardware | Captures | 👍 | 👎 |
|-------|----------|----------|----|----|
|Visual|RGB/IR cameras|Facial Action Units, micro‑expressions, rPPG|Cheap, passive|Lighting, privacy, spoofing|
|Audio|Mic|Prosody, pitch, energy|Works in dark, low HW|Noise, language bias|
|Radar/mmWave|60 GHz chips|Breathing & heartbeat at distance|Lighting‑independent|Motion clutter, small datasets|
|Wearables / Physio|EEG, ECG, EMG, GSR, BVP, EOG|Brain waves, heart rate, skin conductance|Difficult to fake|Intrusive, comfort|
|Multi‑sensor|Any combo|Fusion of above|Robust|Sync, cost, compute|

---

## 3. Fusion Levels

| Level | How | Pros | Cons | Typical use |
|-------|-----|------|------|-------------|
|Pixel / signal|Stack raw streams (e.g., RGB + depth)|Max info|Needs perfect alignment, heavy|RGB‑D gesture|
|Feature|Concat feature vectors (e.g., MFCC + HOG)|Good trade‑off|Must time‑align features|Academic papers, prototypes|
|Decision|Vote/average per‑sensor soft‑max|Sensor‑failure tolerant|Loses cross‑modal cues|Commercial products, modular systems|

---

## 4. Datasets (by modality)

### 4.1 Visual
* **CK+** – posed sequences, action units.  
* **JAFFE** – Japanese females, grayscale stills.  
* **RaFD** – high‑res, varied gaze.  
* **FER2013** – 35 k crowd images, noisy “in the wild”.  
> **Gotchas:** ethnicity imbalance, exaggerated poses.

### 4.2 Audio
* **Emo‑DB** (German, 7 emotions).  
* **RAVDESS** (24 actors, 8 emotions, matched audio‑video).  
* **IEMOCAP** (10 actors, scripts + improvs, multimodal).  
> **Issues:** acted speech ≠ real life; language limits transfer.

### 4.3 Physiological
* **DEAP** (32 ppl, EEG + peripherals, music videos).  
* **SEED** (15 ppl, EEG watching film clips).  
* **DREAMER** (23 ppl, EEG + ECG, audio‑visual stimuli).  
> **Issues:** small N, strong subject variability.

### 4.4 Multimodal
* **MAHNOB‑HCI** (video, EEG, ECG, GSR).  
* **AMIGOS** (group & solo, multi‑sensor).  
* **SAVEE** (British male AV dataset).  
> **Issues:** expensive, sometimes unsynchronised, hard to share.
