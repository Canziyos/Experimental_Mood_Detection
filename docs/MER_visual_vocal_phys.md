# Multimodal Emotion Recognition with Visual, Vocal & Physiological Signals – Review  
*Udahemuka G., Djouani K., Kurien A.M., 2024* :contentReference[oaicite:0]{index=0}

A survey of how recent systems detect **subtle, short, and dynamic** emotions by fusing **video, audio, and physiological** data.  
Focus areas: **micro‑expressions**, **temporal modelling**, and how **deep learning** now beats handcrafted pipelines.

---

## 1.  Modalities Covered

| Modality | Typical cues |
|----------|--------------|
|**Visual**|Facial expressions, gestures, micro‑movements|
|**Vocal** |Prosody (pitch, energy), spectral cues (MFCC)|
|**Physio**|EEG, ECG, EMG, GSR, respiration, etc.|

Each channel offers unique evidence; fusion tries to keep strengths and mask weaknesses.

---

## 2.  Datasets (acted / induced / natural)

| Sensor group | Key sets mentioned | Notes |
|--------------|-------------------|-------|
|Visual|FER2013 · CK+ · JAFFE · CASME II (micro‑exp.)|Many are **acted** or tightly controlled|
|Audio |EMO‑DB · RAVDESS|Language & acting bias remain issues|
|Physio|DEAP · SEED · DREAMER|Small subject counts|
|Multimodal|MAHNOB‑HCI · AMIGOS · SAVEE|Rare, costly, sync problems|

Review calls for **spontaneous, diverse, balanced data** to push the field forward.

---

## 3.  Feature‑Extraction Methods

### 3.1  Hand‑crafted (pre‑DL)

* **LBP / LBP‑TOP** – skin texture & motion  
* **HOG / SIFT** – edges & shape  
* **Optical flow / strain** – frame‑to‑frame motion  

> **Limitations:** require expert tuning, fragile with noise, struggle on partial faces or big identity variation.

### 3.2  Deep‑learning

| Architecture | What it brings |
|--------------|---------------|
|**CNN / 3D‑CNN**|Spatial (& temporal in 3‑D) features from raw frames|
|**Two‑stream nets**|Appearance + optical‑flow channels|
|**RNN / LSTM / GRU**|Sequence modelling of features over time|
|**Capsule / Attention**|Capture fine spatial relations; focus on apex frames|

Hybrid CNN + LSTM stacks dominate benchmarks like CASME II, SAMM, SMIC.

---

## 4.  Temporal Modelling Tips

* Micro‑expressions last **1/25 – 1/5 s** → need ≥ 100 fps cameras.  
* Model **onset → apex → offset** rather than single frames.  
* LSTM/GRU or 3D‑CNN outperforms static CNN for dynamic cues.

---

## 5.  Fusion Layers

* **Early/feature‑level** – concat vectors before classifier.  
* **Late/decision‑level** – vote or weight separate model outputs.  
* Choice depends on data sync, compute budget, and missing‑modality risk.

---

## 6.  Challenges the review lists

* **Data scarcity** for subtle / spontaneous cues.  
* Poor cross‑demographic generalisation.  
* Real‑world noise or sensor dropout.  
* Non‑standard evaluation protocols.

---

### Quick Glossary – Micro‑expression Phases

| Phase  | What happens |
|--------|--------------|
|**Onset** | Muscles start to leave neutral |
|**Apex**  | Expression at maximum intensity |
|**Offset**| Muscles relax back to neutral |

Capturing the full motion curve is key for reliable detection of these blink‑and‑miss emotions.