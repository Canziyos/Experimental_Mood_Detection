# Affective Modeling – Foundations and Modern Trends


## Ekman, FACS, and why this legacy still haunts modern FER

Facial Expression Recognition (FER) did not start with deep learning – it started with Ekman.

Ekman extended Darwin’s idea that facial expressions are biologically hardwired and universal. He defined six (later seven) “basic emotions” – joy, anger, sadness, fear, disgust, surprise, and contempt – that he claimed could be recognized across all cultures. He also co-developed the **Facial Action Coding System (FACS)**, which breaks down facial movements into **Action Units (AUs)** – essentially muscle movements that can be combined to describe expressions.

### Why this became the default:

- FACS gave a **structured, reproducible way** to label facial expressions.
- Ekman’s emotion categories became the backbone of nearly every major FER dataset: FER2013, AffectNet, CK+, JAFFE.
- Easy-to-use labels + structured annotation = good enough for early classifiers → still baked into most modern pipelines.

### What’s often overlooked:

| Point | Why it matters |
|-------|----------------|
| **Cultural variation ≠ 0%**  
(Ekman was criticized by Russell & others) | Meta-analyses show only ~60–70% agreement across cultures. That’s not universal. Yet, FER datasets still assume Ekman categories are globally valid. This introduces baked-in bias during training. |
| **Micro-expressions (Ekman ’02)** | Ultra-fast expressions (<0.5 s), often unconscious. Humans mostly miss them. Deep AU detectors (OpenFace2, Retina-AU 2022) can pick them up reliably – outperforming rule-based systems. |
| **AU detection is a domain of its own** | Instead of predicting emotions directly, models like JAMNet (2020) and Vision-AU (2023) detect AUs from spontaneous datasets (e.g., BP4D-Spont). This makes the system more explainable and less culturally biased. |

**→ Bottom line:** FER models still rely heavily on Ekman-style labels. AU-based systems offer a cleaner, more interpretable path forward but depend on richer datasets and better annotation.

---

## Scherer, prosody, and the real building blocks of SER

Speech Emotion Recognition (SER) took a different route. Instead of focusing on visual cues, it emerged from **Scherer’s theory** that **prosody** (intonation, rhythm, loudness) carries the core emotional signal – not the semantic content.

### Why this mattered early on:

- Prosody is language-independent – you do not need to understand the words to detect emotion.
- This enabled feature-based SER long before deep learning.
- Classical SER relied entirely on handcrafted audio features → usable with traditional classifiers.

### Core tools and datasets:

| Detail | Why it matters |
|--------|----------------|
| **GeMAPS & eGeMAPS (2016)** | Standardized feature sets: 88 (GeMAPS), 23 (eGeMAPS). Still the default in traditional SER studies (e.g., IEMOCAP, EmoDB baselines). |
| **openSMILE toolkit** | The de-facto standard for extracting prosodic features from WAV files. Still widely used for benchmarking deep vs. classical pipelines. |
| **MSP-Podcast & EmoDB** | Naturalistic datasets replacing older, acted ones like IEMOCAP. MSP-Podcast is particularly large and realistic. |
| **Self-supervised learning + prosody**  
(e.g., Shon 2023) | Models like wav2vec2 or HuBERT are now fine-tuned using prosody-aware auxiliary losses. This merges handcrafted signal knowledge with deep embeddings. |

**→ Bottom line:** SER has moved from handcrafted features toward hybrid or self-supervised methods. But GeMAPS/openSMILE still define the baselines.

---

## From emotion to long-term mood – where theory and data hit a wall

FER/SER models are still mostly short-term: they classify visible or audible emotion in seconds. But **mood**, **affect**, and **clinical states** evolve **slowly** – across minutes, hours, or days.

### Key distinctions:

- **Emotion** = reactive, short-lived, externally triggered.
- **Mood** = longer-lasting, often without obvious expression.
- **Affect** = the underlying trend – physiological + behavioral.

### Modeling mood:

| Challenge | Why it matters |
|-----------|----------------|
| **Chronic affect ≠ episodic emotion** | Long-term mood detection needs **temporal fusion** across modalities (e.g., transformers over HR, audio, facial data). One clip ≠ enough. |
| **Lack of datasets** | Only a few datasets even try:  
- WESAD (stress, HR, emotion)  
- CLUE (contextual labeling of emotion)  
- SWELL-KU (stress in workplace context)  
None are large enough, and none focus on clinical mood states (e.g., BPSD). |

**→ Bottom line:** No clinical-grade mood detection system exists yet. Why? We don’t have the right data or long-range models. FER/SER alone can’t handle it.

---

## Model trends – what’s happened since CNNs and ViTs

Deep learning radically shifted affective computing, but many recent trends are direct responses to the limitations of FER/SER as originally framed.

| Trend | What it solves |
|-------|----------------|
| **Spatio-temporal ViTs**  
(e.g., TimeSformer, TokenShift) | Vision Transformers alone miss temporal dynamics. These models attend across space **and time**, learning motion implicitly. Key for micro-expressions and naturalistic data. |
| **Multi-task AU + emotion models**  
(e.g., JAMNet, DEER 2023) | AU branch gives interpretability; emotion branch provides classification. Leverages the structure of FACS while improving performance. |
| **Foundation multimodal models**  
(e.g., ImageBind, Uni-Perceiver v2) | Unified embedding space across text, image, and audio. Allows few-shot or zero-shot adaptation to affective tasks. Useful when labels are sparse or private. |
| **Bias and ethics auditing**  
(e.g., Rhue 2020) | AffectNet-trained FER models over-classify “anger” on darker skin tones. Bias audits are now essential – especially in healthcare or assistive systems. |

**→ Bottom line:**  
- FER is becoming explainable (AU + emotion).  
- SER is blending handcrafted + SSL.  
- Mood detection is still blocked by lack of longitudinal data.  
- Bias isn't an afterthought anymore — it's a showstopper.

---


1. **Ekman = cultural bias + micro-expressions** → Use with caution.  
2. **GeMAPS + openSMILE** → Still the standard prosody pipeline.  
3. **Datasets to know**:  
   - Audio: MSP-Podcast.  
   - AU: BP4D-Spont.  
   - Mood: WESAD, CLUE, SWELL-KU (limited).  
4. **Model trends**:  
   - Spatio-temporal ViTs  
   - Multi-task AU+emotion  
   - Foundation multimodal  
   - Bias-aware modeling  
5. **No true mood detection exists yet** → Datasets and modeling still immature.

