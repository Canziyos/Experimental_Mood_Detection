# Emotion Terms & Multimodal Emotion Recognition Overview

## 1 . Emotion, Feeling, Mood, Affect, Sentiment, Dimensions – quick definitions
| Term | Typical time‑scale | Example |
|------|-------------------|---------|
|**Affect**|milliseconds‑seconds; raw valence/arousal|“pleasant activation”|
|**Emotion**|seconds‑minutes; contextual trigger|anger, joy, fear|
|**Feeling**|subjective report of an emotion|“I feel anxious”|
|**Mood**|minutes‑hours; diffuse|irritable, gloomy|
|**Sentiment**|linguistic attitude, usually text|positive tweet|
|**Emotion dimensions**|continuous axes|valence, arousal, dominance|

---

## 2 . Facial Expression Recognition (FER) inside Multimodal Emotion Recognition (MER)

* **Why FER?** Non‑intrusive, rich affect cues, runs in real time.
* **Psychology**  
  * Discrete view: Ekman’s six basics.  
  * Dimensional view: Valence–Arousal (±Dominance).  

### Typical pipeline
1. Face detection + alignment  
2. Pre‑processing (normalise, temporal smoothing)  
3. Feature extraction  
   * **FACS** (action units)  
   * **CNN / RNN / CNN‑RNN**  
   * Multiview & cross‑attention blocks to focus on eyes, mouth, brow  

---

## 3 . Audio‑Visual Emotion Recognition (A‑V MER)

| Why combine? |
|--------------|
|Faces can be faked, voice tone less so; voice can be noisy, face can help. Fusion is complementary.|

### Fusion flavours

| Level | Key idea | Pros | Cons |
|-------|----------|------|------|
|Early |Concatenate low‑level A & V features |Captures cross‑modal links |Sensitive to mis‑sync & noise|
|Late  |Average / vote final logits |Robust to missing stream |Misses deep interactions|
|Hybrid|Stack early + late with learnable weights |Best of both; uses attention |More parameters|
|Cross‑attention / Transformer|Align tokens across modalities |State‑of‑the‑art accuracy |Compute‑heavy|

---

## 4 . Recent models & headline numbers

| Study (year) | Modalities | Fusion | Result |
|--------------|------------|--------|--------|
|Le et al. 2023|Video + Audio + Text|Transformer (cross‑mod.)|Acc 78.98 % (IEMOCAP); 79.63 % (MOSI)|
|Mocanu et al. 2023|Audio + Face|Cross‑attention|Acc 89.25 % (RAVDESS); 84.57 % (CREMA‑D)|
|Zhang et al. 2022|Face + Speech|Encoder–decoder w. attention|+ 2.8 pp F1 vs. early concat|
|Feng et al. 2022|Face + Speech + Text|Multi‑view attention|Gains on IEMOCAP & MSP‑IMPROV|

*(Numbers are reported by the authors; datasets and splits differ.)*

---

## 5 . Go‑to datasets

* **IEMOCAP** – 12 h acted + improvised dyads, multimodal.  
* **CMU‑MOSEI** – 23 k sentence‑level clips from YouTube.  
* **CREMA‑D / RAVDESS** – lab‑recorded, balanced A‑V.  
* **AFEW / SFEW** – movie scenes “in the wild”, noisy faces.

---

## 6 . Open challenges

1. **A–V sync** – even 100 ms drift hurts early fusion.  
2. **Occlusion & lighting** – scarves, masks, backlight.  
3. **Compute budget** – transformers ↑ accuracy but ↑ latency.  
4. **Label mismatch** – datasets use different emotion taxonomies.  
5. **Real‑time trade‑off** – edge devices need sub‑100 ms inference.
