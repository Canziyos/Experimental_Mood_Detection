The most common *baseline* models and datasets in emotion‑recognition research (as of 2024).


## 1 .FER

* **Classic baseline** – a small VGG‑style CNN (4‑6 conv layers) trained on FER2013.  
* **Modern baseline** – **ResNet‑18** (or ResNet‑50 for larger setups) fine‑tuned on FER+, RAF‑DB and AffectNet.  
* Later work such as **EfficientFace**, **ResMaskNet** or **ViT‑based FER** all start from these CNN foundations but add attention, masking or transformer blocks.

> **Key reference**: Barsoum et al., *“FER+: Training Deep Networks for Facial Expression Recognition with Crowd‑Sourced Label Distribution.”*

---

## 2 . SER

### Two mainstream pipelines

1. **Log‑Mel Spectrogram + 2D‑CNN** (often ResNet‑34 or VGG‑like).  
2. **MFCCs + BiLSTM / 1‑D CNN** (the older but still widely used variant).

> **End‑to‑end waveform**: Trigeorgis et al., *“Adieu Features? End‑to‑End Speech Emotion Recognition Using a Deep Convolutional Recurrent Network,”* shows that raw‑waveform CNN‑RNN can outperform hand‑crafted MFCC pipelines when enough data are available.

* **Benchmark datasets**: **IEMOCAP** (de facto standard), CREMA‑D, RAVDESS.

---

## 3 . Multimodal Emotion / Sentiment Analysis

* **Tensor Fusion Network (TFN)** – first widely adopted end‑to‑end fusion of audio + video + text (Zadeh et al., 2017).  
* Since 2020, most papers also compare against **LMF**, **MulT**, and **MISA**, which provide parameter‑efficient or transformer‑based fusion.

* **Benchmark datasets**: **CMU‑MOSI** and **CMU‑MOSEI**.

---

## 4 . Continuous Affective State / “Mood” Detection

* No single canonical model.  
* Most systems stack **FER** and/or **SER** encoders and add a temporal model (**BLSTM, GRU, Temporal CNN, or Transformer**) to track valence–arousal curves over time.  
* In clinical studies (dementia, depression) researchers often fuse non‑visual signals such as rPPG heart rate, accelerometry, or ambient audio.

* **Datasets**: RECOLA, SEWA for valence/arousal; smaller private clinical sets for dementia.

---

### Ref table

| Task | 2024 *de facto* baseline | Typical datasets |
|------|--------------------------|------------------|
| **FER** | ResNet‑18 (or VGG‑16) fine‑tuned on FER+, RAF‑DB, AffectNet | 30 k – 450 k images |
| **SER** | Log‑Mel Spectrogram + 2D‑CNN **or** MFCC + BiLSTM | IEMOCAP, CREMA‑D |
| **Multimodal Sentiment / Emotion** | TFN (classic) ‑ plus LMF, MulT, MISA in modern comparisons | CMU‑MOSI, CMU‑MOSEI |
| **Mood / Continuous Affect** | Multimodal encoders + temporal model (BLSTM/GRU/Transformer) | RECOLA, SEWA; clinical diaries |

---

**Bottom line**:  
*Small VGG‑type CNNs on FER2013 and MFCC + LSTM pipelines remain historical baselines, but current comparison tables usually start with ResNet‑18 (FER) and log‑mel ResNet or BiLSTM (SER). TFN is still the first “standard” multimodal model, but newer fusion networks (LMF, MulT, MISA) are now co‑baselines. Most mood‑tracking systems simply add a temporal layer on top of proven FER/SER encoders.*