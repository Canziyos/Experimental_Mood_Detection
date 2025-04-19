
| Part | Detail |
|------|--------|
|**Video branch**|15 face frames (3.6 s) → MTCNN crop → **EfficientFace** encoder → 1‑D conv stack for temporal info|
|**Audio branch**|MFCC, Mel‑spec, Spectral‑contrast, Tonnetz → 1‑D CNN (Conv + MaxPool + Dropout)|
|**TFusion**|Transformer attention that learns audio↔video links and ignores a missing stream without zero‑padding|
|Classifier|Fully‑connected → 7 emotions (calm, happy, sad, anger, fear, disgust, surprise)|

Training: Adam, LR 0.0003, batch 32.

---

## 3.  Data

| Corpus | Notes |
|--------|-------|
|**RAVDESS**|North‑American actors; speech only part used here|
|**RML**|720 video clips, 6 emotions|
|**eNTERFACE’05**|More natural expressions ⇒ harder to classify|

All clips trimmed to **3.6 s** to align streams.

---

## 4.  Main results

### Overall accuracy

| Setup | Top‑1 | Top‑3 |
|-------|-------|-------|
|Audio only|37.0 %|77.6 %|
|Video only|67.7 %|92.3 %|
|**Audio + Video (TFusion)**|**77.6 %**|**96.0 %**|

* TFusion lifts accuracy by ≈ 10 pp vs. video‑only and ≫ audio‑only.  
* Works even if one channel is missing.

### Per‑dataset (Top‑1)

* RAVDESS 82.9 %  
* RML 82.5 %  
* eNTERFACE’05 72.3 % (hardest)

---

## 5.  Live demo app

* Python + Tkinter + PyAudio.  
* Multi‑threaded: webcam, mic, TFusion inference in real time.  
* Falls back to single‑modality if camera or mic is absent.  
* Shows camera preview and a bar plot of emotion probabilities.

---

## 6.  Take‑aways

* **TFusion** is effective for bimodal emotion tasks and naturally tolerates missing data.  
* Video branch (EfficientFace + 1‑D conv) gives solid speed for real‑time use.  
* Audio branch still the weak link; authors plan improved audio encoders and extra modalities (EEG, text).

