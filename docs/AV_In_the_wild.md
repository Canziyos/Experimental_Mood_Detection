## Feature‑ and Decision‑Level Audio‑Visual Fusion for Emotion Recognition  

A study on how **combining speech and facial cues** can improve emotion‑recognition systems, with tests on both **feature‑level** and **decision‑level** fusion.

---

### 1.  Why the paper matters  

* Unimodal (audio‑only or video‑only) systems drop in accuracy under real‑world noise.  
* Mixing the two streams can boost performance, but the *how* (feature vs. decision fusion) is still under debate.  
* The authors run head‑to‑head tests of both fusion styles using classical, low‑compute methods.

---

### 2.  Methods in plain terms  

| Stage | Audio branch | Visual branch |
|-------|--------------|---------------|
|**Feature extraction**|Standard speech descriptors (e.g. prosody, MFCC).|Three hand‑crafted image/video descriptors:<br>• **LBP** for single frames<br>• **QLZM** (Zernike‑moment variant)<br>• **LBP‑TOP** for full video volumes|
|**Dimensionality reduction**|PCA|PCA (separately for each visual set)|
|**Classifier**|Support‑Vector Classifier (SMO) or classic feed‑forward **Neural Net**|Same two classifier types|

#### Fusion flavours  

1. **Feature‑level fusion** – concatenate audio + visual feature vectors, then train one classifier.  
2. **Decision‑level fusion** – keep separate classifiers, then combine their soft‑scores per emotion (e.g. weighted sum or majority vote).

---

### 3.  Core results  

| System type | Gain vs. best single stream |
|-------------|----------------------------|
|Best unimodal|baseline|
|**Feature‑level fusion**|**+ 4 pp** accuracy|
|**Decision‑level fusion**|extra **+ 3 pp** on top of feature fusion|

*Some modality‑classifier pairs worked better for certain emotions (“anger” vs. “fear” etc.), so pooling the *best* per‑emotion outputs paid off.*

---

### 4.  How it fits into earlier work  

* **Busso et al. (2004)** – early 90 % accuracy with combined facial+acoustic cues.  
* **Rashid et al. (2012)** – Bayes‑rule decision fusion lifted scores on a smaller corpus.  
* **Kahou et al. (2013)** – deep nets (CNN, DBN) reached 41 % in the AVEC challenge.  
* **Soleymani et al. (2012)** – added EEG + gaze for valence/arousal prediction.  
* **Cruz et al. (2012)** – modelled feature derivatives over time with HMMs.

---

### 5.  Take‑aways  

* **No single modality or model is best for every emotion.**  
* Mixing audio and video at the **decision level** gave the largest jump (total ≈ 7 pp over best unimodal).  
* Classic hand‑crafted features plus SVM/NN are still competitive when compute is limited.  
* Next steps: smarter weighting of modalities over time and deeper temporal models.

