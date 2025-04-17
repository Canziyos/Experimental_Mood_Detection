
> **“Bimodal Emotion Recognition Based on Vocal and Facial Features”**  
> _Wozniak et al., Procedia Computer Science, 2023._

---

### **Core Idea**
Combine **vocal and facial features** using a deep learning. A key contribution is the use of the **TFusion block**, a transformer-based attention module that allows the model to handle **missing modalities** (audio or video is unavailable).

---

### **1. Motivation and Background**
- **Human communication is multimodal**: facial expressions, speech prosody, and gestures all carry emotional cues.
- Relying on only face or only speech is insufficient, especially in real-world settings.
- Existing emotion recognition systems often break down in noisy or occluded conditions.
- Inspired by the **McGurk effect** (how audio and visual inputs affect perception).

---

### **2. Proposed Method**

#### Architecture Overview:
- Two branches: **Audio** and **Video**
- Final fusion with **TFusion block**
- Final classifier outputs 1 of 7 emotions: *calm, happiness, sadness, anger, fear, disgust, surprise*

#### Video Branch:
- Input: 15 face frames over 3.6 seconds
- Face detection: **MTCNN** (Multi-task Cascaded Convolutional Networks)
- Feature extractor: **EfficientFace** (pre-trained)
- Uses **1D convolutions** to reduce complexity and capture temporal dynamics

#### Audio Branch:
- Features: **MFCC**, **Mel Spectrogram**, **Spectral Contrast**, **Tonnetz**
- Similar CNN blocks as video, with added **MaxPooling** and **Dropout**
- Final output is **matched to video branch in shape** using adaptive pooling.

#### TFusion Block:
- **Transformer-based fusion module**
- Handles **missing modalities** naturally (no need for zero-padding).
- Learns **cross-modal correlations** via attention.
- Outputs a **shared representation** for final classification.

---

### **3. Dataset**
Combined three widely-used datasets:
- **RAVDESS**: North American actors performing speech with emotion (used only speech clips).
- **RML**: 720 audiovisual clips, 6 emotions.
- **eNTERFACE’05**: Real people, which made expressions more natural but harder to classify

Each sample was trimmed to 3.6 seconds to standardize inputs.

---

### **4. Experiments & Results**

#### **Performance**

| Modality | Accuracy | Top-3 Accuracy | Explanation |
|----------|----------|----------------|-------------|
| **Audio only** | 37.01% | 77.64% | The model predicted the correct emotion in first place **only ~37%** of the time, but in **top 3 guesses** ~78% of the time. Not very strong alone. |
| **Video only** | 67.74% | 92.27% | Much better performance: nearly **68%** first-place accuracy, and **92%** of the time the correct label is within the top 3. |
| **Bimodal** | **77.56%** | **96.02%** | Best performance. Combining both modalities gave **highest accuracy** and reliability. |

- Audio-only misclassifies happiness and disgust most often.
- Video-only achieves best accuracy on happiness (90%+) and calm (83%+).
- **TFusion fusion** improves overall robustness, especially with **missing data**.

#### Per Dataset Accuracy:
- **RAVDESS**: 82.99%
- **RML**: 82.47%
- **eNTERFACE’05**: 72.27% (hardest due to more realistic acting)

---

### **5. Real-Time Application**
- A **multi-threaded desktop app** was developed to demonstrate the system in practice.
- Implemented in Python using **TKinter** and **PyAudio**
- Handles **live camera and microphone input**
- Automatically degrades to unimodal recognition if either audio or video is missing
- Visual output: camera preview + bar plot of predicted emotions

---

### **6. Conclusion**
- The TFusion-based bimodal system performs comparably to state-of-the-art.
- It handles **incomplete data** and achieves high emotion recognition accuracy.
- Future directions:
  - Improve the **audio branch**.
  - Add **new modalities** like EEG or text.

---

### Technical Innovations
- First use of **TFusion** in bimodal emotion recognition.
- **EfficientFace + 1D Conv** for fast real-time facial processing.
- Fully operational **application**, not just offline evaluation.
- Carefully tuned hyperparameters (Adam optimizer, LR=0.0003, batch size=32).
