## Novel Contributions from Dementia-Focused ER Papers

### 1. Clinical Label: NPI Score
- **NPI** = Neuropsychiatric Inventory.
- Measures 12 BPSD symptoms (e.g., agitation, apathy, hallucinations).
- Each symptom rated by **frequency (0–4)** and **severity (0–3)**.
- Combined score = Frequency × Severity.
- **Used as ground truth** in studies for evaluating emotional state correlation with BPSD.

---

### 2. Emotional Variability
- Not just static emotion classification (e.g., "happy" or "sad"), but analysis over time.
- **Emotional variability** includes:
  - Frequency of emotional state switches (e.g., neutral → angry → sad).
  - Standard deviation or variance in emotion probabilities over time.
  - Emotional inertia (how long a person stays in one emotion).
- Strong correlation found between **high variability + negative emotion dominance** and **higher BPSD severity**.

---

### 3. Multimodal Monitoring in Real Clinical Settings
- Data collected from **day care centers** (e.g., uAge Center).
- **Speech recordings** and **facial video** during natural interaction tasks.
- Context-aware setup vs. clean lab-controlled datasets.

---

### 4. Pipeline Summary (Chen et al. and Gong et al.)
**Step 1**: Record facial expressions or speech from patients.

**Step 2**: Use pretrained FER/SER models to classify emotion per time step.

**Step 3**: Extract features like:
- Mean emotion probability.
- Variance in emotion confidence.
- Number of emotion transitions.

**Step 4**: Feed these into:
- Linear regression.
- Random forest.
- Ensemble regression models.

**Output**: Predict the **NPI total score** or **specific BPSD symptom levels**.

---

### 5. Techniques Not Seen in General ER Surveys
- Use of **clinical scores** (not emotion categories) as prediction targets.
- Focus on **temporal emotion analysis** over single-frame prediction.
- Emotional "switch rate" used as a predictive feature.
- Ensemble models used for **interpretable clinical prediction**.
- Realistic deployment context: **elderly users**, **natural speech**, **low-intrusion sensors**.

---

### 6. Implications for our pro
- Justifies our idea of mood detection as a **BPSD early-warning system**.
- Suggests tracking **emotional dynamics** is more informative than just labels.
- Supports using **late fusion** with time-aware modeling.
- Highlights the importance of including **care-friendly, non-invasive data collection** methods.
- Inspires potential feature engineering directions: emotion fluctuation metrics.

---

- Implement a sliding window emotion tracker for audio/video.
- Analyzing switch rate, emotional stability, and dominance in predicted emotions.
- a simple regression model to simulate BPSD severity levels.
- Seek access to datasets like **Dem@Care** or contact uAge Center researchers.
- re-annotating part of public datasets with synthetic NPI-like labels for testing.