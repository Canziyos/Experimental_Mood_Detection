# Datasets Used in Experimental_Mood_Detection

General-purpose and filtered datasets are used to support facial, so far, emotion recognition with a special focus on elderly individuals.

---

## 1. RAF-DB (Real-world Affective Faces Database)
- **Source**: https://www.
- **Description**: A large-scale facial expression dataset with real-world images labeled for 7 basic emotions.
- **Usage**: Used as the primary dataset for training base models (MobileNetV2 and ResNet18 so far).
- **Format**: Images are categorized into folders by emotion class (1–7).

---

## 2. Dataset_eld
- **Source**: Subset filtered from RAF-DB using DeepFace (Filtered Elderly Dataset).
- **Criteria**: Faces estimated to be ≥ 45 years old.
- **Filtering Method**: Used DeepFace for age estimation and manual inspection to validate.
- **Usage**: Used for testing generalization performance of models on elderly faces.

---
## 3. Humans
- ** **: 

---

## Notes
- Metadata such as emotion labels and filtered IDs are stored separately in CSV files.
