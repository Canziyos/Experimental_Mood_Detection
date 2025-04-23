# Glossary: Multimodal Emotion Recognition Terms

A curated reference of key concepts, models, and architectures used in facial and speech-based emotion recognition, including practical systems for dementia patient care.

---

## General Concepts

**FER (Facial Expression Recognition):**
Identifying emotions by analyzing facial movements using 2D or video input.

**SER (Speech Emotion Recognition):**
Classifying emotions based on acoustic features such as pitch, energy, and MFCCs.

**Multimodal Emotion Recognition:**
Combining two or more input sources.

**BPSD (Behavioral and Psychological Symptoms of Dementia):**
Includes agitation, aggression, depression, apathy, and psychosis. Accurate mood detection can aid non-pharmacological interventions.

---

## Key Models and Modules

**TFusion (Transformer Fusion Block):**
A transformer-based module for combining modality features using attention. Supports missing modalities without padding. First used in emotion recognition by Wozniak et al. (2023).

**EfficientFace:**
CNN model for facial expression recognition. Lightweight and real-time capable. Used as a facial feature extractor.

**Conv1D Block:**
Applies 1D convolutions over temporal sequences to capture dynamics in audio or video input with reduced computational cost.

**MMLatch:**
A feedback-based multimodal fusion system that integrates top-down and bottom-up processing. Incorporates feedback masks (via LSTMs) into the input streams.

**SAMGN (Structure-Aware Multi-Graph Network):**
Graph-based model for emotion recognition in conversations. Constructs separate graphs for each modality and dynamically learns edge weights between utterances. Uses dual-stream propagation.

---

## Audio Feature Types

**MFCC (Mel Frequency Cepstral Coefficients):**
Represents the envelope of the audio spectrum. Key feature in SER.

**Mel Spectrogram:**
Spectrogram mapped to the mel scale, approximating human hearing perception.

**Spectral Contrast:**
Captures difference between peaks and valleys in the frequency spectrum.

**Tonnetz:**
Describes harmonic content in speech, adapted from music analysis.

---

## Fusion Strategies

**Early Fusion:**
Combines raw or low-level features from multiple modalities early in the pipeline.

**Late Fusion:**
Each modality is classified separately, and their outputs are combined.

**Intermediate/Model-Level Fusion:**
Feature representations are merged **post-extraction but before classification**. Often includes attention-based mechanisms.

---

## Datasets

**RAVDESS:**
Audio-video dataset with actors performing scripted speech in 8 emotions.

**RML:**
Smaller dataset with 720 audiovisual samples labeled with basic emotions.

**eNTERFACE'05:**
More natural expressions by non-actors. Audio-video data labeled for six emotions.

**CMU-MOSEI:**
Large-scale multimodal dataset used for sentiment and emotion classification. Includes video, audio, and text.

**IEMOCAP:**
Multimodal dataset (emotion recognition from conversation, rich labeling).

**MELD:**
Multimodal EmotionLines Dataset. Includes context-aware utterance-level labeling.

---

## Base Architectures (Simplified Models)

**AU + Prosody SVM:**
Handcrafted OpenFace Action Units + prosodic audio features (e.g., pitch, jitter). Trained with SVM per patient.

**Two-Stream CNN-LSTM:**
Parallel CNNs for audio (Mel spec) and video (face frames), fused via LSTM. Learns temporal alignment.

**Joint Multimodal Transformer:**
Separate pretrained encoders (e.g., wav2vec 2.0 and ViT). Features are fused via cross-attention transformer layers.

---

## Considerations for Dementia-Oriented Models

- **Challenges:** Weaker expressions, slower speech, short utterances, medical noise.
- **Data Limitations:** Small datasets, ethical restrictions, patient-specific variability.
- **Baselines:** personal SVM, then transfer learning with EfficientNet/VGGish.
- **SSL Adaptation:** Use few-shot tuning with adapters per patient to personalize pretrained transformers.
- **Ethics:** GDPR, secure local inference, **avoid** wrong predictions when uncertain.

---

## Real-World Implementation Notes

- **Real-Time Systems:** Multithreaded app with PyAudio, TKinter GUI, and efficient CNN backbones.
- **Missing Modalities:** Use TFusion or similar attention fusion to allow flexible predictions.
- **Edge Computing:** lightweight models (EfficientNet B0, MobileNet, or SVM baselines).
- **Fail-Safe Modes:** Predictions should should not be output under low confidence or poor signal conditions.

