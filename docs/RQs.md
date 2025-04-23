

#### gap
> Existing emotion-recognition models often overfit to clean, acted data and fine‐grained emotion labels, (robustness to moise? and clinical relevance in general and in dementia care). Investigate how dataset diversity and label simplification affect mood-detection performance and generalization.

- **H1 (Data Diversity):** Combining datasets with varied datsets (recording conditions and demographic profiles) improves model robustness and cross-corpus generalization.  
- **H2 (Label Simplification):** Grouping emotion labels by valence (positive/neutral/negative) boosts classification accuracy and aligns better with clinical needs in dementia care.



Data Diversity RQ:  
   How does training on heterogeneous datasets—differing in recording setups or speaker demographics—impact the accuracy and generalization of non-invasive mood-detection models?

Label Simplification RQ:  
   Does reducing fine-grained emotion labels to valence-based categories (positive/neutral/negative) improve model performance and clinical interpretability for mood monitoring in dementia patients?


- each dataset’s characteristics (acting style, demographics, recording conditions).  
- valence categories (e.g., map “sad,” “angry,” “fear” to “negative”).  
-  held-out testing, plus cross-corpus tests. Report accuracy, F1, and maybe AUC.


- *why* valence grouping matters clinically: caregivers act on *negative shift* more than specific emotions.  
- Consider a small user-study or expert feedback round to validate that your simplified labels match care priorities.

