
## Mood vs. Emotion: Why Mood Detection is Harder

| Aspect | Emotion | Mood |
|--------|---------|------|
| Duration | Seconds to minutes | Hours to days |
| Trigger | External (event-based) | Often internal or untraceable |
| Expression | Clear facial/vocal markers | Subtle or absent cues |
| Labels | Discrete (e.g., anger, joy) | Fuzzy (e.g., apathetic, withdrawn) |
| Modeling | Single clip often sufficient | Needs temporal context / trends |

> In dementia care, many Behavioral and Psychological Symptoms of Dementia (BPSD) stem not from a single emotional outburst but from a **build-up of affective state**—which is mood. Thus, we must move beyond clip-wise classification and incorporate **temporal fusion and multimodal history** to detect these patterns early.

---

## Domain-Specific Considerations: Dementia and Elderly Faces

- Most benchmark datasets (e.g., RAVDESS, IEMOCAP) use young to middle-aged actors. This creates a **domain gap**:
  - Older faces have more wrinkles and asymmetry, altering feature distributions.
  - Micro-expressions may be **weaker or masked** due to neurological decline.
  - Visual occlusions (e.g., glasses, drooping lids) and speech irregularities are common.

**Mitigation strategies:**
- **Fine-tune models on synthetic or real elderly faces** (e.g., use DeepFace to filter for aged faces in AffectNet/RAF-DB).
- Use **soft labels or fuzzy emotion clusters** instead of hard classes to accommodate ambiguous expressions.

---

## Real-World Constraints: Missing or Degraded Signals

- Care homes are **uncontrolled environments**: lighting changes, sensors fail, patients move unpredictably.
- Our system must be:
  - Robust to **partial data** (e.g., only audio, only face),
  - Able to **gracefully degrade** without crashing.

Relevant strategies from literature:
- TFusion (bimodal_av.md): handles missing streams without zero-padding.
- Late-fusion averaging (Srihari_24.md): simple but surprisingly robust.
- Model-level fusion with attention (Wu et al.): dynamically weighs reliable signals.
