
**“Multimodal Sentiment Analysis Using Hierarchical Fusion with Context Modeling”** (Majumder et al., 2018).

#### Problem:
- Sentiment analysis using **just text**, **just audio**, or **just video** often misses the full picture.
- Simple **feature concatenation** across modalities leads to *noisy*, hard-to-train models with redundant or misaligned information.

#### Proposed Solution:
- Use a **hierarchical fusion strategy**:
  - First combine modalities **pairwise** (bimodal fusion).
  - Then fuse the **bimodal representations** into a final trimodal one.
- Also introduce **context modeling**:
  - Each utterance is not analyzed in isolation.
  - Use RNNs (specifically **GRUs**) to understand how utterances relate to one another in a sequence (e.g., in a video or dialogue).


- Multimodal sentiment analysis on **utterances in videos**, particularly using datasets where each spoken segment is labeled for sentiment.
