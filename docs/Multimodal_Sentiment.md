
1. **Bimodal step**  
   * Fuse each pair (L + A, L + V, A + V) with fully‑connected layers that learn joint features.

2. **Trimodal step**  
   * Concatenate the three bimodal embeddings, pass through another dense layer to get the final joint vector.

Benefits: keeps cross‑modal interactions but with far fewer parameters than one giant tensor.

---

## 3.  Context Modeling

* For each video, utterance‑level embeddings are fed into a **GRU**.  
* The GRU captures how sentiment drifts across the dialogue.  
* Output at each time‑step is sent to a soft‑max (binary ±  or 7‑way) or regression head (–3 … +3).

---

## 4.  Experimental setup

| Item | Detail |
|------|--------|
|Text features|Pre‑trained 300‑d GloVe → 1‑layer Bi‑LSTM|
|Audio|74‑d openSMILE prosody & MFCC|
|Video|35‑d FACET facial AUs|
|Dataset|**CMU‑MOSI** (2199 labelled utterances)|
|Baselines|Unimodal, early concat, Tensor Fusion Network (TFN)|

---

## 5.  Main results on MOSI (binary sentiment)

| Model | Acc. | F1 | MAE (reg.) | r (corr.) |
|-------|------|----|------------|-----------|
|Text only|71.6 %|0.71|1.12|0.55|
|Early concat|73.4 %|0.73|1.04|0.62|
|**TFN (2017)**|74.6 %|0.74|1.01|0.65|
|**HFusion + Context (ours)**|**78.2 %**|**0.78**|**0.93**|**0.71**|

*Hierarchical fusion* beats TFN by ≈ 3.5 pp accuracy and lowers MAE.

---

## 6.  Take‑aways

* Pairwise‑then‑trimodal fusion is compact and captures richer interactions.  
* Modeling **dialogue context** with a GRU gives a solid boost (≈ 2 pp).  
* Code runs in real time for short videos; no heavy 3‑D CNN required.  
* Approach generalises to MOSEI and IEMOCAP with similar gains.
