# Fusion Planning – Notes Mood1 and Mood2.

Fusing mood1 (audio+face) and mood2 (face+thermal) models (real-time embedded deployment).

## Output format

Final Models should give the same kind of output, e.g.,

```json
{
  "modality": "",
  "source_id": "mood1_audio_face",
  "timestamp": 1713912432.123,
  "window_duration": 5.0,
  "embedding": [0.12, 0.85, 0.03, ...],
  "label_type": "valence_arousal",
  "confidence": 0.93
}
```
### Fields:
- `modality`: "audio_face" or "face_thermal"
- `timestamp`: UNIX float
- `embedding`: fixed size (e.g., 64-, 128-dim)
- `label_type`: valence/arousal or emotion class

### Usefull fields:

- Fusion Weight.
- smothing over time
- uniuqe per team

- The same kind of output
- the same resolution (valence/arousal or 6 classes).
- Outpout timing allignment (e.g., 5s-window)


```mermaid
flowchart TD
    A1[Mood1 Encoder] --> B1[Output1]
    A2[Mood2 Encoder] --> B2[Output2]
    B1 --> F[Fusion MLP]
    B2 --> F
    F --> OUT[Mood Prediction]
 ```

Fusion model just takes both embeddings and combines them.

Note:  
Fusion deep inside either model = impractical since two independent groups.
