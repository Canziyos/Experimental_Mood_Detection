- Softmax applied during training (loss): Not in model (assumed used via CrossEntropyLoss)
In Srihari et al.’s paper:

They never explicitly state the input shape in terms of feature length.

But they say the final flattening after 5 conv layers gives 23,936 neurons.

That flattening comes from:

Final conv layer --> 128 channels
=> 128 × T = 23,936
=> T ≈ 187 (i.e., time steps)
So their effective input length seems to be T = 187, possibly padded or clipped to match this during preprocessing.

### Deviations

1. Input Length
   * our model: `input_length = 300`
   * Paper: Implied `input_length ≈ 187` (based on 128 × 187 = 23,936 before FC)


3. **Dropout Usage**
   * our model: `Dropout(0.3)` after `fc1`.
   * Paper: No mention of dropout.



* **Input Length**:
  - with 300 (more temporal resolution).
  - Later Action: Downsample/truncate input to 187 to replicate paper architecture exactly or reduce memory use.


- Dropout: to reduce overfitting.

we will make input_length dynamic by computing it during forward() using x.shape[-1] (later).

we will test the case with match the paper's flattening dim exactly, (experiment with input_length = 187) but our version gives higher resolution in time, which might even help!?