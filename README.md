# Experimental_Mood_Detection
Personal sandbox for experimenting with MER models, data strategies, and uncertainty handling in mood detection for elderly care.

## What has been done so far

- Audio model is based on Srihari et al., with slight modifications.

### Dataset Preparation

- Please review the `README` files in both `clean_features` and `augmented_features` to locate your dataset and understand the folder structure.
- To build the datasets, run the following scripts (found in the `scripts` folder):
  - `build_clean_dataset.py`
  - `build_augmented_dataset.py`
- Once the datasets are prepared, you can proceed to run `audio_train.py`.  
  (Note: other components will be added gradually.)


- **Feature space analysis**  
  Feature vectors from the trained models were extracted and visualized using t-SNE to observe how different emotion classes are represented in the latent space.

- **Version control and cleanup**  
  `.gitignore` has been configured to exclude datasets, model weights, image outputs, and other large files from version control.

## Next steps

- Implement uncertainty handling with softmax thresholding to allow the model to output "unknown" for low-confidence predictions
- Experiment with model ensembling between MobileNet and ResNet
- Investigate the use of synthetic elderly faces using face-aging tools or generative models
- Explore training with frozen layers and gradual fine-tuning
