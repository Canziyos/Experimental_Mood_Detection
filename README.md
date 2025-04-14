# Experimental_Mood_Detection
Personal sandbox for experimenting with FER models, data strategies, and uncertainty handling in mood detection for elderly care.

## What has been done so far

- **Dataset preparation**  
  RAF-DB is the base dataset. Faces were filtered by estimated age (≥ 47 years) using DeepFace to isolate examples of older individuals, followed by manual control.

- **Model training**  
  Two models were trained from scratch using full fine-tuning:
  - MobileNetV2 trained for 10 epochs
  - ResNet18 trained for 3 epochs  
  Both models achieved strong accuracy, particularly on subsets containing elderly faces.

- **Evaluation**  
  Accuracy was measured on the full test set and on the filtered elderly subset. Confusion matrices were generated, and all predictions were logged to CSV.

- **Feature space analysis**  
  Feature vectors from the trained models were extracted and visualized using t-SNE to observe how different emotion classes are represented in the latent space.

- **Version control and cleanup**  
  `.gitignore` has been configured to exclude datasets, model weights, image outputs, and other large files from version control.

## Next steps

- Implement uncertainty handling with softmax thresholding to allow the model to output "unknown" for low-confidence predictions
- Experiment with model ensembling between MobileNet and ResNet
- Investigate the use of synthetic elderly faces using face-aging tools or generative models
- Explore training with frozen layers and gradual fine-tuning
