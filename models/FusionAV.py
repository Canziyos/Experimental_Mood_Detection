import torch
import torch.nn as nn
import torch.nn.functional as f
from models.AudioCNN1D import AudioCNN1D
from models.ImageCNN2D import ImageCNN2D


class FusionAV(nn.Module):
    def __init__(self):
        super(FusionAV, self).__init__()

        # our models.
        self.audio_branch = AudioCNN1D()
        self.visual_branch = ImageCNN2D()


    def forward(self, audio_input, visual_input):

        # get softmax from each branch
        aud_probs= f.softmax(self.audio_branch(audio_input), dim=1)       # [batch_size, num_classes]
        vis_probs=f.softmax(self.visual_branch(visual_input), dim= 1)

        #Aggregate probabilities by averaging.

        aggregated_probs=(aud_probs + vis_probs)/2

        return aggregated_probs


    def compare_predictions(self, aggregated_probs, labels):
        # Get predicted labels from aggregated probabilities.
        predicted_labels = torch.argmax(aggregated_probs, dim=1)  # [batch_size]

        # Compare with ground truth labels.
        correct_predictions = (predicted_labels == labels).sum().item()
        total_samples = labels.size(0)

        # Calculate accuracy.
        accuracy = correct_predictions / total_samples
        return accuracy