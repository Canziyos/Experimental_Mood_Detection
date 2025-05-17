import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from mobilenet.audio.audio_loader import make_audio_loaders
from mobilenet.audio.audio_model import AudioMobileNetV2
from mobilenet.audio.aud_trainer import train
import torch

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading data...")
    loaders = make_audio_loaders()

    print("Building model...")
    model = AudioMobileNetV2(pretrained=False, freeze_backbone=False).to(device)

    print("Training...")
    trained_model, history = train(model, loaders)

    print("Training complete. Best model loaded.")
