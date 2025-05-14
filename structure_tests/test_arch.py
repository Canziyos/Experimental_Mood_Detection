import torch
from models.ImageCNN2D import ImageCNN2D
from models.AudioCNN1D import AudioCNN1D

# audio_model = AudioCNN1D(input_channels=15, input_length=300)
# dummy_audio = torch.randn(4, 15, 300)  # (batch_size, channels, time).

# audio_output = audio_model(dummy_audio)
# print("Audio output shape:", audio_output.shape)  # Expect: (4, 6).

# audio_latent = audio_model.extract_latent_vector(dummy_audio)
# print("Audio latent vector shape:", audio_latent.shape)  # Expect: (4, 512).


# image_model = ImageCNN2D()
# dummy_image = torch.randn(4, 1, 48, 48)  # (batch_size, channels, height, width).

# image_output = image_model(dummy_image)
# print("Image output shape:", image_output.shape)  # Expect: (4, 6).

# image_latent = image_model.extract_features(dummy_image)
# print("Image latent vector shape:", image_latent.shape)  # Expect: (4, 64).


# import torch
# from models.FusionAV import FusionAV          # the class above

# BATCH = 4
# num_classes = 6

# # dummy audio feature maps (15×300)
# audio_dummy  = torch.randn(BATCH, 15, 300)

# # dummy face patches 48×48
# visual_dummy = torch.randn(BATCH, 1, 48, 48)

# # fake labels 0‑5
# labels = torch.randint(0, num_classes, (BATCH,))

# model = FusionAV(alpha=0.6, fusion_mode="avg").eval()

# with torch.no_grad():
#     probs = model(audio_dummy, visual_dummy)
#     preds = probs.argmax(1)
#     accuracy = (preds == labels).float().mean().item()

# print("Dummy batch accuracy (should float around 0.2‑0.3 for 6 classes):",
#       round(accuracy, 3))
# print("Output prob shape:", probs.shape)

# from config import Config
# import numpy as np

# cfg = Config()

# y = np.load(cfg.y_img_path)
# print("Labels loaded! Shape:", y.shape)

# X = np.load(cfg.x_img_path)
# print("Images loaded! Shape:", X.shape)

from config import Config
from dataset.face_loader import ImageEmotionDataset
from torch.utils.data import DataLoader


cfg = Config()

dataset = ImageEmotionDataset(cfg.x_img_path, cfg.y_img_path)

print("Number of samples:", len(dataset))
x, y = dataset[0]
print("Sample image shape:", x.shape)  # Expect: torch.Size([1, 48, 48])
print("Sample label:", y)              # Expect: integer in range 0–5


loader = DataLoader(dataset, batch_size=32, shuffle=True)

batch = next(iter(loader))
images, labels = batch
print("Batch image shape:", images.shape)   # Should be [32, 1, 48, 48]
print("Batch label shape:", labels.shape)   # Should be [32]
