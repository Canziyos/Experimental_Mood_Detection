import torch
from models.image_model import ImageCNN2D
from models.audio_model import AudioCNN1D

audio_model = AudioCNN1D(input_channels=15, input_length=300)
dummy_audio = torch.randn(4, 15, 300)  # (batch_size, channels, time).

audio_output = audio_model(dummy_audio)
print("Audio output shape:", audio_output.shape)  # Expect: (4, 6).

audio_latent = audio_model.extract_latent_vector(dummy_audio)
print("Audio latent vector shape:", audio_latent.shape)  # Expect: (4, 512).


image_model = ImageCNN2D()
dummy_image = torch.randn(4, 1, 48, 48)  # (batch_size, channels, height, width).

image_output = image_model(dummy_image)
print("Image output shape:", image_output.shape)  # Expect: (4, 6).

image_latent = image_model.extract_features(dummy_image)
print("Image latent vector shape:", image_latent.shape)  # Expect: (4, 64).
