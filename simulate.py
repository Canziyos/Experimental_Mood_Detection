import torch
from models.FusionAV import FusionAV


model = FusionAV(alpha=0.5, fusion_mode="avg", learn_gate=True)

audio_input = torch.randn(8, 1, 16000)


visual_input = torch.randn(8, 3, 224, 224)


output_probs = model(audio_input, visual_input)

print(output_probs)