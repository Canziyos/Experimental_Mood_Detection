import os
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, Resample
from models.CNN1D import CNN1D
from utils import load_config

# Load config
cfg = load_config("config.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
sr_target = cfg["preprocessing"]["sample_rate"]
n_mels = cfg["training"]["input_features"]
target_length = 400
ckpt_path = cfg["ckpt"]["cnn1d"]
class_names = cfg["classes"]

# === Modify this line to test a different .wav file ===
# === Modify this line to test a different .wav file ===
wav_path = "animal_samples/dog.ogg"
assert os.path.exists(wav_path), f"File not found: {wav_path}"

# Optional: normalization
normalize = "global_mean" in cfg["preprocessing"]
if normalize:
    mean = np.array(cfg["preprocessing"]["global_mean"], dtype=np.float32)
    std = np.array(cfg["preprocessing"]["global_std"], dtype=np.float32)

# Transforms
mel_spec = MelSpectrogram(
    sample_rate=sr_target,
    n_fft=400,
    win_length=int(cfg["preprocessing"]["win_length"] * sr_target),
    hop_length=int(cfg["preprocessing"]["hop_length"] * sr_target),
    n_mels=n_mels
)
to_db = AmplitudeToDB()

def fix_length(mel_np, target_length):
    cur_len = mel_np.shape[1]
    if cur_len > target_length:
        return mel_np[:, :target_length]
    elif cur_len < target_length:
        return np.pad(mel_np, ((0, 0), (0, target_length - cur_len)), mode='constant')
    return mel_np

# Load audio
waveform, sr = torchaudio.load(wav_path)
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)
if sr != sr_target:
    waveform = Resample(sr, sr_target)(waveform)

# Extract features
mel = mel_spec(waveform)
mel_db = to_db(mel)
mel_np = mel_db.squeeze(0).numpy().astype(np.float32)
mel_np = fix_length(mel_np, target_length)

# === Debug before normalization ===
print("mel_np BEFORE normalization: mean =", mel_np.mean(), ", std =", mel_np.std())

if normalize:
    mel_np = (mel_np - mean[:, None]) / std[:, None]
    print("Normalization applied.")
    print("Mean shape:", mean.shape, "Std shape:", std.shape)

# === Continue inference ===
x = torch.tensor(mel_np).unsqueeze(0).to(device)

# Load model
model = CNN1D(input_features=n_mels, num_classes=len(class_names)).to(device)
ckpt = torch.load(ckpt_path, map_location=device)
if "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
elif "model" in ckpt:
    model.load_state_dict(ckpt["model"])
else:
    model.load_state_dict(ckpt)
model.eval()

# Inference
with torch.no_grad():
    logits = model(x)
    pred_idx = torch.argmax(logits, dim=1).item()

print("Predicted class:", class_names[pred_idx])
