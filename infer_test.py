import torch
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from mobilenet.audio.AudioEmotionCNN1D import AudioMobileNetV2
from mobilenet.audio.log_mel import LogMelSpec
from mobilenet.fusion.FusionAV import FusionAV

# will be refactored soon.
video_path = "4.mp4"
audio_ckpt = "checkpoints/mobilenet_aud.pth"
fusion_mode = "avg"  # Can be: avg, prod, gate, mlp, latent
n_classes = 6
class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Models
aud_net = AudioMobileNetV2(pretrained=False, freeze_backbone=False).to(device).eval()
aud_net.load_state_dict(torch.load(audio_ckpt, map_location=device))

fusion = FusionAV(
    num_classes=n_classes,
    fusion_mode=fusion_mode,
    latent_dim_audio=128,
    latent_dim_image=128,
    use_pre_softmax=(fusion_mode in {"mlp", "gate"})
).to(device).eval()

logmel_model = LogMelSpec(img_size=(96, 192)).to(device)

# fake image embedding and logits
v_vec = torch.randn((1, 128), device=device)         # Fake image embedding
v_logits = torch.randn((1, n_classes), device=device)
v_prob = torch.softmax(v_logits, dim=1)

# extract audio features from video.
clip = VideoFileClip(video_path)
audio_path = video_path + ".wav"
clip.audio.write_audiofile(audio_path, fps=16000, verbose=False, logger=None)

wav_np, sr = sf.read(audio_path)
if wav_np.ndim > 1:
    wav_np = wav_np.mean(axis=1)
wav_torch = torch.tensor(wav_np, dtype=torch.float32).unsqueeze(0).to(device)

# prepare audio segment
samples = sr * 2
seg = wav_torch[..., :samples]
if seg.shape[-1] < samples:
    seg = torch.nn.functional.pad(seg, (0, samples - seg.shape[-1]))

spec = logmel_model(seg).unsqueeze(0)  # (1, 1, 96, 192)

# audio inference.
with torch.inference_mode():
    a_vec = aud_net.extract_features(spec)
    a_logits = aud_net(spec)
    a_prob = torch.softmax(a_logits, dim=1)

# fuse outputs.
with torch.inference_mode():
    fused = fusion.fuse_probs(
        probs_audio=a_prob,
        probs_image=v_prob,
        latent_audio=a_vec if fusion_mode == "latent" else None,
        latent_image=v_vec if fusion_mode == "latent" else None,
        pre_softmax_audio=a_logits if fusion.use_pre_softmax else None,
        pre_softmax_image=v_logits if fusion.use_pre_softmax else None
    )

cls = fused.argmax(1).item()
print("\n=== Fusion Output ===")
print(f"Fused class: {class_names[cls]} (index {cls})")
print(f"Audio probs: {a_prob.cpu().numpy().round(3).tolist()}")
print(f"Image probs: {v_prob.cpu().numpy().round(3).tolist()}")
print(f"Fused probs: {fused.cpu().numpy().round(3).tolist()}")

#visualize.
def visualize_fusion_outputs(a_prob, v_prob, fused, class_names=None):
    a = a_prob.squeeze().cpu().numpy()
    v = v_prob.squeeze().cpu().numpy()
    f = fused.squeeze().cpu().numpy()
    
    classes = class_names if class_names else [f'Class {i}' for i in range(len(a))]
    x = np.arange(len(classes))

    plt.figure(figsize=(10, 6))
    plt.bar(x - 0.25, a, width=0.25, label='Audio', color='skyblue')
    plt.bar(x, v, width=0.25, label='Image', color='salmon')
    plt.bar(x + 0.25, f, width=0.25, label='Fused', color='limegreen')
    plt.xticks(x, classes, rotation=45)
    plt.ylabel("Probability")
    plt.title(f"Fusion Comparison — Mode: {fusion_mode}")
    plt.legend()
    plt.tight_layout()
    plt.show()

visualize_fusion_outputs(a_prob, v_prob, fused, class_names)
