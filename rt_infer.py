import torch, json, time, cv2, numpy as np
import torchvision.transforms as T
import soundfile as sf
from collections import Counter
from moviepy.editor import VideoFileClip
from mobilenet.config import Config
from mobilenet.fusion.FusionAV import FusionAV
from mobilenet.image.image_model import ImageMobileNetV2
from mobilenet.audio.audio_model import AudioMobileNetV2
from insightface.app import FaceAnalysis
import torchaudio

from mobilenet.audio.log_mel import LogMelSpec

logmel_model = LogMelSpec(img_size=(96, 192)).to("cuda" if torch.cuda.is_available() else "cpu")

def wav_to_logmel(wav: torch.Tensor) -> torch.Tensor:
    """
    Converts waveform tensor to log-mel spectrogram (1, 1, 96, 192).
    """
    spec = logmel_model(wav)      # (1, 96, 192)
    return spec.unsqueeze(0)      # (1, 1, 96, 192)

# tx_img = T.Compose([
#     T.ToPILImage(),
#     T.Resize(224),
#     T.ToTensor(),
#     T.Normalize(mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225])
# ])

# detect = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
# detect.prepare(ctx_id=0)

# def crop_face(frame_bgr):
#     faces = detect.get(frame_bgr)
#     if faces:
#         face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
#         x1, y1, x2, y2 = list(map(int, face.bbox))
#         h, w = frame_bgr.shape[:2]
#         pad = 20
#         x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
#         x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
#         return frame_bgr[y1:y2, x1:x2]
#     return frame_bgr

def extract_audio_frames(video_path, cap):
    audio_path = str(video_path) + ".wav"
    clip = VideoFileClip(str(video_path))
    clip.audio.write_audiofile(audio_path, fps=16000, verbose=False, logger=None)
    wav_np, sr = sf.read(audio_path)
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=1)
    wav_torch = torch.tensor(wav_np, dtype=torch.float32).unsqueeze(0)

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_samples_per_frame = int(sr * (1.0 / fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    audio_frames = []
    for i in range(total_frames):
        start = i * n_samples_per_frame
        end = start + n_samples_per_frame * 2
        seg = wav_torch[..., start:end] if end <= wav_torch.shape[-1] else wav_torch[..., start:]
        if seg.numel() < sr:
            seg = torch.nn.functional.pad(seg, (0, sr - seg.shape[-1]))
        audio_frames.append(seg)
    return audio_frames

# ----- Model loader -----

def get_models(cfg, device, fusion_mode):
    img_net = ImageMobileNetV2(pretrained=False, freeze_backbone=False).to(device).eval()
    img_net.load_state_dict(torch.load(cfg.checkpoint_dir / "mobilenet_img.pth", map_location=device))

    aud_net = AudioMobileNetV2(pretrained=False, freeze_backbone=False).to(device).eval()
    aud_net.load_state_dict(torch.load(cfg.checkpoint_dir / "mobilenet_aud.pth", map_location=device))

    fusion = FusionAV(
        num_classes=len(cfg.class_names),
        fusion_mode=fusion_mode,
        latent_dim_audio=128,
        latent_dim_image=128,
        use_pre_softmax=(fusion_mode in {"mlp", "gate"})
    ).to(device).eval()
    fusion.load_state_dict(torch.load(cfg.checkpoint_dir / f"fusion_{fusion_mode}.pth", map_location=device))

    return img_net, aud_net, fusion

# ----- Fusion evaluation per mode -----

def run_fusion_test(cfg, video_path, fusion_mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_net, aud_net, fusion = get_models(cfg, device, fusion_mode)

    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"Cannot open video file: {video_path}"
    audio_frames = extract_audio_frames(video_path, cap)

    predictions = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        crop = crop_face(frame)
        img_t = tx_img(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)

        with torch.inference_mode():
            v_vec = img_net.extract_features(img_t)
            v_logits = img_net(img_t)
            v_prob = torch.softmax(v_logits, dim=1)

        if frame_idx < len(audio_frames):
            wav = audio_frames[frame_idx].squeeze(0)
            spec = wav_to_logmel(wav).to(device)
            with torch.inference_mode():
                a_vec = aud_net.extract_features(spec)
                a_logits = aud_net(spec)
                a_prob = torch.softmax(a_logits, dim=1)
        else:
            a_vec = torch.zeros((1, 128), device=device)
            a_prob = torch.ones((1, len(cfg.class_names)), device=device) / len(cfg.class_names)
            a_logits = a_prob if fusion.use_pre_softmax else None

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
        predictions.append(cfg.class_names[cls])
        frame_idx += 1

    cap.release()
    counter = Counter(predictions)
    print(f"\nFusion mode: {fusion_mode}")
    print("Class distribution:", dict(counter))
    return {fusion_mode: dict(counter)}

# ----- Entry Point -----

if __name__ == "__main__":
    cfg = Config()
    video_path = "4.mp4"
    all_results = {}
    for mode in ["avg", "prod", "gate", "mlp", "latent"]:
        result = run_fusion_test(cfg, video_path, fusion_mode=mode)
        all_results.update(result)

    with open("fusion_test_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved results to fusion_test_summary.json")
