#!/usr/bin/env python
"""
video_infer.py: End-to-end demo: video -> (image logits, audio logits, fusion).

Note-:
    pip install opencv-python moviepy librosa soundfile torchvision torchaudio.
"""

import argparse, cv2, os, torch, numpy as np
import torchvision.transforms as T
import librosa, soundfile as sf
from moviepy.editor import VideoFileClip
from pathlib import Path
from typing import List

from models.mobilenet_v2_embed import MobileNetV2Encap
from models.audio_cnn1d import AudioCNN1D
from config import Config

# -- Face detector (same as preprocess_img) --
from insightface.app import FaceAnalysis
_det = None
def crop_face(frame_bgr):
    global _det
    if _det is None:
        _det = FaceAnalysis(name="buffalo_l",
                            providers=['CPUExecutionProvider'])
        _det.prepare(ctx_id=-1)
    faces = _det.get(frame_bgr)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    x1,y1,x2,y2 = map(int, face.bbox)
    return frame_bgr[y1:y2, x1:x2]

# -- Image preprocessing --
_tx_eval = T.Compose([
    T.ToPILImage(),
    T.Resize(224),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])

def frames_to_tensors(frames_bgr: List[np.ndarray]) -> torch.Tensor:
    imgs = []
    for bgr in frames_bgr:
        crop = crop_face(bgr) or bgr
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        imgs.append(_tx_eval(gray))
    return torch.stack(imgs)   # (N,1,224,224)

# -- Audio feature extractor (placeholder) --
def audio_to_windows(audio_path: Path,
                     win_sec: float = 3.0,
                     sr: int = 16000) -> torch.Tensor:
    """
    3-second window --> 300 time-steps with 15 dims each (fake).
    will be replaced with actual 15*300 feature extractor.
    """
    wav, file_sr = sf.read(audio_path)
    if file_sr != sr:
        wav = librosa.resample(wav, file_sr, sr)
    win_len = int(sr * win_sec)
    feats = []
    for i in range(0, len(wav) - win_len + 1, win_len):
        seg = wav[i:i+win_len]
        # TODO: real feature extractor here.
        mel = librosa.feature.melspectrogram(seg, sr=sr, n_mels=15,
                                             hop_length=win_len//300,
                                             n_fft=512)          # (15, 300)
        feats.append(mel.astype(np.float32))
    return torch.tensor(np.stack(feats))   # (N,15,300)

# -- Aggregation helpers --
def mean_softmax(logits):
    probs = torch.softmax(logits, dim=1)
    return probs.mean(0)                   # (6,)

def majority_vote(preds):
    vals, counts = torch.unique(preds, return_counts=True)
    return vals[counts.argmax()].item()

# --Main routine--
def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("video", help="Path to video (with audio track)")
    argp.add_argument("--ckpt_img", required=True, help="MobileNet .pth")
    argp.add_argument("--ckpt_aud", required=True, help="AudioCNN1D .pth")
    args = argp.parse_args()

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    img_model = MobileNetV2Encap(pretrained=False).to(device)
    img_model.load_state_dict(torch.load(args.ckpt_img, map_location=device))
    img_model.eval()

    aud_model = AudioCNN1D(cfg.input_channels, cfg.input_length).to(device)
    aud_model.load_state_dict(torch.load(args.ckpt_aud, map_location=device))
    aud_model.eval()

    # --- Extract frames & audio ---
    clip = VideoFileClip(args.video)
    frames = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
              for f in clip.iter_frames(fps=2)]      # 2 FPS sampling.
    audio_tmp = Path("tmp_audio.wav")
    clip.audio.write_audiofile(audio_tmp.as_posix(),
                               fps=16000, verbose=False, logger=None)

    # Image branch --
    img_batch = frames_to_tensors(frames).to(device)         # (N,1,224,224)
    with torch.no_grad():
        img_logits  = img_model(img_batch)
        img_latent  = img_model.extract_features(img_batch).mean(0)   # (128,)
    img_prob   = mean_softmax(img_logits)
    img_pred   = img_prob.argmax().item()

    # Audio branch --
    aud_batch = audio_to_windows(audio_tmp).to(device)       # (M,15,300)
    with torch.no_grad():
        aud_logits = aud_model(aud_batch)
        aud_latent = aud_model.extract_latent_vector(aud_batch).mean(0) # (512,)
    aud_prob  = mean_softmax(aud_logits)
    aud_pred  = aud_prob.argmax().item()

    # Early‑fusion example --
    fused_vec = torch.cat([img_latent, aud_latent], dim=0)   # (640,)
    # Simple logistic fusion (learn this offline):
    fusion_w = torch.randn(6, 640) * 0.01
    fusion_b = torch.zeros(6)
    fuse_logits = F.linear(fused_vec.unsqueeze(0), fusion_w, fusion_b)
    fuse_pred = fuse_logits.argmax(1).item()

    # --- Output ---
    names = cfg.class_names
    print(f"Image branch -> {names[img_pred]} (p={img_prob[img_pred]:.2f})")
    print(f"Audio branch -> {names[aud_pred]} (p={aud_prob[aud_pred]:.2f})")
    print(f"Fusion       -> {names[fuse_pred]}  (toy FC weights)")

    # cleanup tmp
    audio_tmp.unlink()

if __name__ == "__main__":
    main()
