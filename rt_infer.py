#!/usr/bin/env python3
"""
Real-time multimodal emotion inference.

  python rt_infer.py --ckpt_img checkpoints/mobilenet_img.pth \
                     --ckpt_aud checkpoints/mobilenet_aud.pth \
                     --ckpt_fusion checkpoints/fusion_latent.pth
"""

import argparse, time, threading, queue, signal, cv2, torch, numpy as np, sounddevice as sd, torchaudio
import torchvision.transforms as T
from pathlib import Path
from insightface.app import FaceAnalysis

from models.mobilenet_v2_embed import MobileNetV2Encap
from models.mobilenet_v2_audio import MobileNetV2Audio      # <- MobileNetV2Encap with in_ch=1
from models.FusionAV import FusionAV
from config import Config

# ------------------------------------------------------------------ #
# Audio preprocessing
_mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=16_000, n_fft=512, hop_length=160,
        n_mels=64, f_min=50, f_max=8000)

def wav_to_logmel(wav: torch.Tensor) -> torch.Tensor:
    mel = _mel(wav)
    logmel = torch.log1p(mel)                        # (64,T)
    if logmel.size(1) < 200:
        logmel = torch.nn.functional.pad(logmel, (0, 200-logmel.size(1)))
    logmel = logmel[:, :200]                         # (64,200)
    logmel = torch.nn.functional.interpolate(
                logmel.unsqueeze(0), size=(96,192), mode="bilinear", align_corners=False)
    return logmel.unsqueeze(0)                       # (1,1,96,192)

# ------------------------------------------------------------------ #
# Face detector with caching
_det = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
_det.prepare(ctx_id=0)
_prev_box, _miss = None, 0

def crop_face(frame_bgr):
    global _prev_box, _miss
    h, w = frame_bgr.shape[:2]
    if _prev_box is not None and _miss < 10:
        x1,y1,x2,y2 = _prev_box
        pad = 20
        x1,y1 = max(0,x1-pad), max(0,y1-pad)
        x2,y2 = min(w,x2+pad), min(h,y2+pad)
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size:
            _miss += 1
            return roi
    faces = _det.get(frame_bgr)
    if faces:
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        _prev_box = list(map(int, face.bbox))
        _miss = 0
        x1,y1,x2,y2 = _prev_box
        return frame_bgr[y1:y2, x1:x2]
    return frame_bgr

_tx_img = T.Compose([
    T.ToPILImage(),
    T.Resize(224),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])

# ------------------------------------------------------------------ #
def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--ckpt_img", required=True)
    argp.add_argument("--ckpt_aud", required=True)
    argp.add_argument("--ckpt_fusion", required=True)
    args = argp.parse_args()

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    img_net = MobileNetV2Encap(pretrained=False).to(device).eval()
    img_net.load_state_dict(torch.load(args.ckpt_img, map_location=device))
    aud_net = MobileNetV2Audio().to(device).eval()
    aud_net.load_state_dict(torch.load(args.ckpt_aud, map_location=device))

    fusion = FusionAV(num_classes=len(cfg.class_names),
                      fusion_mode="latent",
                      latent_dim_audio=128,
                      latent_dim_image=128).to(device).eval()
    fusion.load_state_dict(torch.load(args.ckpt_fusion, map_location=device))

    # ------------------------------------------------------------------ #
    # Audio thread
    audio_q: queue.Queue = queue.Queue(maxsize=4)
    buf = np.zeros(16_000*2, dtype=np.float32)       # 2 s ring buffer

    def audio_loop():
        with sd.InputStream(channels=1, samplerate=16_000, blocksize=1600) as stream:
            while running.is_set():
                block, _ = stream.read(1600)
                np.roll(buf, -1600); buf[-1600:] = block[:,0]
                if stream.time < 2.0:
                    continue
                wav = torch.tensor(buf.copy(), dtype=torch.float32, device=device)
                with torch.inference_mode(), torch.cuda.amp.autocast():
                    spec = wav_to_logmel(wav).to(device, non_blocking=True)
                    a_vec = aud_net.extract_features(spec)
                    a_prob = torch.softmax(aud_net(spec), 1)
                try:
                    audio_q.put_nowait((a_vec, a_prob))
                except queue.Full:
                    pass

    # ------------------------------------------------------------------ #
    # Start threads and video capture
    running = threading.Event(); running.set()
    th = threading.Thread(target=audio_loop, daemon=True); th.start()
    cap = cv2.VideoCapture(0)

    def sig_handler(sig, frame):     # graceful Ctrl-C
        running.clear()
    signal.signal(signal.SIGINT, sig_handler)

    print("Press Ctrl-C to stop.")
    while running.is_set():
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()
        crop = crop_face(frame)
        img_t = _tx_img(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device, non_blocking=True)
        with torch.inference_mode(), torch.cuda.amp.autocast():
            v_vec = img_net.extract_features(img_t)
            v_prob = torch.softmax(img_net(img_t), 1)

        # get most recent audio packet
        try:
            while audio_q.qsize() > 1:
                audio_q.get_nowait()
            a_vec, a_prob = audio_q.get_nowait()
        except queue.Empty:
            continue

        with torch.inference_mode():
            fused = fusion.fuse_probs(probs_audio=a_prob,
                                       probs_image=v_prob,
                                       latent_audio=a_vec,
                                       latent_image=v_vec)
        cls = fused.argmax(1).item()
        print(f"{cfg.class_names[cls]:7s}  {(time.time()-t0)*1e3:5.1f} ms")

    cap.release()

if __name__ == "__main__":
    main()
