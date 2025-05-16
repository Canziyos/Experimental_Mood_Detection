"""
Unified multimodal emotion inference: run live (webcam/mic) or offline (video file).
Usage:
    python multimodal_infer.py --online
    python multimodal_infer.py --video path/to/clip.mp4
"""

import time, threading, queue, signal, cv2, torch, numpy as np, sounddevice as sd, torchaudio, argparse
import torchvision.transforms as T
from pathlib import Path
from insightface.app import FaceAnalysis
from models.mobilenet_v2_embed import MobileNetV2Encap
from models.mobilenet import MobileNetV2Audio
from models.FusionAV import FusionAV
from config import Config
import soundfile as sf




mode = "offline"  # "online" or "offline"

vedio_path = r"mobilenet/16.mp4"
# ------------------------------- Config and Setup ------------------------------- #
mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=16_000, n_fft=512, hop_length=160,
        n_mels=64, f_min=50, f_max=8000)


def wav_to_logmel(wav: torch.Tensor) -> torch.Tensor:
    m = mel(wav)
    logmel = torch.log1p(m)
    if logmel.size(1) < 200:
        logmel = torch.nn.functional.pad(logmel, (0, 200-logmel.size(1)))
    logmel = logmel[:, :200]
    logmel = torch.nn.functional.interpolate(
                logmel.unsqueeze(0), size=(96,192), mode="bilinear", align_corners=False)
    return logmel.unsqueeze(0)




detect = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
detect.prepare(ctx_id=0)
prev_box, _miss = None, 0

def crop_face(frame_bgr):
    global prev_box, _miss
    h, w = frame_bgr.shape[:2]
    if prev_box is not None and _miss < 10:
        x1,y1,x2,y2 = prev_box
        pad = 20
        x1,y1 = max(0,x1-pad), max(0,y1-pad)
        x2,y2 = min(w,x2+pad), min(h,y2+pad)
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size:
            _miss += 1
            return roi
    faces = detect.get(frame_bgr)
    if faces:
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        prev_box = list(map(int, face.bbox))
        _miss = 0
        x1,y1,x2,y2 = prev_box
        return frame_bgr[y1:y2, x1:x2]
    return frame_bgr

tx_img = T.Compose([
    T.ToPILImage(),
    T.Resize(224),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])

def get_models(cfg, device):
    ckpt_img = cfg.checkpoint_dir / "mobilenet_img.pth"
    ckpt_aud = cfg.checkpoint_dir / "mobilenet_aud.pth"
    ckpt_fusion = cfg.checkpoint_dir / "fusion_latent.pth"

    img_net = MobileNetV2Encap(pretrained=False).to(device).eval()
    img_net.load_state_dict(torch.load(ckpt_img, map_location=device))

    aud_net = MobileNetV2Audio().to(device).eval()
    aud_net.load_state_dict(torch.load(ckpt_aud, map_location=device))

    fusion = FusionAV(num_classes=len(cfg.class_names),
                      fusion_mode="latent",
                      latent_dim_audio=128,
                      latent_dim_image=128).to(device).eval()
    fusion.load_state_dict(torch.load(ckpt_fusion, map_location=device))

    return img_net, aud_net, fusion

# =========== Online (Webcam/Mic) Inference ============= #
def run_online(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_net, aud_net, fusion = get_models(cfg, device)

    audio_q: queue.Queue = queue.Queue(maxsize=4)
    buf = np.zeros(16_000*2, dtype=np.float32)

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

    running = threading.Event(); running.set()
    th = threading.Thread(target=audio_loop, daemon=True); th.start()
    cap = cv2.VideoCapture(0)

    def sig_handler(sig, frame):  # graceful Ctrl-C
        running.clear()
    signal.signal(signal.SIGINT, sig_handler)

    print("Press Ctrl-C to stop.")
    while running.is_set():
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()
        crop = crop_face(frame)
        img_t = tx_img(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device, non_blocking=True)
        with torch.inference_mode(), torch.cuda.amp.autocast():
            v_vec = img_net.extract_features(img_t)
            v_prob = torch.softmax(img_net(img_t), 1)

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

# ------------------------ Offline (Video File) Inference ------------------------- #
def run_offline(cfg, video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_net, aud_net, fusion = get_models(cfg, device)

    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"Cannot open video file: {video_path}"

    # If audio track is available, extract and process
    audio_path = str(video_path) + ".wav"
    try:
        import moviepy.editor as mp
        clip = mp.VideoFileClip(str(video_path))
        clip.audio.write_audiofile(audio_path, fps=16000, verbose=False, logger=None)
        wav_np, sr = sf.read(audio_path)
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=1)  # Convert to mono
        wav_torch = torch.tensor(wav_np, dtype=torch.float32).unsqueeze(0).to(device)
        # Segment audio into logmel for each frame (simple alignment)
        audio_frames = []
        n_samples_per_frame = int(sr * (1.0 / cap.get(cv2.CAP_PROP_FPS)))
        for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            start = i * n_samples_per_frame
            end = start + n_samples_per_frame * 2
            seg = wav_torch[..., start:end] if end <= wav_torch.shape[-1] else wav_torch[..., start:]
            if seg.numel() < sr:  # At least 1s of audio
                seg = torch.nn.functional.pad(seg, (0, sr - seg.shape[-1]))
            audio_frames.append(seg)
    except Exception as e:
        print(f"[WARN] Could not extract/process audio: {e}")
        audio_frames = None

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()
        crop = crop_face(frame)
        img_t = tx_img(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device, non_blocking=True)
        with torch.inference_mode(), torch.cuda.amp.autocast():
            v_vec = img_net.extract_features(img_t)
            v_prob = torch.softmax(img_net(img_t), 1)

        # Audio for this frame
        if audio_frames is not None and frame_idx < len(audio_frames):
            wav = audio_frames[frame_idx].squeeze(0)
            spec = wav_to_logmel(wav)
            with torch.inference_mode():
                a_vec = aud_net.extract_features(spec)
                a_prob = torch.softmax(aud_net(spec), 1)
        else:
            a_vec = torch.zeros((1,128), device=device)
            a_prob = torch.ones((1,len(cfg.class_names)), device=device)/len(cfg.class_names)

        with torch.inference_mode():
            fused = fusion.fuse_probs(probs_audio=a_prob,
                                       probs_image=v_prob,
                                       latent_audio=a_vec,
                                       latent_image=v_vec)
        cls = fused.argmax(1).item()
        print(f"{cfg.class_names[cls]:7s}  {(time.time()-t0)*1e3:5.1f} ms")
        frame_idx += 1

    cap.release()

# ------------------------------------ MAIN -------------------------------------- #
# if __name__ == "__main__":

#     cfg = Config()

#     if args.video:
#         run_offline(cfg, args.video)
#     else:
#         run_online(cfg)
