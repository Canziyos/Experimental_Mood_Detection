#!/usr/bin/env python
"""
preprocess_img.py: Create X_img.npy / y_img.npy for MobileNetV2 pipeline (RGB version).

• Walks dataset/images/<class_name> folders from Config.
• Detects largest face with RetinaFace (InsightFace) by default; --mtcnn flag switches to MTCNN.
• Handles RGB, grayscale, and 4-channel (RGBA) source files.
• Falls back to full frame if no face found.
• Converts to RGB, resizes to 224×224, saves uint8 [0-255].
"""

import argparse, os, cv2, numpy as np, torch
from pathlib import Path
from PIL import Image
from config import Config

# --------------------------------#
def get_detector(use_mtcnn=False):
    if use_mtcnn:
        from facenet_pytorch import MTCNN
        return MTCNN(keep_all=False,
                     device='cuda' if torch.cuda.is_available() else 'cpu')
    else:
        from insightface.app import FaceAnalysis
        providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)
        return app

# ----------------------------------------------------------#
def ensure_bgr(img):
    """Convert grayscale or BGRA images to 3channel BGR."""
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

# --------#
def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--mtcnn", action="store_true",
                      help="use MTCNN instead of RetinaFace")
    args = argp.parse_args()

    cfg = Config()
    img_dir  = cfg.image_dir
    save_x   = cfg.x_img_path
    save_y   = cfg.y_img_path
    out_size = (224, 224)

    save_x.parent.mkdir(parents=True, exist_ok=True)
    detector = get_detector(args.mtcnn)
    label_map = {name: idx for idx, name in enumerate(cfg.class_names)}

    X, y = [], []
    totals = dict(processed=0, detected=0, fallback=0, errors=0)

    print("=== Pre-processing (RGB mode) ===")
    print(f"Src  : {img_dir}")
    print(f"Dst  : {save_x.name}, {save_y.name}")
    print(f"Size : {out_size}, detector={'MTCNN' if args.mtcnn else 'RetinaFace'}")
    print("----------------------")

    for cls, idx in label_map.items():
        folder: Path = img_dir / cls
        if not folder.is_dir():
            print(f"[WARN] missing folder {folder}")
            continue

        print(f"* {cls:7s} → label {idx}")
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            fpath = folder / fname
            totals["processed"] += 1
            try:
                img_bgr = cv2.imread(str(fpath), cv2.IMREAD_UNCHANGED)
                img_bgr = ensure_bgr(img_bgr)
                if img_bgr is None:
                    raise ValueError("cv2.imread failed")

                face_rgb = None
                if args.mtcnn:
                    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                    face = detector(pil)
                    if face is not None:
                        totals["detected"] += 1
                        face_rgb = face.permute(1, 2, 0).cpu().numpy()
                else:
                    faces = detector.get(img_bgr)
                    if faces:
                        totals["detected"] += 1
                        face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                        x1, y1, x2, y2 = map(int, face.bbox)
                        crop = img_bgr[y1:y2, x1:x2]
                        face_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                if face_rgb is None:
                    totals["fallback"] += 1
                    face_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                face_resized = cv2.resize(face_rgb, out_size, interpolation=cv2.INTER_AREA)
                X.append(face_resized.astype(np.uint8))
                y.append(idx)

            except Exception as e:
                totals["errors"] += 1
                print(f"  ! {fname}: {e}")

    print("\n--- Summary ---")
    for k, v in totals.items():
        print(f"{k:10s}: {v}")

    X = np.array(X, dtype=np.uint8).transpose(0, 3, 1, 2)  # (N,3,224,224)
    y = np.array(y, dtype=np.int64)
    np.save(save_x, X)
    np.save(save_y, y)
    print(f"Saved {X.shape} → {save_x.name}")
    print(f"Saved {y.shape} → {save_y.name}")


if __name__ == "__main__":
    main()
