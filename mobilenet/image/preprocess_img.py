#!/usr/bin/env python
"""
preprocess_img.py: Create X_img.npy / y_img.npy for MobileNetV2 pipeline (RGB version).

- Walks dataset/images/<class_name> folders from Config.
- Skips face detection and assumes input images are already faces.
- Handles RGB, grayscale, and 4-channel (RGBA) source files.
- Converts to RGB, resizes to 224x224, saves uint8 [0-255].
"""

import os, cv2, numpy as np
from pathlib import Path
from mobilenet.config import Config

# ----------------------------------------------------------#
def ensure_bgr(img):
    """Convert grayscale or BGRA images to 3channel BGR. Return None if invalid."""
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 3 and img.shape[2] == 3:
        return img
    return None

# --------#
def main():
    cfg = Config()
    img_dir  = cfg.image_dir
    save_x   = cfg.x_img_path
    save_y   = cfg.y_img_path
    out_size = (224, 224)

    save_x.parent.mkdir(parents=True, exist_ok=True)
    label_map = {name: idx for idx, name in enumerate(cfg.class_names)}

    X, y = [], []
    totals = dict(processed=0, saved=0, errors=0)

    print("=== Pre-processing (cropped faces) ===")
    print(f"Src  : {img_dir}")
    print(f"Dst  : {save_x.name}, {save_y.name}")
    print(f"Size : {out_size}")
    print("----------------------")

    for cls, idx in label_map.items():
        folder: Path = img_dir / cls
        if not folder.is_dir():
            print(f"[WARN] missing folder {folder}")
            continue

        print(f"* {cls:7s} --> label {idx}")
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            fpath = folder / fname
            totals["processed"] += 1
            try:
                img_bgr = cv2.imread(str(fpath), cv2.IMREAD_UNCHANGED)
                img_bgr = ensure_bgr(img_bgr)
                if img_bgr is None:
                    raise ValueError("Unreadable image or bad format")

                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, out_size, interpolation=cv2.INTER_AREA)

                X.append(img_resized.astype(np.uint8))
                y.append(idx)
                totals["saved"] += 1

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
    print(f"Saved {X.shape} --> {save_x.name}")
    print(f"Saved {y.shape} --> {save_y.name}")


if __name__ == "__main__":
    main()
