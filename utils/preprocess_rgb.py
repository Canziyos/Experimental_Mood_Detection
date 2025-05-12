import os
import numpy as np
import cv2
import torch
from insightface.app import FaceAnalysis

# === Config crap ===
img_dir = r"data\image"
save_X = "processed_data/X_img_rgb.npy"
save_y = "processed_data/y_img_rgb.npy"
target_size = (48, 48)

label_map = {
    "Angry": 0,
    "Disgust": 1,
    "Fear": 2,
    "Happy": 3,
    "Neutral": 4,
    "Sad": 5,
}

# === Load RetinaFace model ===
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(224, 224))

X, y = [], []

print("Starting RGB-only preprocessing with RetinaFace...")
print(f"Dataset root - {img_dir}")
print(f"Target size - {target_size}")
print(f"Device - {'cuda' if torch.cuda.is_available() else 'cpu'}")


total_processed = 0
total_detected = 0
total_fallbacks = 0
total_errors = 0

for label_name, label_idx in label_map.items():
    folder = os.path.join(img_dir, label_name)
    if not os.path.isdir(folder):
        print(f"Skipping missing folder - {folder}")
        continue

    print(f"Processing class - {label_name} (label {label_idx})")
    count = 0

    for fname in os.listdir(folder):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        fpath = os.path.join(folder, fname)
        count += 1
        total_processed += 1

        try:
            img_cv = cv2.imread(fpath)

            # If grayscale, convert to 3-channel BGR manually.
            if img_cv is not None and len(img_cv.shape) == 2:
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)

            if img_cv is None or img_cv.size == 0:
                print(f"Failed to load image - {fpath}")
                total_errors += 1
                continue

            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            faces = app.get(img_rgb)

            # aligned face if available, else fallback.
            if faces and hasattr(faces[0], "aligned") and faces[0].aligned is not None:
                face_rgb = faces[0].aligned
                if face_rgb is None or face_rgb.size == 0:
                    raise ValueError("Aligned face is empty.")
                total_detected += 1
            else:
                total_fallbacks += 1
                if img_rgb is None or img_rgb.size == 0:
                    raise ValueError("Fallback image is empty.")
                face_rgb = img_rgb

            face_resized = cv2.resize(face_rgb, target_size)
            face_resized = face_resized.astype("float32") / 255.0

            X.append(face_resized)
            y.append(label_idx)

            if total_processed % 50 == 0:
                print(f"{total_processed} images processed...")

        except Exception as e:
            total_errors += 1
            print(f"Error processing {fname} - {str(e)}")


print("\nSummary:")
print(f"Total images processed - {total_processed}")
print(f"Faces detected (aligned) - {total_detected}")
print(f"Fallbacks used - {total_fallbacks}")
print(f"Errors encountered - {total_errors}")
print("Converting to numpy arrays...")

# RGB → shape should be (B, 3, 48, 48)
X = np.array(X, dtype=np.float32).transpose(0, 3, 1, 2)  # From (B, H, W, 3) to (B, 3, H, W)
y = np.array(y)

np.save(save_X, X)
np.save(save_y, y)

print(f"Saved - {save_X} with shape {X.shape}.")
print(f"Saved - {save_y} with shape {y.shape}.")
