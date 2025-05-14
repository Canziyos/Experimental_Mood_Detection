# import os
# import numpy as np
# import cv2
# from PIL import Image
# from facenet_pytorch import MTCNN
# import torch
# from config import Config

# cfg = Config()

# # config paths
# img_dir = cfg.image_dir
# save_x = cfg.x_img_path
# save_y = cfg.y_img_path
# resize_to = (48, 48)

# save_x.parent.mkdir(parents=True, exist_ok=True)

# # Use class names from config to build label map
# label_map = {name: idx for idx, name in enumerate(cfg.class_names)}

# mtcnn = MTCNN(keep_all=False, device='cuda' if torch.cuda.is_available() else 'cpu')

# X, y = [], []

# print("Starting preprocessing pipeline...")
# print(f"Dataset path: {img_dir}")
# print(f"Saving to: {save_x}, {save_y}")
# print(f"Target size: {resize_to}")
# print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# # Check folders before starting.
# print("\nChecking expected class folders:")
# for name in cfg.class_names:
#     folder = img_dir / name
#     print(f"{folder} --> {'OK' if folder.exists() else 'MISSING'}")

# total_processed = 0
# total_detected = 0
# total_fallbacks = 0
# total_errors = 0

# # Loop through each class folder
# for label_name, label_idx in label_map.items():
#     folder = img_dir / label_name
#     if not folder.is_dir():
#         print(f"[Skip] Missing folder: {folder}")
#         continue

#     print(f"\nProcessing '{label_name}' --> class {label_idx}")
#     count = 0

#     for fname in os.listdir(folder):
#         if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
#             continue

#         fpath = folder / fname
#         count += 1
#         total_processed += 1

#         try:
#             img_cv = cv2.imread(str(fpath))
#             if img_cv is None:
#                 print(f"Failed to load image: {fpath}")
#                 total_errors += 1
#                 continue

#             img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
#             img_pil = Image.fromarray(img_rgb)

#             face = mtcnn(img_pil)
#             if face is not None:
#                 total_detected += 1
#                 face = face.permute(1, 2, 0).cpu().numpy()
#                 face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
#             else:
#                 total_fallbacks += 1
#                 face = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

#             face_resized = cv2.resize(face, resize_to)
#             face_resized = face_resized.astype("float32") / 255.0

#             X.append(face_resized)
#             y.append(label_idx)

#         except Exception as e:
#             total_errors += 1
#             print(f"Error processing {fname}: {str(e)}")

# print("\nSummary:")
# print(f"Total images processed : {total_processed}")
# print(f"Faces detected         : {total_detected}")
# print(f"Fallbacks used         : {total_fallbacks}")
# print(f"Errors encountered     : {total_errors}")

# print("Saving NumPy arrays...")
# X = np.array(X).reshape(-1, 1, 48, 48)
# y = np.array(y)

# np.save(save_x, X)
# np.save(save_y, y)

# print(f"Saved: {save_x} with shape {X.shape}")
# print(f"Saved: {save_y} with shape {y.shape}")

# import numpy as np
# import matplotlib.pyplot as plt
# from config import Config

# cfg = Config()

# # Load preprocessed images and labels
# X = np.load(cfg.x_img_path)
# y = np.load(cfg.y_img_path)

# print(f"Loaded {X.shape[0]} samples.")
# print(f"Image shape: {X.shape[1:]}")

# # Show 12 random samples with their class labels
# fig, axes = plt.subplots(3, 4, figsize=(12, 9))
# indices = np.random.choice(len(X), size=12, replace=False)

# for ax, idx in zip(axes.flatten(), indices):
#     img = X[idx][0]  # shape: [1, 48, 48] → take the 0-channel
#     label = cfg.class_names[y[idx]]
#     ax.imshow(img, cmap='gray')
#     ax.set_title(f"{label}", fontsize=10)
#     ax.axis('off')

# plt.tight_layout()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from config import Config

cfg = Config()
y = np.load(cfg.y_img_path)

# Count samples per class.
unique, counts = np.unique(y, return_counts=True)

# Map to class names.
class_names = cfg.class_names
label_counts = dict(zip([class_names[i] for i in unique], counts))


print("Class distribution:")
for label, count in label_counts.items():
    print(f"{label:<10}: {count} samples")


plt.bar(label_counts.keys(), label_counts.values())
plt.title("Image Dataset Class Distribution")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
