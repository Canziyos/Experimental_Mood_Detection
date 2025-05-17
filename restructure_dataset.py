# import shutil, random
# from pathlib import Path
# from sklearn.model_selection import train_test_split

# # === Paths ===
# audio_root = Path("dataset/audio")
# source_dirs = [d for d in audio_root.iterdir() if d.is_dir() and d.name not in ("train", "val", "test")]
# target_dirs = {
#     "train": audio_root / "train",
#     "val": audio_root / "val",
#     "test": audio_root / "test"
# }

# # === Create target folders ===
# for split in target_dirs.values():
#     for cls_dir in source_dirs:
#         (split / cls_dir.name).mkdir(parents=True, exist_ok=True)

# # === Config ===
# val_size = 0.10
# test_size = 0.10
# seed = 42
# random.seed(seed)

# # === Split and move files ===
# for cls_dir in source_dirs:
#     wavs = list(cls_dir.glob("*.wav"))
#     labels = [cls_dir.name] * len(wavs)

#     # Initial split: train vs val+test
#     train_files, temp_files = train_test_split(wavs, test_size=val_size+test_size, stratify=labels, random_state=seed)

#     # Then split val+test evenly
#     rel_val = val_size / (val_size + test_size)
#     val_files, test_files = train_test_split(temp_files, test_size=1-rel_val, stratify=[cls_dir.name]*len(temp_files), random_state=seed)

#     for f in train_files:
#         shutil.copy(str(f), str(target_dirs["train"] / cls_dir.name / f.name))
#     for f in val_files:
#         shutil.copy(str(f), str(target_dirs["val"] / cls_dir.name / f.name))
#     for f in test_files:
#         shutil.copy(str(f), str(target_dirs["test"] / cls_dir.name / f.name))

# print("[INFO] Dataset split complete. New structure:")
# print("- train/", sum(len(list((target_dirs['train'] / d.name).glob('*.wav'))) for d in source_dirs), "files")
# print("- val/  ", sum(len(list((target_dirs['val'] / d.name).glob('*.wav'))) for d in source_dirs), "files")
# print("- test/ ", sum(len(list((target_dirs['test'] / d.name).glob('*.wav'))) for d in source_dirs), "files")

# import shutil, random
# from pathlib import Path
# from sklearn.model_selection import train_test_split

# # === Paths ===
# image_root = Path("dataset/images")
# source_dirs = [d for d in image_root.iterdir() if d.is_dir() and d.name not in ("train", "val", "test")]
# target_dirs = {
#     "train": image_root / "train",
#     "val": image_root / "val",
#     "test": image_root / "test"
# }

# # === Create target folders ===
# for split in target_dirs.values():
#     for cls_dir in source_dirs:
#         (split / cls_dir.name).mkdir(parents=True, exist_ok=True)

# # === Config ===
# val_size = 0.10
# test_size = 0.10
# seed = 42
# random.seed(seed)

# # === Split and copy files ===
# for cls_dir in source_dirs:
#     imgs = list(cls_dir.glob("*.*"))  # handles .png, .jpg, etc.
#     labels = [cls_dir.name] * len(imgs)

#     train_files, temp_files = train_test_split(imgs, test_size=val_size+test_size, stratify=labels, random_state=seed)
#     rel_val = val_size / (val_size + test_size)
#     val_files, test_files = train_test_split(temp_files, test_size=1-rel_val, stratify=[cls_dir.name]*len(temp_files), random_state=seed)

#     for f in train_files:
#         shutil.copy(str(f), str(target_dirs["train"] / cls_dir.name / f.name))
#     for f in val_files:
#         shutil.copy(str(f), str(target_dirs["val"] / cls_dir.name / f.name))
#     for f in test_files:
#         shutil.copy(str(f), str(target_dirs["test"] / cls_dir.name / f.name))

# print("[INFO] Image dataset split complete. New structure:")
# print("- train/", sum(len(list((target_dirs['train'] / d.name).glob('*'))) for d in source_dirs), "files")
# print("- val/  ", sum(len(list((target_dirs['val'] / d.name).glob('*'))) for d in source_dirs), "files")
# print("- test/ ", sum(len(list((target_dirs['test'] / d.name).glob('*'))) for d in source_dirs), "files")

