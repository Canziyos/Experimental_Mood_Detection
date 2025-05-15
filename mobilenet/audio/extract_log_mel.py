"""Offline extractor that walks through folders of WAVs and stores <file_list.npy> & <y.npy>."""
import argparse, json, torchaudio, numpy as np
from pathlib import Path
from config import Config

ap = argparse.ArgumentParser(); ap.add_argument("src", help="root dir with class subfolders")
ap.add_argument("--outfile", default="data")
args = ap.parse_args()

src = Path(args.src)
label_map = {d.name:i for i,d in enumerate(sorted(src.iterdir())) if d.is_dir()}
paths, labels = [], []
for cls, idx in label_map.items():
    for wav in (src/cls).rglob("*.wav"):
        paths.append(str(wav)); labels.append(idx)
paths = np.array(paths); labels = np.array(labels)
np.save(f"{args.outfile}_paths.npy", paths); np.save(f"{args.outfile}_y.npy", labels)
with open(f"{args.outfile}_labelmap.json","w") as f:
    json.dump(label_map, f, indent=2)
print(f"Saved {len(labels)} items → {args.outfile}_paths.npy / y.npy")