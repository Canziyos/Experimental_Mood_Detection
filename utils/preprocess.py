import os
import pandas as pd
from utils import process_audio_file

audio_dir = "audio_raw"
features_dir = "features"
metadata_path = "metadata.csv"

entries = []

for filename in os.listdir(audio_dir):
    if filename.endswith(".wav"):
        filepath = os.path.join(audio_dir, filename)
        try:
            entry = process_audio_file(filepath, features_dir)
            entries.append(entry)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

df = pd.DataFrame(entries)
df.to_csv(metadata_path, index=False)
print(f"Saved metadata to {metadata_path}")
