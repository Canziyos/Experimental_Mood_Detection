from deepface import DeepFace
import pandas as pd
import os

# Load original dataset
df = pd.read_csv("train_labels.csv")

# Create a list to store rows with elderly faces (age ≥ 47)
elderly_rows = []

for i in range(len(df)):
    img_name = df.loc[i, "image"]
    label = str(df.loc[i, "label"])
    img_path = os.path.join("DATASET", "train", label, img_name)

    # Print progress
    print(f"Analyzing {i+1}/{len(df)}: {img_path}")

    try:
        analysis = DeepFace.analyze(img_path=img_path, actions=["age"], enforce_detection=False)
        estimated_age = analysis[0]["age"]
        if estimated_age >= 43:
            elderly_rows.append(df.loc[i])
    except Exception as e:
        print(f"Skipping image {img_path}: {e}")

# Save filtered elderly.
elderly = pd.DataFrame(elderly_rows)
elderly.to_csv("train_elder.csv", index=False)
