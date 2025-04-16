import pandas as pd
import shutil
import os

# Load the CSV file containing elderly face image info
df = pd.read_csv("train_elder.csv")

# Output directory for elderly-only images
output_root = "Dataset_elder"
os.makedirs(output_root, exist_ok=True)

for i in range(len(df)):
    # Get image filename and label from the DataFrame
    img_name = df.loc[i, "image"]
    label = str(df.loc[i, "label"])

    # Construct source path: where the image currently is.
    src_path = os.path.join("DATASET", "train", label, img_name)

    # Construct destination path: where we want to copy it
    dest_dir = os.path.join(output_root, "train", label)
    os.makedirs(dest_dir, exist_ok=True)

    # Copy image to the new directory.
    shutil.copy2(src_path, dest_dir)
