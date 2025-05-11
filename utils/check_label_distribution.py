import numpy as np

# Load the label array
y = np.load("./processed_data/y_aug.npy")

# Same label map you used when building the dataset
label_map = {
    0: "Angery",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad"
}

# Count the number of occurrences of each label
unique, counts = np.unique(y, return_counts=True)

print("Label distribution in y_aug.npy:")
for label_id, count in zip(unique, counts):
    label_name = label_map.get(label_id, f"Unknown({label_id})")
    print(f"  {label_name:<8} (label {label_id}): {count} samples")

print(f"\nTotal samples: {len(y)}")
