import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

# --------- CONFIGURATION ---------
model_path = "mobilenet_augmented.pth"
test_dir = "DATASET/test"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_map = {
    "1": "Surprise",
    "2": "Fear",
    "3": "Disgust",
    "4": "Happiness",
    "5": "Sadness",
    "6": "Anger",
    "7": "Neutral"
}
# ----------------------------------

# --------- TRANSFORM ---------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
# ------------------------------

# --------- LOAD MODEL ---------
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 7)

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Hook to extract feature vector
def get_feature_hook(module, input, output):
    global features
    features = output.squeeze().detach().cpu().numpy()

hook = model.features.register_forward_hook(get_feature_hook)
# ------------------------------

# --------- FEATURE EXTRACTION ---------
all_images = glob(os.path.join(test_dir, "*", "*.jpg"))
data = []

for path in all_images:
    img = Image.open(path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    label_num = os.path.basename(os.path.dirname(path))
    label = label_map.get(label_num, "Unknown")

    _ = model(img_tensor)

    data.append((features, label))

hook.remove()

X = np.array([f for f, _ in data])
y = [l for _, l in data]
df = pd.DataFrame(X)
df['label'] = y
# --------------------------------------

# --------- t-SNE REDUCTION ---------
print("Reducing dimensions with t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_2D = tsne.fit_transform(X)

df['x'] = X_2D[:, 0]
df['y'] = X_2D[:, 1]
# ----------------------------------

# --------- PLOTTING ---------
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x="x", y="y", hue="label", palette="deep", s=60, alpha=0.8)
plt.title("t-SNE clustering of MobileNet features")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend(title="Emotion")
plt.tight_layout()
plt.savefig("mobilenet_feature_clusters.png")
plt.show()
# ----------------------------

# --------- SAVE FEATURE CSV ---------
df.to_csv("mobilenet_feature_vectors.csv", index=False)
print("t-SNE plot and feature vectors saved.")
