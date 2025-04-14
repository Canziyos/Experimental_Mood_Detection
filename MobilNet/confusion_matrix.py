import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

df = pd.read_csv("predictions.csv")

label_map = {
    1: "Surprise",
    2: "Fear",
    3: "Disgust",
    4: "Happiness",
    5: "Sadness",
    6: "Anger",
    7: "Neutral"
}


true_labels = df["True Class"].map(label_map)
predicted_labels = df["Predicted Class"].map(label_map)
# true_labels = df["True Class"]
# predicted_labels = df["Predicted Class"]

labels = list(label_map.values())
cm = confusion_matrix(true_labels, predicted_labels, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix – Mood Classifier")
plt.tight_layout()
plt.show()
