import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load and shuffle the original data
data = pd.read_csv("physio_labeled.csv")
shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Extract features and labels
X_both = shuffled[["heart_rate", "gsr"]]
y_raw = shuffled["label"]

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)

# Consistent test split
_, X_test, _, y_test = train_test_split(X_both, y, test_size=0.2, random_state=42)

# Combine for easy filtering
df_test = X_test.copy()
df_test["label"] = y_test

# Create class imbalance
imbalanced_df = pd.concat([
    df_test[df_test["label"] == 0].sample(frac=0.3, random_state=42),  # agitated (reduce)
    df_test[df_test["label"] == 1],                                    # calm (keep all)
    df_test[df_test["label"] == 2].sample(frac=0.1, random_state=42)   # depressed (heavily reduce)
])

# Shuffle the imbalanced test set
imbalanced_df = imbalanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Separate features and labels
X_imbalanced = imbalanced_df[["heart_rate", "gsr"]]
y_imbalanced = imbalanced_df["label"].to_numpy()

# Load models
model_hr = joblib.load("model_hr.pkl")
model_gsr = joblib.load("model_gsr.pkl")
model_both = joblib.load("model_both.pkl")

# Evaluate each model
acc_hr = model_hr.score(X_imbalanced[["heart_rate"]], y_imbalanced)
acc_gsr = model_gsr.score(X_imbalanced[["gsr"]], y_imbalanced)
acc_both = model_both.score(X_imbalanced, y_imbalanced)

# Decision fusion (confidence-based)
proba_hr = model_hr.predict_proba(X_imbalanced[["heart_rate"]])
proba_gsr = model_gsr.predict_proba(X_imbalanced[["gsr"]])

fused_preds = []
for i in range(len(proba_hr)):
    conf_hr = max(proba_hr[i])
    conf_gsr = max(proba_gsr[i])
    if conf_hr >= conf_gsr:
        fused_preds.append(np.argmax(proba_hr[i]))
    else:
        fused_preds.append(np.argmax(proba_gsr[i]))

acc_fused = accuracy_score(y_imbalanced, fused_preds)

# Print results
print(f"[IMBALANCE TEST] HR-only Accuracy: {acc_hr:.2f}")
print(f"[IMBALANCE TEST] GSR-only Accuracy: {acc_gsr:.2f}")
print(f"[IMBALANCE TEST] Feature Fusion Accuracy: {acc_both:.2f}")
print(f"[IMBALANCE TEST] Decision Fusion Accuracy: {acc_fused:.2f}")
