import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Load the shuffled, labeled data again (same one used for training)
data = pd.read_csv("physio_labeled.csv")
shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Recreate features and labels
X_both = shuffled[["heart_rate", "gsr"]]
y = shuffled["label"]

# Encode labels again in same order
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split the same way (ensure consistency)
_, X_test_both, _, y_test = train_test_split(X_both, y, test_size=0.2, random_state=42)

# Add Gaussian noise to HR (simulate sensor noise)
X_noisy_hr = X_test_both.copy()
X_noisy_hr["heart_rate"] += np.random.normal(0, 15, size=X_noisy_hr.shape[0])
# Save noisy test data and labels
X_noisy_hr.to_csv("X_test_noisyHR.csv", index=False)
pd.DataFrame(y_test, columns=["label"]).to_csv("y_test.csv", index=False)

# Load saved models
model_hr = joblib.load("model_hr.pkl")
model_gsr = joblib.load("model_gsr.pkl")
model_both = joblib.load("model_both.pkl")

# Predict and evaluate
acc_hr = model_hr.score(X_noisy_hr[["heart_rate"]], y_test)
acc_gsr = model_gsr.score(X_noisy_hr[["gsr"]], y_test)
acc_both = model_both.score(X_noisy_hr, y_test)

# Decision fusion (confidence-based)
proba_hr = model_hr.predict_proba(X_noisy_hr[["heart_rate"]])
proba_gsr = model_gsr.predict_proba(X_noisy_hr[["gsr"]])

fused_preds = []
for i in range(len(proba_hr)):
    conf_hr = max(proba_hr[i])
    conf_gsr = max(proba_gsr[i])
    if conf_hr >= conf_gsr:
        fused_preds.append(np.argmax(proba_hr[i]))
    else:
        fused_preds.append(np.argmax(proba_gsr[i]))

acc_fused = accuracy_score(y_test, fused_preds)

# Print results
print(f"[NOISE TEST] HR-only Accuracy: {acc_hr:.2f}")
print(f"[NOISE TEST] GSR-only Accuracy: {acc_gsr:.2f}")
print(f"[NOISE TEST] Feature Fusion Accuracy: {acc_both:.2f}")
print(f"[NOISE TEST] Decision Fusion Accuracy: {acc_fused:.2f}")
