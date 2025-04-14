import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_csv("physio_labeled.csv")
shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Extract features and labels
X_both = shuffled[["heart_rate", "gsr"]]
y_raw = shuffled["label"]

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)

# Consistent split for test set
_, X_test, _, y_test = train_test_split(X_both, y, test_size=0.2, random_state=42)

# Add Gaussian noise to GSR
X_noisy_gsr = X_test.copy()
X_noisy_gsr["gsr"] += np.random.normal(0, 2.5, size=X_noisy_gsr.shape[0])

# Save noisy test set (optional)
X_noisy_gsr.to_csv("X_test_noisyGSR.csv", index=False)
pd.DataFrame(y_test, columns=["label"]).to_csv("y_test.csv", index=False)

# Load models
model_hr = joblib.load("model_hr.pkl")
model_gsr = joblib.load("model_gsr.pkl")
model_both = joblib.load("model_both.pkl")

# Evaluate models
acc_hr = model_hr.score(X_noisy_gsr[["heart_rate"]], y_test)
acc_gsr = model_gsr.score(X_noisy_gsr[["gsr"]], y_test)
acc_both = model_both.score(X_noisy_gsr, y_test)

# Decision fusion (confidence-based)
proba_hr = model_hr.predict_proba(X_noisy_gsr[["heart_rate"]])
proba_gsr = model_gsr.predict_proba(X_noisy_gsr[["gsr"]])

fused_preds = []
for i in range(len(proba_hr)):
    conf_hr = max(proba_hr[i])
    conf_gsr = max(proba_gsr[i])
    if conf_hr >= conf_gsr:
        fused_preds.append(np.argmax(proba_hr[i]))
    else:
        fused_preds.append(np.argmax(proba_gsr[i]))

acc_fused = accuracy_score(y_test, fused_preds)

# Output results
print(f"[NOISE GSR TEST] HR-only Accuracy: {acc_hr:.2f}")
print(f"[NOISE GSR TEST] GSR-only Accuracy: {acc_gsr:.2f}")
print(f"[NOISE GSR TEST] Feature Fusion Accuracy: {acc_both:.2f}")
print(f"[NOISE GSR TEST] Decision Fusion Accuracy: {acc_fused:.2f}")
