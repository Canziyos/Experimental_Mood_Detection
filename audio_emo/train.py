import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import joblib

# Load and shuffle the dataset
data = pd.read_csv("physio_labeled.csv")
shuffled = data.sample(frac=1).reset_index(drop=True)

# Extract and encode labels
y_raw = shuffled["label"]
encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)

# Define input features
X_hr = shuffled[["heart_rate"]]
X_gsr = shuffled[["gsr"]]
X_both = shuffled[["heart_rate", "gsr"]]

# HR-only model
x_train_hr, x_test_hr, y_train_hr, y_test_hr = train_test_split(X_hr, y, test_size=0.2, random_state=42)
model_hr = lr(max_iter=1000)
model_hr.fit(x_train_hr, y_train_hr)
acc_hr = model_hr.score(x_test_hr, y_test_hr)
print(f"Accuracy (HR only): {acc_hr:.2f}")

# GSR-only model
X_train_gsr, X_test_gsr, y_train_gsr, y_test_gsr = train_test_split(X_gsr, y, test_size=0.2, random_state=42)
model_gsr = lr(max_iter=1000)
model_gsr.fit(X_train_gsr, y_train_gsr)
acc_gsr = model_gsr.score(X_test_gsr, y_test_gsr)
print(f"Accuracy (GSR only): {acc_gsr:.2f}")

# Feature-level fusion (HR + GSR)
X_train_both, X_test_both, y_train_both, y_test_both = train_test_split(X_both, y, test_size=0.2, random_state=42)
model_both = lr(max_iter=1000)
model_both.fit(X_train_both, y_train_both)
acc_both = model_both.score(X_test_both, y_test_both)
print(f"Accuracy (Both HR and GSR): {acc_both:.2f}")

# Decision-level fusion: shared test set
X_shared_train, X_shared_test, y_shared_train, y_shared_test = train_test_split(X_both, y, test_size=0.2, random_state=42)

proba_hr = model_hr.predict_proba(X_shared_test[["heart_rate"]])
proba_gsr = model_gsr.predict_proba(X_shared_test[["gsr"]])


fused_preds = []

for i in range(len(proba_hr)):
    conf_hr = max(proba_hr[i])
    conf_gsr = max(proba_gsr[i])

    if conf_hr >= conf_gsr:
        fused_preds.append(np.argmax(proba_hr[i]))  # pick HR's prediction
    else:
        fused_preds.append(np.argmax(proba_gsr[i]))  # pick GSR's prediction


# Evaluate fused model
acc_fused = accuracy_score(y_shared_test, fused_preds)
print(f"Accuracy (Decision-Level Fusion): {acc_fused:.2f}")
joblib.dump(model_hr, "model_hr.pkl")
joblib.dump(model_gsr, "model_gsr.pkl")
joblib.dump(model_both, "model_both.pkl")
