import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

metadata_path = "metadata.csv"
features_dir = "features"

# Load metadata
df = pd.read_csv(metadata_path)

X = []
y = []

for _, row in df.iterrows():
    mfcc_path = os.path.join(features_dir, row["mfcc_file"])
    if os.path.exists(mfcc_path):
        mfcc = np.load(mfcc_path)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        features = np.concatenate([mfcc_mean, mfcc_std])
        X.append(features)
        y.append(row["emotion"])

X = np.array(X)
y = np.array(y)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
