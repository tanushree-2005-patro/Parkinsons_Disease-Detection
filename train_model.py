# ------------------ train_model.py ------------------

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ------------------ Load Dataset ------------------
df = pd.read_csv("parkinsons.csv")   # Parkinson dataset

# Drop name column if exists
if "name" in df.columns:
    df = df.drop("name", axis=1)

# Features & target
X = df.drop("status", axis=1)
y = df["status"]

# ------------------ Scaling ------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------ Train Test Split ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ------------------ RandomForest Model ------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)

model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# ------------------ Save Model & Scaler ------------------
joblib.dump(model, "parkinson_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and Scaler Saved Successfully!")

# ------------------ Feature Importance Graph ------------------

# Feature names (22)
features = list(X.columns)
importances = model.feature_importances_

indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 8))
plt.barh(np.array(features)[indices], importances[indices])
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("RandomForest Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("Feature importance graph saved as feature_importance.png")

plt.show()