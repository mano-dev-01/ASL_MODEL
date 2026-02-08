import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import joblib

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DATA_FILE = os.path.join(DATA_DIR, "dataset.csv")

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "model.pkl"

# Load dataset
data = pd.read_csv(DATA_FILE)

# Separate features and labels
X = data.drop("label", axis=1).values
y = data["label"].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", probability=True)
)

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
output_name = MODEL_NAME.strip()
if not output_name.lower().endswith(".pkl"):
    output_name += ".pkl"

output_path = os.path.join(MODEL_DIR, output_name)

joblib.dump(model, output_path)

print(f"\nModel saved as {output_path}")
