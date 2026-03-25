import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("4\creditcard.csv")

# Features
X = df.drop("Class", axis=1)
y = df["Class"]

# Split (keep fraud cases)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Isolation Forest
model = IsolationForest(contamination=0.001, random_state=42)
model.fit(X_train)

# Predict
pred = model.predict(X_test)

# Convert (-1 → fraud, 1 → normal)
pred = [1 if p == -1 else 0 for p in pred]

# Evaluation
print(classification_report(y_test, pred))

# Save
pickle.dump(model, open("fraud_model.pkl", "wb"))