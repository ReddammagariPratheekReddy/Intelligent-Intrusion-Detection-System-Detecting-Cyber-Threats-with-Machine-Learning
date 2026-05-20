# train_model.py

import os
import pandas as pd
import matplotlib.pyplot as plt  # Make sure matplotlib is installed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -------------------------------
# File path to dataset
# -------------------------------
data_path = os.path.join(os.path.dirname(__file__), '../data/KDDTrain+.txt')

# -------------------------------
# Load dataset
# -------------------------------
try:
    data = pd.read_csv(data_path, header=None)
    print(f"✅ Dataset loaded successfully! Shape: {data.shape}")
except FileNotFoundError:
    print(f"❌ File not found at {data_path}. Please check the path.")
    exit()

# -------------------------------
# Basic preprocessing
# -------------------------------
# Example: Encode categorical features if any (adjust columns as needed)
categorical_cols = [1, 2, 3]  # Adjust based on your dataset columns
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Separate features and labels
X = data.iloc[:, :-1]  # All columns except last
y = data.iloc[:, -1]   # Last column as target

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Train model
# -------------------------------
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# -------------------------------
# Evaluate model
# -------------------------------
y_pred = clf.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

# -------------------------------
# Optional: Plot feature importance
# -------------------------------
plt.figure(figsize=(10, 6))
plt.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
plt.title("Feature Importances")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.show()

