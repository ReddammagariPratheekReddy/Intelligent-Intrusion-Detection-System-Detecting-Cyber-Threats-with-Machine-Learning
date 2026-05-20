import requests
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------
# Load preprocessed test dataset
# ---------------------------
TEST_CSV_PATH = "/Users/pratheek/Desktop/Major Project/data/test_preprocessed.csv"
test_df = pd.read_csv(TEST_CSV_PATH)

# Separate features and true labels
features = test_df.iloc[:, :-1].values.tolist()
true_labels = test_df.iloc[:, -1].tolist()

# ---------------------------
# Send batch request to backend
# ---------------------------
url = "http://127.0.0.1:8000/predict-batch"

try:
    response = requests.post(url, json={"features_list": features})
    response.raise_for_status()  # raise error if request failed
except requests.exceptions.RequestException as e:
    print(f"❌ Failed to connect to backend: {e}")
    exit()

# Get predicted labels
pred_labels = response.json().get("predictions", [])
if not pred_labels:
    print("❌ No predictions returned from backend.")
    exit()

# ---------------------------
# Compute metrics
# ---------------------------
# Overall accuracy
correct = sum([1 for t, p in zip(true_labels, pred_labels) if t == p])
accuracy = correct / len(true_labels)
print(f"\n🎯 Backend Overall Accuracy: {accuracy*100:.2f}%\n")

# Classification report
print("📊 Classification Report:\n")
print(classification_report(true_labels, pred_labels, zero_division=0))

# Confusion matrix
print("🧮 Confusion Matrix:\n")
cm = confusion_matrix(true_labels, pred_labels, labels=list(sorted(set(true_labels))))
print(cm)
