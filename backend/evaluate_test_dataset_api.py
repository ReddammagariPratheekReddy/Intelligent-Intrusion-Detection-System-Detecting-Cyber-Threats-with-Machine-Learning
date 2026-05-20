import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# -------------------------
# Config
# -------------------------
API_URL = "http://127.0.0.1:8000/test-model-batch"
TEST_DATA_PATH = "/Users/pratheek/Desktop/Major Project/data/KDDTest+.txt"
CHUNK_SIZE = 500  # number of rows per API request
CATEGORICAL_COLS = [1, 2, 3]  # protocol_type, service, flag
FEATURE_COLS = list(range(41))  # assuming first 41 columns are features
TARGET_COL = 41  # assuming last column is the label

# -------------------------
# Load test data
# -------------------------
df = pd.read_csv(TEST_DATA_PATH, header=None)
print(f"✅ Test data loaded: {df.shape}")

# -------------------------
# Encode categorical columns
# -------------------------
# IMPORTANT: Use same encoding as training
encoders = {}
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Separate features and labels
X = df[FEATURE_COLS].values
y_true = df[TARGET_COL].values

# -------------------------
# Send data in chunks to FastAPI
# -------------------------
all_predictions = []

num_chunks = int(np.ceil(len(X) / CHUNK_SIZE))
print(f"Sending {len(X)} rows in {num_chunks} chunks...")

for i in range(num_chunks):
    start = i * CHUNK_SIZE
    end = min((i + 1) * CHUNK_SIZE, len(X))
    chunk = X[start:end].tolist()
    
    response = requests.post(API_URL, json={"features_list": chunk})
    
    if response.status_code != 200:
        print(f"❌ Chunk {i+1}/{num_chunks} failed: {response.text}")
        continue
    
    result = response.json()
    if "predictions" not in result:
        print(f"❌ Chunk {i+1}/{num_chunks} has no predictions")
        continue
    
    all_predictions.extend(result["predictions"])
    print(f"✅ Chunk {i+1}/{num_chunks} processed")

# -------------------------
# Evaluation
# -------------------------
if len(all_predictions) != len(y_true):
    print("❌ Number of predictions does not match number of samples!")
else:
    print("✅ All predictions received. Calculating metrics...")
    print(f"Accuracy: {accuracy_score(y_true, all_predictions):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, all_predictions))
