import requests
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import math
import pickle  # for loading saved encoders

# ------------------------
# Configuration
# ------------------------
API_URL = "http://127.0.0.1:8000/test-model-batch"
TEST_DATA_PATH = "/Users/pratheek/Desktop/Major Project/data/KDDTest+.txt"
OUTPUT_PATH = "/Users/pratheek/Desktop/Major Project/data/predictions.csv"
CHUNK_SIZE = 500  # number of rows per batch request

# ------------------------
# Load Test Dataset
# ------------------------
df = pd.read_csv(TEST_DATA_PATH, header=None)

# ------------------------
# Assign feature & label columns
# ------------------------
FEATURE_COLUMNS = list(range(41))  # columns 0-40 are features
LABEL_COLUMN = 41                   # column 41 has the label
# Column 42 is ignored

# ------------------------
# Load saved LabelEncoders from training (pickle files)
# ------------------------
# Make sure you saved these encoders during training for columns 1,2,3
encoder_files = {
    1: "/Users/pratheek/Desktop/Major Project/backend/encoders/protocol_type_encoder.pkl",
    2: "/Users/pratheek/Desktop/Major Project/backend/encoders/service_encoder.pkl",
    3: "/Users/pratheek/Desktop/Major Project/backend/encoders/flag_encoder.pkl"
}

encoders = {}
for col, path in encoder_files.items():
    with open(path, "rb") as f:
        encoders[col] = pickle.load(f)
        df[col] = encoders[col].transform(df[col])  # convert strings to numeric

# ------------------------
# Prepare list of feature vectors
# ------------------------
features_list = df[FEATURE_COLUMNS].values.tolist()

# ------------------------
# Batch prediction in chunks
# ------------------------
all_predictions = []
num_chunks = math.ceil(len(features_list) / CHUNK_SIZE)

print(f"Sending {len(features_list)} rows in {num_chunks} chunks...")

for i in range(num_chunks):
    start = i * CHUNK_SIZE
    end = start + CHUNK_SIZE
    chunk = features_list[start:end]

    payload = {"features_list": chunk}
    response = requests.post(API_URL, json=payload)

    # Debug: print response if error occurs
    if response.status_code != 200:
        print(f"❌ Chunk {i+1} request failed: {response.status_code}")
        print(response.text)
        break

    data = response.json()
    chunk_predictions = data.get("predictions", [])

    if not chunk_predictions:
        print(f"❌ Chunk {i+1} returned empty predictions. Error: {data.get('error')}")
        break

    all_predictions.extend(chunk_predictions)
    print(f"✅ Chunk {i+1}/{num_chunks} processed")

# ------------------------
# Save predictions to CSV
# ------------------------
if all_predictions:
    df["prediction"] = all_predictions
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Predictions saved to {OUTPUT_PATH}")

    # ------------------------
    # Compute evaluation metrics
    # ------------------------
    y_true = df[LABEL_COLUMN].tolist()
    y_pred = df["prediction"].tolist()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\nEvaluation Metrics:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
else:
    print("❌ No predictions to evaluate.")
