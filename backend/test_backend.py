import requests
import pandas as pd

# Correct paths to preprocessed test dataset
test_df = pd.read_csv("/Users/pratheek/Desktop/Major Project/data/test_preprocessed.csv")  

# Separate features and true labels
features = test_df.iloc[:, :-1].values.tolist()  # all columns except label
true_labels = test_df.iloc[:, -1].tolist()       # last column = labels

# Send batch request to backend
url = "http://127.0.0.1:8000/predict-batch"
response = requests.post(url, json={"features_list": features})

# Check if request was successful
if response.status_code == 200:
    pred_labels = response.json()["predictions"]
    # Compute accuracy
    correct = sum([1 for t, p in zip(true_labels, pred_labels) if t == p])
    accuracy = correct / len(true_labels)
    print(f"Backend prediction accuracy: {accuracy*100:.2f}%")
else:
    print(f"Error: {response.status_code}, {response.text}")
