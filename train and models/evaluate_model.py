import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# -----------------------------
# Step 1: Load the test dataset
# -----------------------------
test_file_path = "/Users/pratheek/Desktop/Major Project/data/test_preprocessed.csv"

try:
    print("📂 Loading test dataset...")
    test_df = pd.read_csv(test_file_path)
    print(f"✅ Test dataset loaded successfully! Shape: {test_df.shape}")
except Exception as e:
    print(f"❌ Error loading test dataset: {e}")
    exit()

# -----------------------------
# Step 2: Separate features and target
# -----------------------------
target_column = "label"  # Update if different
X_test = test_df.drop(columns=[target_column])
y_test = test_df[target_column].astype(str)

# -----------------------------
# Step 3: Load the optimized trained model
# -----------------------------
model_file_path = "/Users/pratheek/Desktop/Major Project/train/rf_ids_model_optimized.pkl"

try:
    print("📂 Loading optimized trained model...")
    model = joblib.load(model_file_path)
    print("✅ Optimized model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# -----------------------------
# Step 4: Make predictions
# -----------------------------
try:
    print("🔮 Making predictions on test data...")
    y_pred = model.predict(X_test)
    y_pred = pd.Series(y_pred).astype(str)  # Ensure string type
    print("✅ Predictions completed!")
except Exception as e:
    print(f"❌ Error during prediction: {e}")
    exit()

# Optional debug
print("Sample y_test labels:", y_test[:10].tolist())
print("Sample y_pred labels:", y_pred[:10].tolist())

# -----------------------------
# Step 5: Evaluate the model
# -----------------------------
try:
    print("\n📊 Evaluation Metrics:")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall   : {recall:.4f}")
    print(f"Weighted F1-Score : {f1:.4f}")
    print(f"Macro F1-Score    : {macro_f1:.4f}")

    print("\nConfusion Matrix:")
    print(conf_matrix)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # -----------------------------
    # Step 6: Visualize Normalized Confusion Matrix
    # -----------------------------
    plt.figure(figsize=(12, 10))
    labels = sorted(list(set(y_test)))
    normalized_cm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, None]
    sns.heatmap(normalized_cm, annot=False, cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Normalized Confusion Matrix Heatmap")
    plt.show()

except Exception as e:
    print(f"❌ Error during evaluation: {e}")
