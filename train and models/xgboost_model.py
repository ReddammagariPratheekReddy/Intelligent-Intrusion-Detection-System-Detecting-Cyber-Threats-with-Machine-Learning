import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib

print("📥 Loading preprocessed train and test datasets ...")

# Paths to preprocessed datasets
train_path = "/Users/pratheek/Desktop/Major Project/data/train_preprocessed.csv"
test_path  = "/Users/pratheek/Desktop/Major Project/data/test_preprocessed.csv"

# Load datasets
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(f"✅ Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Separate features and labels
X_train = train_df.drop("label", axis=1)
y_train = train_df["label"]
X_test  = test_df.drop("label", axis=1)
y_test  = test_df["label"]

print(f"ℹ️ Unique label examples (train): {y_train.unique()[:10]}")

# -----------------------------
# Handle label encoding safely
# -----------------------------
print("🔁 Encoding labels with LabelEncoder ...")

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)

# Handle unseen labels in test set
y_test_mapped = y_test.apply(lambda x: x if x in le.classes_ else "unknown")

# Add 'unknown' class if not present
le_classes = list(le.classes_)
if "unknown" not in le_classes:
    le_classes.append("unknown")
le.classes_ = np.array(le_classes)

y_test_enc = le.transform(y_test_mapped)

# -----------------------------
# Optimized XGBoost model
# -----------------------------
print("\n🚀 Training Optimized XGBoost model ...")

xgb_model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.1,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.2,
    reg_lambda=1.5,
    random_state=42,
    use_label_encoder=False,
    eval_metric="mlogloss"
)

xgb_model.fit(X_train, y_train_enc)

print("✅ Model training complete!")

# -----------------------------
# Evaluate model
# -----------------------------
print("\n📊 Evaluating model on test data ...")

y_pred = xgb_model.predict(X_test)

accuracy = accuracy_score(y_test_enc, y_pred)
print(f"🎯 Accuracy: {accuracy * 100:.2f}%\n")

print("📋 Classification Report:")
print(classification_report(y_test_enc, y_pred, zero_division=0))

print("🧮 Confusion Matrix:")
print(confusion_matrix(y_test_enc, y_pred))

# -----------------------------
# Save model and label encoder
# -----------------------------
model_path = "/Users/pratheek/Desktop/Major Project/backend/xgboost_ids_model.pkl"
encoder_path = "/Users/pratheek/Desktop/Major Project/backend/label_encoder.pkl"

joblib.dump(xgb_model, model_path)
joblib.dump(le, encoder_path)

print(f"\n💾 Model saved to: {model_path}")
print(f"💾 Label Encoder saved to: {encoder_path}")

print("\n✅ Optimized XGBoost model ready for backend integration!")
