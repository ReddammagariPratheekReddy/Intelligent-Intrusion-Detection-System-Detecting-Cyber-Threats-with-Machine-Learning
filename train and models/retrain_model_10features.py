import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import os

print("📥 Loading preprocessed train and test datasets ...")

# ----------------------------
# 1️⃣ Paths to preprocessed datasets
# ----------------------------
train_path = "/Users/pratheek/Desktop/Major Project/data/train_preprocessed.csv"
test_path  = "/Users/pratheek/Desktop/Major Project/data/test_preprocessed.csv"

if not os.path.exists(train_path) or not os.path.exists(test_path):
    raise FileNotFoundError("❌ Preprocessed datasets not found. Please check your /data folder.")

train_df = pd.read_csv(train_path, low_memory=False)
test_df = pd.read_csv(test_path, low_memory=False)

print(f"✅ Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# ----------------------------
# 2️⃣ Select only 10 features + label
# ----------------------------
selected_features = [
    'duration', 'protocol_type', 'service', 'flag',
    'src_bytes', 'dst_bytes', 'count', 'same_srv_rate',
    'wrong_fragment', 'urgent'
]

print(f"📊 Using selected features: {selected_features}")

train_df = train_df[selected_features + ['label']]
test_df = test_df[selected_features + ['label']]

# ----------------------------
# 3️⃣ Separate features and labels
# ----------------------------
X_train = train_df.drop("label", axis=1)
y_train = train_df["label"]
X_test  = test_df.drop("label", axis=1)
y_test  = test_df["label"]

print(f"ℹ️ Unique label examples (train): {y_train.unique()[:10]}")

# ----------------------------
# 4️⃣ Encode categorical columns safely
# ----------------------------
categorical_cols = ['protocol_type', 'service', 'flag']
label_encoders = {}

for col in categorical_cols:
    print(f"🔠 Encoding {col} ...")
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

    le_col = LabelEncoder()
    X_train[col] = le_col.fit_transform(X_train[col])
    
    # Handle unseen test values
    unseen = set(X_test[col]) - set(le_col.transform(le_col.classes_))
    X_test[col] = X_test[col].apply(lambda x: x if x in le_col.classes_ else le_col.classes_[0])
    X_test[col] = le_col.transform(X_test[col])
    
    label_encoders[col] = le_col

# ----------------------------
# 5️⃣ Encode labels
# ----------------------------
print("🔁 Encoding labels with LabelEncoder ...")

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train.astype(str))

# Handle unseen test labels
y_test_mapped = y_test.apply(lambda x: x if x in le.classes_ else "unknown")
le_classes = list(le.classes_)
if "unknown" not in le_classes:
    le_classes.append("unknown")
le.classes_ = np.array(le_classes)
y_test_enc = le.transform(y_test_mapped)

# ----------------------------
# 6️⃣ Train Optimized XGBoost model
# ----------------------------
print("\n🚀 Retraining Optimized XGBoost model (10 features) ...")

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
print("✅ Model retraining complete!")

# ----------------------------
# 7️⃣ Evaluate model
# ----------------------------
print("\n📊 Evaluating model on test data ...")
y_pred = xgb_model.predict(X_test)

accuracy = accuracy_score(y_test_enc, y_pred)
print(f"🎯 Accuracy: {accuracy * 100:.2f}%\n")

print("📋 Classification Report:")
print(classification_report(y_test_enc, y_pred, zero_division=0))

print("🧮 Confusion Matrix:")
print(confusion_matrix(y_test_enc, y_pred))

# ----------------------------
# 8️⃣ Save retrained model + encoders
# ----------------------------
model_path = "/Users/pratheek/Desktop/Major Project/backend/xgboost_model_10features.pkl"
label_encoder_path = "/Users/pratheek/Desktop/Major Project/backend/label_encoder_10features.pkl"
feature_encoders_path = "/Users/pratheek/Desktop/Major Project/backend/feature_encoders.pkl"

joblib.dump(xgb_model, model_path)
joblib.dump(le, label_encoder_path)
joblib.dump(label_encoders, feature_encoders_path)

print(f"\n💾 Model saved to: {model_path}")
print(f"💾 Label Encoder saved to: {label_encoder_path}")
print(f"💾 Feature Encoders saved to: {feature_encoders_path}")

print("\n✅ Retraining complete! Model and encoders are ready for backend integration.")
