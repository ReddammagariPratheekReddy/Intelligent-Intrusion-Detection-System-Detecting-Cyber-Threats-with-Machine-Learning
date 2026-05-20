import pandas as pd
import joblib
import numpy as np

# ================================
# 1️⃣ Load model and encoders
# ================================
print("📦 Loading model and encoders ...")

model_path = "/Users/pratheek/Desktop/Major Project/backend/xgboost_model_10features.pkl"
label_encoder_path = "/Users/pratheek/Desktop/Major Project/backend/label_encoder_10features.pkl"
feature_encoders_path = "/Users/pratheek/Desktop/Major Project/backend/feature_encoders.pkl"

model = joblib.load(model_path)
label_encoder = joblib.load(label_encoder_path)
feature_encoders = joblib.load(feature_encoders_path)

print("✅ Model and encoders loaded successfully!\n")

# ================================
# 2️⃣ Define selected features
# ================================
selected_features = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "count",
    "same_srv_rate",
    "wrong_fragment",
    "urgent"
]

# ================================
# 3️⃣ Sample input data for testing
# ================================
# NOTE: keep categorical values exactly as used during training (case-sensitive)
input_data = [
    {
        "duration": 0,
        "protocol_type": "tcp",
        "service": "http",
        "flag": "SF",
        "src_bytes": 181,
        "dst_bytes": 5450,
        "count": 2,
        "same_srv_rate": 1.0,
        "wrong_fragment": 0,
        "urgent": 0
    },
    {
        "duration": 0,
        "protocol_type": "udp",
        "service": "domain_u",
        "flag": "SF",
        "src_bytes": 105,
        "dst_bytes": 146,
        "count": 5,
        "same_srv_rate": 0.8,
        "wrong_fragment": 0,
        "urgent": 0
    },
    {
        "duration": 2,
        "protocol_type": "icmp",
        "service": "ecr_i",
        "flag": "SF",
        "src_bytes": 0,
        "dst_bytes": 0,
        "count": 1,
        "same_srv_rate": 0.0,
        "wrong_fragment": 0,
        "urgent": 0
    }
]

# ================================
# 4️⃣ Convert to DataFrame
# ================================
df = pd.DataFrame(input_data)
print("🧾 Original Input:")
print(df, "\n")

# ================================
# 5️⃣ Encode categorical features
# ================================
categorical_cols = ["protocol_type", "service", "flag"]

for col in categorical_cols:
    encoder = feature_encoders[col]
    # Handle unseen categories safely
    df[col] = df[col].apply(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
    df[col] = encoder.transform(df[col])

print("🔢 Encoded Features:")
print(df, "\n")

# ================================
# 6️⃣ Make predictions
# ================================
predicted_numeric = model.predict(df)
predicted_labels = label_encoder.inverse_transform(predicted_numeric)

print("🧮 Predicted numeric labels:", predicted_numeric)
print("✅ Decoded labels:", predicted_labels, "\n")

# ================================
# 7️⃣ Display final results
# ================================
results_df = pd.DataFrame({
    "Input_Index": range(len(predicted_labels)),
    "Prediction_Label": predicted_labels
})

print("📊 Final Predictions:")
print(results_df)
