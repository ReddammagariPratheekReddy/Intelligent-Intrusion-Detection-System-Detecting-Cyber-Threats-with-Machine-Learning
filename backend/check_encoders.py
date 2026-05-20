import joblib

feature_encoders_path = "/Users/pratheek/Desktop/Major Project/backend/feature_encoders.pkl"
feature_encoders = joblib.load(feature_encoders_path)

for col, enc in feature_encoders.items():
    print(f"\n🔠 Encoder for '{col}':")
    print(enc.classes_)
