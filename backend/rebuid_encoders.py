import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# ✅ Use your correct training dataset path
train_path = "/Users/pratheek/Desktop/Major Project/data/train_preprocessed.csv"

print("📥 Loading training data to rebuild encoders...")
train_df = pd.read_csv(train_path, low_memory=False)

# ✅ Columns to encode
categorical_cols = ['protocol_type', 'service', 'flag']
feature_encoders = {}

for col in categorical_cols:
    print(f"🔠 Rebuilding encoder for '{col}'...")
    le = LabelEncoder()
    train_df[col] = train_df[col].astype(str)
    le.fit(train_df[col])
    feature_encoders[col] = le
    print(f"✅ {col} encoder built with {len(le.classes_)} classes.")

# ✅ Save encoders to backend folder using pickle (recommended)
import pickle

encoder_path = "/Users/pratheek/Desktop/Major Project/backend/feature_encoders.pkl"
with open(encoder_path, "wb") as f:
    pickle.dump(feature_encoders, f)

print(f"\n💾 Feature encoders saved successfully to: {encoder_path}")


# 🔍 Quick verification
print("\n🔍 Sample classes:")
for col, enc in feature_encoders.items():
    print(f"{col}: {list(enc.classes_)[:10]} ... ({len(enc.classes_)} total)")
