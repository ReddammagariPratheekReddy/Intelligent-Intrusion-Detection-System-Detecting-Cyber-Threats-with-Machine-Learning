import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ===============================
# 📂 File paths (absolute paths)
# ===============================
train_path = "/Users/pratheek/Desktop/Major Project/data/KDDTrain+.txt"
test_path = "/Users/pratheek/Desktop/Major Project/data/KDDTest+.txt"

print("📥 Loading raw datasets ...")

# ===============================
# 🧩 Define column names (42 total)
# ===============================
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty_level'
]

# ===============================
# 📊 Load datasets
# ===============================
train_df = pd.read_csv(train_path, names=columns)
test_df = pd.read_csv(test_path, names=columns)

print("✅ Loaded datasets successfully!")
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# ===============================
# 🔍 Drop unnecessary columns
# ===============================
train_df.drop(['difficulty_level'], axis=1, inplace=True)
test_df.drop(['difficulty_level'], axis=1, inplace=True)

# ===============================
# 🏷️ Encode categorical features
# ===============================
cat_cols = ['protocol_type', 'service', 'flag']
encoder = LabelEncoder()

print("🔠 Encoding categorical columns:", cat_cols)
for col in cat_cols:
    train_df[col] = encoder.fit_transform(train_df[col])
    test_df[col] = encoder.transform(test_df[col])

# ===============================
# ⚖️ Normalize numeric features
# ===============================
scaler = StandardScaler()

X_train = train_df.drop(['label'], axis=1)
X_test = test_df.drop(['label'], axis=1)

print("⚙️ Scaling numerical features ...")
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# 🧩 Combine scaled data with labels
# ===============================
train_preprocessed = pd.DataFrame(X_train_scaled, columns=X_train.columns)
train_preprocessed['label'] = train_df['label']

test_preprocessed = pd.DataFrame(X_test_scaled, columns=X_test.columns)
test_preprocessed['label'] = test_df['label']

# ===============================
# 💾 Save preprocessed data
# ===============================
train_preprocessed.to_csv("/Users/pratheek/Desktop/Major Project/data/train_preprocessed.csv", index=False)
test_preprocessed.to_csv("/Users/pratheek/Desktop/Major Project/data/test_preprocessed.csv", index=False)

print("💾 Preprocessed data saved successfully!")
print("✅ Files created:")
print("   • train_preprocessed.csv")
print("   • test_preprocessed.csv")
