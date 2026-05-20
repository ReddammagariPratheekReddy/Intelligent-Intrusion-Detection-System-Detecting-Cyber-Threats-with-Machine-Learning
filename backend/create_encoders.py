from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle

# Load dataset used in training
df_train = pd.read_csv("/Users/pratheek/Desktop/Major Project/data/KDDTrain+.txt", header=None)

# Create encoders
protocol_encoder = LabelEncoder()
service_encoder = LabelEncoder()
flag_encoder = LabelEncoder()

protocol_encoder.fit(df_train[1])  # protocol_type column
service_encoder.fit(df_train[2])   # service column
flag_encoder.fit(df_train[3])      # flag column

# Save encoders to backend/encoders/
import os
os.makedirs("/Users/pratheek/Desktop/Major Project/backend/encoders", exist_ok=True)

with open("/Users/pratheek/Desktop/Major Project/backend/encoders/protocol_type_encoder.pkl", "wb") as f:
    pickle.dump(protocol_encoder, f)

with open("/Users/pratheek/Desktop/Major Project/backend/encoders/service_encoder.pkl", "wb") as f:
    pickle.dump(service_encoder, f)

with open("/Users/pratheek/Desktop/Major Project/backend/encoders/flag_encoder.pkl", "wb") as f:
    pickle.dump(flag_encoder, f)
