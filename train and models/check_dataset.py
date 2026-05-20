import pandas as pd

# Load your processed dataset
df = pd.read_csv("/Users/pratheek/Desktop/Major Project/data/train_preprocessed.csv")

# Print all column names
print("✅ Columns in your dataset:")
print(df.columns.tolist())

# Show first few rows
print("\n✅ First 5 rows:")
print(df.head())
