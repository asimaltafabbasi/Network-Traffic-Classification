import pandas as pd

# Load CSV
df = pd.read_csv("traffic_data.csv")

# Show first 10 rows
print(df.head(10))

# Show column names
print("\nColumns:")
print(df.columns)

# Show basic info
print("\nDataset Info:")
print(df.info())

# Count protocol types
print("\nProtocol Count:")
print(df["Protocol"].value_counts())