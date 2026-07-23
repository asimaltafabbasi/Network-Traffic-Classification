import pandas as pd

df = pd.read_csv("traffic_data.csv")

print("Columns:")
print(df.columns)

print("\nProtocol Counts:")
print(df["Protocol"].value_counts())

print("\nDataset Shape:")
print(df.shape)