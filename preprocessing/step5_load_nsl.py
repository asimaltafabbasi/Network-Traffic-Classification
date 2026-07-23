import pandas as pd

df = pd.read_csv("KDDTrain+.txt", header=None)

print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())