import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("KDDTrain+.txt", header=None)

# Create binary target
df["target"] = df[41].apply(
    lambda x: 0 if x == "normal" else 1
)

# Encode categorical columns
encoder = LabelEncoder()

df[1] = encoder.fit_transform(df[1])
df[2] = encoder.fit_transform(df[2])
df[3] = encoder.fit_transform(df[3])

print(df[[1, 2, 3]].head(10))

print("\nDataset Shape:")
print(df.shape)