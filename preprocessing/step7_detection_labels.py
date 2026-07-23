import pandas as pd

# Load dataset
df = pd.read_csv("KDDTrain+.txt", header=None)

# Create binary labels
df["target"] = df[41].apply(
    lambda x: 0 if x == "normal" else 1
)

print("Detection Labels Count:")
print(df["target"].value_counts())

print("\nSample:")
print(df[[41, "target"]].head(20))