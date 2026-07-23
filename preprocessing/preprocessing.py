import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

column_names = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate",
    "dst_host_srv_rerror_rate","label","difficulty"
]
df = pd.read_csv(
    "dataset/raw/KDDTrain+.txt",
    header=None,
    names=column_names
)
# Columns to remove
columns_to_remove = [
    "difficulty",
    "land",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login"
]
# Remove unnecessary columns
df = df.drop(columns=columns_to_remove)


# Check unique values in categorical columns
print("Protocol Types:")
print(df["protocol_type"].unique())

print("\nServices:")
print(df["service"].unique())

print("\nFlags:")
print(df["flag"].unique())

print("\nLabels:")
print(df["label"].unique())

protocol_encoder = LabelEncoder()
service_encoder = LabelEncoder()
flag_encoder = LabelEncoder()
label_encoder = LabelEncoder()

df["protocol_type"] = protocol_encoder.fit_transform(df["protocol_type"])
df["service"] = service_encoder.fit_transform(df["service"])
df["flag"] = flag_encoder.fit_transform(df["flag"])
df["label"] = label_encoder.fit_transform(df["label"])

joblib.dump(protocol_encoder, "models/protocol_encoder.pkl")
joblib.dump(service_encoder, "models/service_encoder.pkl")
joblib.dump(flag_encoder, "models/flag_encoder.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")

# Create scaler
scaler = MinMaxScaler()
feature_columns = df.columns.drop("label")
df[feature_columns] = scaler.fit_transform(df[feature_columns])
joblib.dump(scaler, "models/minmax_scaler.pkl")

print(df.head())
# Save the processed dataset
df.to_csv("dataset/processed/cleaned_dataset.csv", index=False)

print("Processed dataset saved successfully!")