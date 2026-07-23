import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load dataset
df = pd.read_csv("KDDTrain+.txt", header=None)

# Target
df["target"] = df[41].apply(
    lambda x: 0 if x == "normal" else 1
)

# Encode categorical columns
encoder = LabelEncoder()

df[1] = encoder.fit_transform(df[1])
df[2] = encoder.fit_transform(df[2])
df[3] = encoder.fit_transform(df[3])

# Features and target
X = df.drop(columns=[41, 42, "target"])
y = df["target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = DecisionTreeClassifier()

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Results
print("Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))