import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("KDDTrain+.txt", header=None)

# Create target column
df["target"] = df[41].apply(
    lambda x: 0 if x == "normal" else 1
)

# Encode categorical columns
encoder = LabelEncoder()

df[1] = encoder.fit_transform(df[1])
df[2] = encoder.fit_transform(df[2])
df[3] = encoder.fit_transform(df[3])

# Features (X)
X = df.drop(columns=[41, 42, "target"])

# Target (y)
y = df["target"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = DecisionTreeClassifier()

# Training
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:")
print(accuracy)