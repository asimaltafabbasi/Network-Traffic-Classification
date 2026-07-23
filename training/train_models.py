
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
df = pd.read_csv("dataset/processed/cleaned_dataset.csv")
print("Dataset Loaded Successfully!")
print("\nFirst 5 Rows:")
print(df.head())
print("\nDataset Shape:")
print(df.shape)
print("\nColumn Names:")
print(df.columns.tolist())
print("\nDataset Information:")
print(df.info())

# Separate Features and Label
X = df.drop("label", axis=1)
y = df["label"]
print("\nFeatures Shape:")
print(X.shape)
print("\nLabel Shape:")
print(y.shape)
print("\nFeature Columns:")
print(X.columns.tolist())
print("\nUnique Labels:")
print(sorted(y.unique()))

# Split Dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    shuffle=True,
    stratify=y
)
print("\nDataset Split Successfully!")
print("\nTraining Features Shape:")
print(X_train.shape)
print("\nTesting Features Shape:")
print(X_test.shape)
print("\nTraining Labels Shape:")
print(y_train.shape)
print("\nTesting Labels Shape:")
print(y_test.shape)

# Check Class Distribution
print("\nClass Distribution in Complete Dataset:")
print(y.value_counts().sort_index())
print("\nClass Distribution in Training Dataset:")
print(y_train.value_counts().sort_index())
print("\nClass Distribution in Testing Dataset:")
print(y_test.value_counts().sort_index())

# ==========================================
# Logistic Regression Model
# ==========================================
print("\nTraining Logistic Regression Model...")
logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)
logistic_model.fit(X_train, y_train)
print("Logistic Regression Training Completed!")

# ==========================================
# Predict Using Logistic Regression
# ==========================================
print("\nMaking Predictions...")
y_pred = logistic_model.predict(X_test)
print("Prediction Completed!")
print("\nFirst 20 Predictions:")
print(y_pred[:20])
print("\nFirst 20 Actual Labels:")
print(y_test.values[:20])

# ==========================================
# Calculate Accuracy
# ==========================================
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

# ==========================================
# Confusion Matrix
# ==========================================
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ==========================================
# Precision, Recall and F1 Score
# ==========================================
precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)
recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)
f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)
print("\nModel Evaluation")
print(f"Accuracy : {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall   : {recall*100:.2f}%")
print(f"F1 Score : {f1*100:.2f}%")

# ==========================================
# Decision Tree Model
# ==========================================
print("\nTraining Decision Tree Model...")
decision_tree = DecisionTreeClassifier(
    random_state=42
)
decision_tree.fit(X_train, y_train)
print("Decision Tree Training Completed!")

# ==========================================
# Predict Using Decision Tree
# ==========================================
print("\nMaking Predictions...")
y_pred_dt = decision_tree.predict(X_test)
print("Prediction Completed!")
print("\nFirst 20 Predictions:")
print(y_pred_dt[:20])
print("\nFirst 20 Actual Labels:")
print(y_test.values[:20])

# ==========================================
# Decision Tree Accuracy
# ==========================================
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("\nDecision Tree Accuracy:")
print(f"{accuracy_dt * 100:.2f}%")

# ==========================================
# Decision Tree Confusion Matrix
# ==========================================
cm_dt = confusion_matrix(y_test, y_pred_dt)
print("\nDecision Tree Confusion Matrix:")
print(cm_dt)

# ==========================================
# Decision Tree Evaluation
# ==========================================
precision_dt = precision_score(
    y_test,
    y_pred_dt,
    average="weighted",
    zero_division=0
)
recall_dt = recall_score(
    y_test,
    y_pred_dt,
    average="weighted",
    zero_division=0
)
f1_dt = f1_score(
    y_test,
    y_pred_dt,
    average="weighted",
    zero_division=0
)
print("\nDecision Tree Evaluation")
print(f"Accuracy : {accuracy_dt*100:.2f}%")
print(f"Precision: {precision_dt*100:.2f}%")
print(f"Recall   : {recall_dt*100:.2f}%")
print(f"F1 Score : {f1_dt*100:.2f}%")

# ==========================================
# Random Forest Model
# ==========================================
print("\nTraining Random Forest Model...")
random_forest = RandomForestClassifier(
    random_state=42
)
random_forest.fit(X_train, y_train)
print("Random Forest Training Completed!")

# ==========================================
# Predict Using Random Forest
# ==========================================
y_pred_rf = random_forest.predict(X_test)
print("\nRandom Forest Prediction:")
print("\nFirst 20 Predictions:")
print(y_pred_rf[:20])
print("\nFirst 20 Actual Labels:")
print(y_test.values[:20])

# ==========================================
# Random Forest Accuracy
# ==========================================
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("\nRandom Forest Accuracy:")
print(f"{accuracy_rf * 100:.2f}%")

# ==========================================
# Random Forest Confusion Matrix
# ==========================================
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("\nRandom Forest Confusion Matrix:")
print(cm_rf)

# ==========================================
# Random Forest Evaluation
# ==========================================
#Precision
precision_rf = precision_score(
    y_test,
    y_pred_rf,
    average="weighted",
    zero_division=0
)
#Recall
recall_rf = recall_score(
    y_test,
    y_pred_rf,
    average="weighted",
    zero_division=0
)
#F1 call
f1_rf = f1_score(
    y_test,
    y_pred_rf,
    average="weighted",
    zero_division=0
)
print("\nRandom Forest Evaluation")
print(f"Accuracy : {accuracy_rf * 100:.2f}%")
print(f"Precision: {precision_rf * 100:.2f}%")
print(f"Recall   : {recall_rf * 100:.2f}%")
print(f"F1 Score : {f1_rf * 100:.2f}%")

# ==========================================
# K-Nearest Neighbors (KNN) Model
# ==========================================
print("\nTraining K-Nearest Neighbors (KNN) Model...")
knn_model = KNeighborsClassifier(
    n_neighbors=5
)
knn_model.fit(X_train, y_train)
print("KNN Training Completed!")

# ==========================================
# Predict Using KNN
# ==========================================
print("\nKNN Prediction:")
y_pred_knn = knn_model.predict(X_test)
print("\nFirst 20 Predictions:")
print(y_pred_knn[:20])
print("\nFirst 20 Actual Labels:")
print(y_test.values[:20])

# ==========================================
# KNN Accuracy
# ==========================================
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("\nKNN Accuracy:")
print(f"{accuracy_knn * 100:.2f}%")

# ==========================================
# KNN Confusion Matrix
# ==========================================
cm_knn = confusion_matrix(y_test, y_pred_knn)
print("\nKNN Confusion Matrix:")
print(cm_knn)

# ==========================================
# KNN Evaluation
# ==========================================
precision_knn = precision_score(
    y_test,
    y_pred_knn,
    average="weighted",
    zero_division=0
)
recall_knn = recall_score(
    y_test,
    y_pred_knn,
    average="weighted",
    zero_division=0
)
f1_knn = f1_score(
    y_test,
    y_pred_knn,
    average="weighted",
    zero_division=0
)
print("\nKNN Evaluation")
print(f"Accuracy : {accuracy_knn * 100:.2f}%")
print(f"Precision: {precision_knn * 100:.2f}%")
print(f"Recall   : {recall_knn * 100:.2f}%")
print(f"F1 Score : {f1_knn * 100:.2f}%")

# ==========================================
# Save Best Model (Random Forest)
# ==========================================
import joblib
joblib.dump(random_forest, "models/best_model.pkl")
print("\nBest Model Saved Successfully!")