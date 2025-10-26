import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                             precision_score, recall_score, f1_score, log_loss)
from sklearn.preprocessing import LabelEncoder

# 1. Load the dataset
file_path = 'E:/Coding/sem 3/AI/Practical/datasets/badminton/badminton_dataset.csv'
df = pd.read_csv(file_path)

# 2. Preprocess the data
# Separate features (X) and target (y)
X = df.drop('Play_Badminton', axis=1)
y = df['Play_Badminton']

# Encode all categorical features and the target variable
encoders = {}
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

le_play = LabelEncoder()
y_encoded = le_play.fit_transform(y)

# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# 4. Create and train the Categorical Naive Bayes model
model = CategoricalNB()
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# 6. Evaluate the model

# --- Confusion Matrix ---
conf_matrix = confusion_matrix(y_test, y_pred)
print("--- Naive Bayes Classifier Evaluation ---")
print("\nConfusion Matrix:")
print(conf_matrix)
# TN, FP, FN, TP
tn, fp, fn, tp = conf_matrix.ravel()

# --- Accuracy, Precision, Recall, F1 Score ---
accuracy = accuracy_score(y_test, y_pred)
# We specify pos_label=1 to calculate metrics for the 'Yes' class
precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

print(f"\nAccuracy: {accuracy:.2f}")
print(f"Precision (for 'Yes'): {precision:.2f}")
print(f"Recall (Sensitivity) (for 'Yes'): {recall:.2f}")
print(f"F1 Score (for 'Yes'): {f1:.2f}")

# --- Specificity ---
# Specificity = True Negatives / (True Negatives + False Positives)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
print(f"Specificity (for 'No'): {specificity:.2f}")

# --- Log Loss ---
# This measures the performance of a classifier where the prediction is a probability
loss = log_loss(y_test, y_pred_proba)
print(f"Log Loss: {loss:.2f}")

print("\n--- Full Classification Report ---")
print(classification_report(y_test, y_pred, target_names=le_play.classes_, zero_division=0))

print("\nNote: RMSE and R2 Score are metrics for regression models and are not applicable to this classification task.")
