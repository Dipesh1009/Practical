import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss
import numpy as np

# Load the Titanic dataset
titanic = sns.load_dataset('titanic')

# --- Displaying Dataset Information ---
print("---" + "-" * 5 + " First 5 Rows of the Titanic Dataset " + "-" * 5 + "---")
print(titanic.head())
print("\n---" + "-" * 5 + " Dataset Info " + "-" * 5 + "---")
titanic.info()
print("\n" + "=" * 50 + "\n")

# Drop rows with missing 'age' or 'embarked' values for simplicity
titanic.dropna(subset=['age', 'embarked'], inplace=True)

# Fill missing 'deck' values with 'Unknown'
titanic['deck'] = titanic['deck'].cat.add_categories('Unknown').fillna('Unknown')

# Select features and target
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'deck']
target = 'survived'

# Convert categorical features to dummy variables
titanic_processed = pd.get_dummies(titanic[features], drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    titanic_processed, titanic[target], test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test) # For log_loss

# --- Evaluate the Model ---
print("---" + "-" * 5 + " Model Evaluation Metrics " + "-" * 5 + "---")

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)
# Note: [[TN, FP], [FN, TP]]

# Specificity
tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn + fp)
print(f"\nSpecificity (True Negative Rate): {specificity:.4f}")

# Loss Function (Log Loss)
loss = log_loss(y_test, y_pred_proba)
print(f"Log Loss (Loss Function): {loss:.4f}")

# Classification Report (Precision, Recall, F1-Score)
report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)