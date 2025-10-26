
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, r2_score
import numpy as np

# --- Data Loading and Preparation ---
try:
    # Load the dataset
    df = pd.read_csv('E:\\Coding\\sem 3\\AI\\practical\\datasets\\wine\\winequality-red.csv', sep=';')
except FileNotFoundError:
    print("Error: 'datasets/wine/winequality-red.csv' not found.")
    print("Please make sure the dataset is in the correct directory.")
    exit()

# Prepare data for classification and regression
X = df.drop('quality', axis=1)
y = df['quality']

# For classification, we'll create a binary target: 'good' (1) vs 'bad' (0)
# We'll define 'good' wine as having a quality score of 6 or higher
y_class = (y >= 6).astype(int)

# For regression, we use the original quality score
y_reg = y

# Split data for both tasks
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
)


# --- Part 1: Random Forest for Classification ---

print("--- Random Forest Classification Results ---")
print("Target: Wine quality (Good: >=6, Bad: <6)\n")

# Initialize and train the classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_class_train)

# Make predictions
y_class_pred = classifier.predict(X_test)

# Calculate classification metrics
accuracy = accuracy_score(y_class_test, y_class_pred)
conf_matrix = confusion_matrix(y_class_test, y_class_pred)
precision = precision_score(y_class_test, y_class_pred)
recall = recall_score(y_class_test, y_class_pred)
f1 = f1_score(y_class_test, y_class_pred)

# Specificity = True Negatives / (True Negatives + False Positives)
tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn + fp)

# Display classification metrics
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Specificity: {specificity:.4f}")


# --- Part 2: Random Forest for Regression ---

print("\n" + "="*40 + "\n")
print("--- Random Forest Regression Results ---")
print("Target: Wine quality (Score 3-8)\n")

# Initialize and train the regressor
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_reg_train)

# Make predictions
y_reg_pred = regressor.predict(X_test)

# Calculate regression metrics
r2 = r2_score(y_reg_test, y_reg_pred)
rmse = np.sqrt(np.mean((y_reg_pred - y_reg_test)**2))

# Display regression metrics
print(f"R2 Score: {r2:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

print("\n" + "="*40 + "\n")
print("Note on 'Loss Function':")
print("Random Forest does not optimize a single loss function in the way models like neural networks do.")
print("A common performance measure is the Out-of-Bag (OOB) score, which is an internal cross-validation estimate of accuracy or R2 score.")
