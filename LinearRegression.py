import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# Load the dataset
df = pd.read_csv('datasets/wine/winequality-red.csv', sep=';')

# Features and target
X = df.drop('quality', axis=1)
y = df['quality']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# --- Regression Evaluation Metrics ---
# These metrics are used for regression tasks where the goal is to predict a continuous value.

# Mean Squared Error (MSE) - This is a common loss function for regression
mse = mean_squared_error(y_test, y_pred)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# R-squared (R2)
r2 = r2_score(y_test, y_pred)


# Display the results
print("--- Regression Model Evaluation ---")
print(f'Mean Squared Error (Loss): {mse:.4f}')
print(f'Root Mean Squared Error: {rmse:.4f}')
print(f'Mean Absolute Error: {mae:.4f}')
print(f'R2 Score: {r2:.4f}')

# --- Note on Classification Metrics ---
print("\n--- A Note on Classification Metrics ---")
print("Metrics like Accuracy, Precision, Recall, F1-Score, and Confusion Matrix are used for CLASSIFICATION tasks, not regression.")
print("To use them, you would need to convert the 'quality' score into categories (e.g., 'good' vs. 'bad') and use a classification model.")