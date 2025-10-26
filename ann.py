import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ---
# Note: This script requires TensorFlow. Please install it if you haven't already:
# pip install tensorflow
# ---

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --- Data Loading and Preparation ---
# Load the MNIST digits dataset from sklearn
digits = load_digits()
X = digits.data
y = digits.target.reshape(-1, 1)

print(f"Dataset loaded. Number of samples: {X.shape[0]}, features per sample: {X.shape[1]}\n")

# --- Preprocessing ---
# 1. One-Hot Encode the categorical target variable (digits 0-9)
encoder = OneHotEncoder(sparse_output=False, categories='auto')
y_encoded = encoder.fit_transform(y)

# 2. Scale the numerical features (pixel values)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y
)

# --- ANN Model Building ---
# Define the model
model = Sequential([
    # Input layer (64 features for 8x8 images) and first hidden layer
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    # Second hidden layer
    Dense(32, activation='relu'),
    # Output layer - 10 units for 10 digits (0-9), softmax for multi-class probability
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- Model Training ---
print("--- Training Artificial Neural Network on MNIST Digits ---")
# Train the model on the training data
# verbose=0 will hide the training progress per epoch
history = model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0, validation_split=0.1)
print("Model training complete.\n")

# --- Model Evaluation ---
print("--- ANN Performance Evaluation ---")

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Loss: {loss:.4f}\n")

# Make predictions
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculate and display other metrics
conf_matrix = confusion_matrix(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Precision (Macro): {precision:.4f}")
print(f"Recall (Macro): {recall:.4f}")
print(f"F1 Score (Macro): {f1:.4f}")
