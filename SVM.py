import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Load the dataset
file_path = 'E:/Coding/sem 3/AI/Practical/datasets/iris/Iris.csv'
df = pd.read_csv(file_path)

# 2. Prepare the data for binary classification
# Drop the 'Id' column
df = df.drop('Id', axis=1)

# Filter the dataset to keep only two classes: Iris-versicolor and Iris-virginica
df_binary = df[df['Species'].isin(['Iris-versicolor', 'Iris-virginica'])]

# Separate features (X) and target (y)
X = df_binary.drop('Species', axis=1)
y = df_binary['Species']

# 3. Preprocess the data
# Encode the target labels into numerical format
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale the features
# Feature scaling is crucial for SVMs to perform well
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# 5. Create and train the SVM model
# We use the default RBF (Radial Basis Function) kernel, which is effective for non-linear data
model = SVC(kernel='rbf', random_state=42)
model.fit(X_train, y_train)

# 6. Make predictions on the test set
y_pred = model.predict(X_test)

# 7. Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)

# Display the evaluation results
print("--- Support Vector Machine (SVM) Binary Classifier Evaluation ---")
print(f"Dataset: Iris (Versicolor vs. Virginica)")
print(f"\nAccuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)

# 8. Example of predicting a new, unseen sample
# Sample values are [SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]
# This sample is characteristic of an Iris-versicolor
new_sample = [[6.0, 2.9, 4.5, 1.5]]

# Scale the new sample using the same scaler
new_sample_scaled = scaler.transform(new_sample)

# Predict the class for the new sample
prediction_encoded = model.predict(new_sample_scaled)

# Decode the prediction
predicted_species = le.inverse_transform(prediction_encoded)

print("\n--- Prediction for a New Sample ---")
print(f"Sample data: {new_sample[0]}")
print(f"Predicted species: {predicted_species[0]}")

