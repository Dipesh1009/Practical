'''
This script implements the Gaussian Naive Bayes classifier for the Iris dataset.
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# 1. Load the dataset
# The path to the Iris dataset CSV file
file_path = 'E:/Coding/sem 3/AI/Practical/datasets/iris/Iris.csv'
df = pd.read_csv(file_path)

# 2. Preprocess the data
# Drop the 'Id' column as it is not a predictive feature
df = df.drop('Id', axis=1)

# Separate the features (X) from the target variable (y)
X = df.drop('Species', axis=1)
y = df['Species']

# Encode the categorical target variable 'Species' into numerical form
# This is necessary for the scikit-learn model to process the labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 3. Split the data into training and testing sets
# We use 80% of the data for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 4. Create and train the Naive Bayes model
# We use GaussianNB because the features (sepal/petal length/width) are continuous
model = GaussianNB()
model.fit(X_train, y_train)

# 5. Make predictions on the test set
y_pred = model.predict(X_test)

# 6. Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the evaluation results
print("--- Naive Bayesian Classifier Evaluation ---")
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)
print("\nConfusion Matrix:")
print(conf_matrix)

# 7. Example of predicting a new, unseen sample
# This demonstrates how to use the trained model for a new prediction
# The sample values are [SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]
new_sample = [[5.1, 3.5, 1.4, 0.2]]  # This is a typical Iris-setosa

# Predict the class for the new sample
prediction_encoded = model.predict(new_sample)

# Decode the prediction to get the original species name
predicted_species = le.inverse_transform(prediction_encoded)

print("\n--- Prediction for a New Sample ---")
print(f"Sample data: {new_sample[0]}")
print(f"Predicted species: {predicted_species[0]}")
