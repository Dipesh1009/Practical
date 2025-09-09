import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load the dataset
file_path = "E:/Coding/sem 3/AI/Practical/datasets/student stress/StressLevelDataset.csv"
data = pd.read_csv(file_path)

# Separate features and target
X = data.drop('stress_level', axis=1)
y = data['stress_level']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Decision Tree model
# Using a small max_depth to make the tree visualization readable
dtree = DecisionTreeClassifier(max_depth=3, random_state=42)
dtree.fit(X_train, y_train)

# --- Visualize the Decision Tree ---
print("--- Decision Tree Visualization ---")
plt.figure(figsize=(20,10))
plot_tree(dtree, feature_names=X.columns, class_names=[str(i) for i in sorted(y.unique())], filled=True)
plt.title("Decision Tree Visualization (max_depth=3)")
plt.show()

# --- Evaluate the Model ---
print("\n--- Model Evaluation ---")
y_pred = dtree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- Classify a New Sample ---
# Create a new sample (you can change these values)
new_sample = pd.DataFrame([{
    'anxiety_level': 10,
    'self_esteem': 15,
    'mental_health_history': 1,
    'depression': 10,
    'headache': 2,
    'blood_pressure': 1,
    'sleep_quality': 3,
    'breathing_problem': 2,
    'noise_level': 2,
    'living_conditions': 3,
    'safety': 3,
    'basic_needs': 2,
    'academic_performance': 3,
    'study_load': 3,
    'teacher_student_relationship': 3,
    'future_career_concerns': 2,
    'social_support': 2,
    'peer_pressure': 3,
    'extracurricular_activities': 2,
    'bullying': 2
}])

print("\n--- Classifying a New Sample ---")
print("New sample data:")
print(new_sample)

# Predict the stress level for the new sample
new_prediction = dtree.predict(new_sample)

# Map the numeric prediction to a meaningful label
stress_level_map = {0: 'Low Stress', 1: 'Moderate Stress', 2: 'High Stress'}
predicted_stress_label = stress_level_map.get(new_prediction[0], "Unknown")

print(f"\nPredicted stress level for the new sample: {predicted_stress_label}")
