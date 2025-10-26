import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Load the dataset
df = pd.read_csv('E:\\Coding\\sem 3\\AI\\practical\\datasets\\penguin\\penguins.csv')

# --- Preprocessing ---

# Drop rows with missing values
df.dropna(inplace=True)

# There is a row with '.' in the 'sex' column, which is invalid.
df = df[df['sex'] != '.']

# Encode the 'sex' column to be our ground truth for evaluation
le = LabelEncoder()
df['sex_encoded'] = le.fit_transform(df['sex'])
y_true = df['sex_encoded']

# Select features for clustering
features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
X = df[features]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- K-Means Clustering ---

# We have two sexes, so we'll use 2 clusters to see how well K-means can separate them.
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(X_scaled)
cluster_labels = kmeans.labels_

# --- Evaluation ---

# K-means is unsupervised, so it doesn't know which cluster corresponds to which sex.
# We need to map the cluster labels (0 and 1) to the sex labels (MALE and FEMALE).
# We can do this by looking at the most common sex in each cluster.

# Create a mapping from cluster label to the most frequent true label in that cluster
cluster_to_sex_map = {}
for i in range(n_clusters):
    # Get the true labels for data points in the current cluster
    labels_in_cluster = y_true[cluster_labels == i]
    # Find the most frequent label in this cluster
    most_frequent_label = labels_in_cluster.mode()[0]
    cluster_to_sex_map[i] = most_frequent_label

# Use the mapping to get the predicted labels
y_pred = np.array([cluster_to_sex_map[label] for label in cluster_labels])

# Now we can calculate the classification metrics
conf_matrix = confusion_matrix(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='binary')
recall = recall_score(y_true, y_pred, average='binary')
f1 = f1_score(y_true, y_pred, average='binary')

# Calculate Specificity
tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn + fp)

# The "loss function" for K-means is the within-cluster sum of squares (WCSS), also called inertia.
inertia = kmeans.inertia_

# --- Display Results ---

print("K-Means Clustering Evaluation Results")
print("=====================================")
print(f"Number of clusters (K): {n_clusters}")
print("\n--- Evaluation Metrics (comparing clusters to 'sex') ---")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"K-Means Inertia (WCSS): {inertia:.4f}")

print("\nNote on other metrics:")
print("RMSE and R2 Score are metrics for regression tasks and are not applicable to clustering.")

