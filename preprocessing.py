
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression

# Load the Wine Quality dataset
file_path = "E:/Coding/sem 3/AI/Practical/datasets/wine/winequality-red.csv"
data = pd.read_csv(file_path, sep=';')

# --- 1. Imputation ---
# Introduce some missing values for demonstration
data_with_missing = data.copy()
for col in data_with_missing.columns:
    if data_with_missing[col].dtype != 'object':
        indices = data_with_missing.sample(frac=0.1).index
        data_with_missing.loc[indices, col] = np.nan

print("--- Imputation ---")
print("Data with missing values (first 5 rows):")
print(data_with_missing.head())

# Impute missing values using the mean
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data_with_missing), columns=data.columns)

print("\nData after imputation (first 5 rows):")
print(data_imputed.head())

# --- 2. Standardization ---
print("\n--- Standardization ---")
scaler = StandardScaler()
data_standardized = pd.DataFrame(scaler.fit_transform(data_imputed.drop('quality', axis=1)), columns=data.columns[:-1])
print("Standardized data (first 5 rows):")
print(data_standardized.head())

# --- 3. Handling Categorical Variables ---
# Create a dummy categorical column
data_imputed['quality_category'] = pd.cut(data_imputed['quality'], bins=[0, 5, 6, 8], labels=['low', 'medium', 'high'])

print("\n--- Handling Categorical Variables ---")
print("Data with categorical column (first 5 rows):")
print(data_imputed.head())

# One-hot encode the categorical column
encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(data_imputed[['quality_category']]).toarray()
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['quality_category']))
data_encoded = pd.concat([data_imputed, encoded_df], axis=1).drop('quality_category', axis=1)

print("\nData after one-hot encoding (first 5 rows):")
print(data_encoded.head())

# --- 4. Outlier Management ---
print("\n--- Outlier Management ---")
Q1 = data['alcohol'].quantile(0.25)
Q3 = data['alcohol'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = data[(data['alcohol'] < lower_bound) | (data['alcohol'] > upper_bound)]
print(f"Number of outliers in 'alcohol' column: {len(outliers)}")

# Remove outliers
data_no_outliers = data[(data['alcohol'] >= lower_bound) & (data['alcohol'] <= upper_bound)]
print(f"Shape of data after removing outliers: {data_no_outliers.shape}")

# --- 5. Cross-Validation ---
print("\n--- Cross-Validation ---")
X = data.drop('quality', axis=1)
y = data['quality']

# Binarize the target for logistic regression (e.g., good vs. not good)
y_binary = (y > 5).astype(int)

model = LogisticRegression(max_iter=1000)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y_binary, cv=kf)

print(f"Cross-validation scores: {scores}")
print(f"Mean cross-validation score: {scores.mean()}")
