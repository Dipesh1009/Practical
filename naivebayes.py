import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re

# 1. Load the dataset
# The path to the spam dataset CSV file
file_path = 'E:/Coding/sem 3/AI/Practical/datasets/spam email/spam.csv'
# Load the dataset with specified encoding and handle potential errors
df = pd.read_csv(file_path, encoding='latin1')

# 2. Preprocess the data
# Drop unnecessary columns and rename the main columns for clarity
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
df = df.rename(columns={'v1': 'label', 'v2': 'message'})

# Map labels to numerical values: 'ham' -> 0, 'spam' -> 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Clean the text messages by removing punctuation and converting to lowercase
def clean_text(text):
    # Remove any character that is not a letter or a number
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

df['message'] = df['message'].apply(clean_text)

# 3. Feature extraction using TF-IDF
# Separate the features (X) from the target variable (y)
X = df['message']
y = df['label']

# Initialize the TF-IDF Vectorizer
# This will convert the text messages into a matrix of TF-IDF features
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

# Fit and transform the training data
X_tfidf = tfidf_vectorizer.fit_transform(X)

# 4. Split the data into training and testing sets
# We use 80% of the data for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# 5. Create and train the Naive Bayes model
# We use MultinomialNB, which is well-suited for text classification problems
model = MultinomialNB()
model.fit(X_train, y_train)

# 6. Make predictions on the test set
y_pred = model.predict(X_test)

# 7. Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the evaluation results
print("--- Multinomial Naive Bayes Classifier Evaluation for Spam Detection ---")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)
print("\nConfusion Matrix:")
print(conf_matrix)

# 8. Example of predicting a new, unseen message
# This demonstrates how to use the trained model for a new prediction
new_messages = [
    "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
    "Hey, are you coming to the party tonight?",
    "URGENT! You have won a 1 week FREE membership in our prize jackpot.",
]

# Clean and transform the new messages
new_messages_cleaned = [clean_text(msg) for msg in new_messages]
new_messages_tfidf = tfidf_vectorizer.transform(new_messages_cleaned)

# Predict the class for the new messages
predictions = model.predict(new_messages_tfidf)
prediction_labels = ['Spam' if pred == 1 else 'Ham' for pred in predictions]

print("\n--- Predictions for New Samples ---")
for i, msg in enumerate(new_messages):
    print(f"Message: '{msg}'")
    print(f"Predicted as: {prediction_labels[i]}\n")