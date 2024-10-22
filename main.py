import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import nltk
from nltk.corpus import stopwords
import string

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# Step 1: Keep only the first two columns (assuming there are extra columns)
df = df.iloc[:, [0, 1]]

# Step 2: Rename columns to 'label' and 'message'
df.columns = pd.Index(['label', 'message'])

# Step 3: Data Cleaning
# Remove missing values
df.dropna(inplace=True)

# Remove rows where 'message' is empty
df = df[df['message'] != '']

# Convert all messages to lowercase
df['message'] = df['message'].str.lower()

# Remove special characters from messages
df['message'] = df['message'].str.replace(r'[^\w\s]', '', regex=True)

# Step 4: Text Vectorization using TF-IDF
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['message'])

# Step 5: Convert labels to binary (spam = 1, ham = 0)
y = df['label'].map({'ham': 0, 'spam': 1})

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 8: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
print('Confusion Matrix:')
print(conf_matrix)

# Function to classify new messages
def classify_message(message):
    # Clean the message
    message = message.lower()
    message = message.translate(str.maketrans('', '', string.punctuation))
    vectorized_message = tfidf.transform([message])
    prediction = model.predict(vectorized_message)
    return 'Spam' if prediction == 1 else 'Not Spam'

# Test with a new message
test_message = "HEllo How are You?."
print(f'Test Message: {test_message}')
print(f'Prediction: {classify_message(test_message)}')
