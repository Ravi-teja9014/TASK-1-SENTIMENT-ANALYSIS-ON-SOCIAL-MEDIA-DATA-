import nltk
nltk.download('punkt')
import nltk
nltk.download('stopwords')

# Text preprocessing
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back into string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Preprocess data
preprocessed_data = [(preprocess_text(text), label) for text, label in data]

# Feature extraction
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform([text for text, _ in preprocessed_data])
y = [label for _, label in preprocessed_data]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection and training
model = MultinomialNB()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Example usage
def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    vectorized_text = tfidf_vectorizer.transform([preprocessed_text])
    sentiment = model.predict(vectorized_text)[0]
    return sentiment

# Example usage
text_to_predict = "I'm feeling happy today"
predicted_sentiment = predict_sentiment(text_to_predict)
print("Predicted sentiment:", predicted_sentiment)
