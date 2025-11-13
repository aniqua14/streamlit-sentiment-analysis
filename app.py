import pickle

# Load saved models and vectorizer
with open("sentiment_lr_model.pkl", "rb") as f:
    lr = pickle.load(f)

with open("sentiment_nb_model.pkl", "rb") as f:
    nb = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Function to preprocess text (same logic as before)


def preprocess_text(text):
    import re
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text


# Test sentiment prediction
text = ["I absolutely loved this movie!"]
cleaned = [preprocess_text(t) for t in text]
vectorized = tfidf.transform(cleaned)

# Predict using Logistic Regression
prediction_lr = lr.predict(vectorized)[0]

# Predict using Naive Bayes
prediction_nb = nb.predict(vectorized)[0]

print(f"Logistic Regression Prediction: {prediction_lr}")
print(f"Naive Bayes Prediction: {prediction_nb}")
