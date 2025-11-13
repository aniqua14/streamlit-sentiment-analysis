import streamlit as st
import pickle
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns


# Load Models and Vectorizer

with open("sentiment_lr_model.pkl", "rb") as f:
    lr = pickle.load(f)

with open("sentiment_nb_model.pkl", "rb") as f:
    nb = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)


# Text Preprocessing Function


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text


# Streamlit App
st.set_page_config(page_title="Sentiment Analysis App", page_icon="🎬")

st.title("🎬 Sentiment Analysis on Movie/Product Reviews")
st.write("This app predicts whether a review is **Positive** or **Negative** using two models.")

# User input
user_input = st.text_area(
    "Enter a review:", placeholder="Type your review here...")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        cleaned = [preprocess_text(user_input)]
        vectorized = tfidf.transform(cleaned)

        pred_lr = lr.predict(vectorized)[0]
        pred_nb = nb.predict(vectorized)[0]

        st.subheader("Results:")
        st.write(f"**Logistic Regression Prediction:** {pred_lr}")
        st.write(f"**Naive Bayes Prediction:** {pred_nb}")

        if pred_lr == "positive" or pred_nb == "positive":
            st.success("🌟 This review is likely Positive!")
        else:
            st.error("😞 This review is likely Negative.")

        # WORD CLOUD SECTION

        st.subheader("Word Cloud of Review Text")
        wordcloud = WordCloud(width=800, height=400,
                              background_color='white').generate(user_input)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

        # PIE CHART SECTION

        st.subheader("Sentiment Word Proportion (Pie Chart)")

        # Simple word-based sentiment count
        positive_words = ["good", "great", "amazing", "love",
                          "wonderful", "excellent", "best", "beautiful", "happy"]
        negative_words = ["bad", "terrible", "awful", "worst",
                          "boring", "poor", "sad", "hate", "disappointing"]

        text_words = preprocess_text(user_input).split()

        pos_count = sum(1 for word in text_words if word in positive_words)
        neg_count = sum(1 for word in text_words if word in negative_words)

        labels = ['Positive Words', 'Negative Words']
        sizes = [pos_count, neg_count]

        if pos_count == 0 and neg_count == 0:
            st.info("No clear positive or negative words detected.")
        else:
            colors = ['#4CAF50', '#F44336']
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                    startangle=90, colors=colors)
            ax1.axis('equal')
            st.pyplot(fig1)
