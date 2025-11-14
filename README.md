# 🎬 Sentiment Analysis App (Streamlit)

This is an end-to-end **Sentiment Analysis Web App** built using **Python, Scikit-Learn, and Streamlit**.  
It predicts whether a given review is **Positive** or **Negative** using:

- **Logistic Regression Model**
- **Naive Bayes Model**
- **TF-IDF Vectorizer**

The app also includes:
- 🌥️ **Word Cloud Visualization**
- 📊 **Sentiment Summary Pie Chart**
- ✨ Clean UI built with Streamlit



## 🚀 Features

### ✔️ **Two ML Models for Prediction**
- Logistic Regression  
- Multinomial Naive Bayes  

Both models are trained on TF-IDF vectors.

### ✔️ **Text Preprocessing**
- Lowercasing  
- Removing punctuation  
- Keeping only alphabets  

### ✔️ **Visualizations**
- **Word Cloud** of the input review  
- **Sentiment pie chart** based on positive vs negative keywords  

### ✔️ **Fully Interactive Web Interface**
Built using Streamlit for instant predictions.



## 📂 Project Structure
sentiment_analysis_app/
│
├── app_streamlit.py          # Streamlit frontend
├── app.py                    # Local testing script (optional)
├── sentiment_lr_model.pkl    # Logistic Regression model
├── sentiment_nb_model.pkl    # Naive Bayes model
├── tfidf_vectorizer.pkl      # TF-IDF Vectorizer
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation
# My Sentiment Analysis App

Here is how the app looks:

![App screenshot](assets/screenshot.png)


