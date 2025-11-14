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

```
sentiment_analysis_app/
│
├── app_streamlit.py # Streamlit frontend
├── app.py # Local testing script (optional)
├── sentiment_lr_model.pkl # Logistic Regression model
├── sentiment_nb_model.pkl # Naive Bayes model
├── tfidf_vectorizer.pkl # TF-IDF Vectorizer
├── requirements.txt # Dependencies
├── assets/
│ ├── ss1.png # Screenshot 1
│ ├── ss2.png # Screenshot 2
│ └── ss3.png # Screenshot 3
└── README.md # Project documentation

```

---

## 📸 App Screenshots

### 🖼️ Screenshot 1
![Screenshot 1](assets/ss1.png)

### 🖼️ Screenshot 2
![Screenshot 2](assets/ss2.png)

### 🖼️ Screenshot 3
![Screenshot 3](assets/ss3.png)

---

## 🚀 Run the App Locally

---

## 🙌 Author  
Developed by **Aniqua Nawar** as part of a learning project on **Natural Language Processing (NLP)** and **Streamlit**.


