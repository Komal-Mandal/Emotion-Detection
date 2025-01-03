import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords")
stopwords = set(stopwords.words("english"))

def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)



def predict_emotion(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf.transform([cleaned_text])

    # Predict emotion
    predicted_label = model.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]
    label =  np.max(model.predict(input_vectorized))

    return predicted_emotion,label


model = pickle.load(open("logistic_regression.pkl","rb"))
lb = pickle.load(open("label_encoder.pkl","rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl","rb"))

st.set_page_config(page_title="Emotion Detection App", page_icon="😊", layout="wide")
st.title("😊 Six NLP Emotions Detection App")

st.write("=================================================")
st.write("['Joy,'Fear','Anger','Love','Sadness','Surprise']")
st.write("=================================================")

input_text = st.text_input("Enter your text")

if st.button("predict"):
    emotion, level = predict_emotion(input_text)
    st.write(f"Predicted Emotion: {emotion}")
    st.write(f"Emotion Level: {level:.2f}")





