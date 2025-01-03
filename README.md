# 😊 Emotion Detection App

The Emotion Detection App is a user-friendly web application designed to identify emotions in textual data using natural language processing (NLP). It leverages machine learning models to classify input text into one of six emotional categories: Joy, Fear, Anger, Love, Sadness, Surprise.

# 🚀 Key Features

Multi-Emotion Detection: Classifies text into one of six emotions: Joy, Fear, Anger, Love, Sadness, and Surprise.

Deep Learning Power: Utilizes an LSTM model for contextual emotion prediction.

Interactive Web Interface: Built with Streamlit, ensuring an intuitive and responsive user experience.

Explainable Results: Displays the predicted emotion and its intensity level.

# 🌟 Walkthrough of Backend Code

# 🛠️ How It Works

# 1. Text Preprocessing
Input text is cleaned by:

Removing non-alphabetic characters.

Converting to lowercase.

Removing stopwords (common words like "the", "is").

Stemming words to their root forms (e.g., "running" → "run").

# 2. Feature Extraction
Text is transformed into a numerical representation using TF-IDF Vectorization.

# 3. Emotion Classification
The processed text is passed through a logistic regression model to predict the most likely emotion.
The app also outputs the intensity level of the emotion.

# ⚙️ Technologies Used
Python: Programming language.

Streamlit: Web application framework.

LSTM (Keras/TensorFlow): For deep learning-based emotion detection.

scikit-learn: Used for preprocessing and traditional ML models.

NLTK: Natural Language Toolkit for text processing

# 🌟 Walkthrough of  Code

![Screenshot 2025-01-03 230559](https://github.com/user-attachments/assets/e0a7afd7-6ad8-4683-9afd-522202e379df)

The code preprocesses text by cleaning and simplifying it for analysis. It removes non-alphabetic characters, converts text to lowercase, splits it into words, removes common stopwords, and stems words to their root forms using the PorterStemmer. Finally, the cleaned words are rejoined into a single string. For example, the input "I absolutely LOVE this amazing app!!!" is transformed into "absolut love amaz app"

![Screenshot 2025-01-03 230540](https://github.com/user-attachments/assets/88e3a15e-d26e-4bec-803b-1dc9be27fa11)

The predict_emotion function analyzes input text to identify its emotion. It cleans the text, transforms it into numerical features using tfidf_vectorizer, and uses a logistic regression model (lg) to predict the emotion. The predicted label is decoded into a readable emotion (e.g., "Joy") with a confidence score. For example, the input "I am so happy today!" might return "Joy" with 95% confidence.

# 🖥️ Installation & Setup

# Prerequisites

1.Python 3.8 or above
2.Streamlit
3.Required libraries: nltk, pandas, scikit-learn, numpy

# Steps

# 1.Clone the Repository

![Screenshot 2025-01-03 231836](https://github.com/user-attachments/assets/8ae33a23-4db9-4d61-86c5-00be0b9effef)

# 2.Install Dependencies

![Screenshot 2024-12-19 004107](https://github.com/user-attachments/assets/b73af8cd-6062-4d09-91e5-24d502e6dedb)

# 3.Download NLTK Data

![Screenshot 2025-01-03 233044](https://github.com/user-attachments/assets/cb73e997-ac6e-4cdc-a5a6-13b05f5c475d)

# 4.Run the Application

![Screenshot 2025-01-03 233100](https://github.com/user-attachments/assets/554f1757-f9e5-4715-ba47-659c0d5e56ba)

# 🌐 Deployment

This project is deployed using Streamlit Community Cloud, allowing users to access it directly without installation.

Steps to Deploy on Streamlit Community Cloud:

1.Create a Streamlit Community Cloud account at Streamlit Share.

2.Connect your GitHub repository containing the project.

3.Select the streamlit(emotion).py file as the main entry point.

4.Deploy the app and share the generated link!
link:https://emotion-detection-7vpkzpzalmrgz2msm3c6hd.streamlit.app/






