import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# load stopwords
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# load model + vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# cleaning function (same as notebook)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s!]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# UI
st.set_page_config(page_title="Fake Review Detector", layout="centered")

st.title("🧠 Fake Review Detector")
st.write("Enter a review and check if it's fake or real.")

user_input = st.text_area("Enter your review:")

if st.button("Analyze"):
    cleaned = clean_text(user_input)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    proba = model.predict_proba(vector)

    if prediction[0] == 1:
        st.error("🚨 Fake Review")
    else:
        st.success("✅ Real Review")

    st.write("Confidence:", round(max(proba[0]) * 100, 2), "%")
