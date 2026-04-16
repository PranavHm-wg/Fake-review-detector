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

# Custom CSS for better look
st.markdown("""
    <style>
        .main {background-color: #0e1117;}
        .title {text-align: center; font-size: 36px; font-weight: bold; color: #00ffcc;}
        .subtitle {text-align: center; font-size: 16px; color: #cccccc; margin-bottom: 20px;}
        .stButton>button {
            background-color: #00ffcc;
            color: black;
            border-radius: 8px;
            height: 3em;
            width: 100%;
            font-size: 16px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title"> Fake Review Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered NLP model to detect fake vs real reviews</div>', unsafe_allow_html=True)

# Input box
user_input = st.text_area(
    "✍️ Enter your review here:",
    height=150,
    placeholder="Try: 'This product exceeded expectations and works perfectly...'",
)

st.markdown("<br>", unsafe_allow_html=True)
analyze_clicked = st.button("🔍 Analyze Review")

if user_input and analyze_clicked:
    with st.spinner("Analyzing review..."):
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)
        proba = model.predict_proba(vector)

    confidence = round(max(proba[0]) * 100, 2)

    st.markdown("---")

    # Handle both numeric and string labels
    pred = prediction[0]

    if pred == 1 or pred == "CG":
        st.error(f" Fake Review Detected ({confidence}% confidence)")
    else:
        st.success(f" Real Review ({confidence}% confidence)")

    st.write("###  Model Confidence Breakdown")
    st.write(f"Real Probability: {round(proba[0][1]*100, 2)}%")
    st.write(f"Fake Probability: {round(proba[0][0]*100, 2)}%")
