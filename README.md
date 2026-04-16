# Fake-review-detector

Fake Review Detector

A machine learning + NLP web application that detects whether a product review is fake (computer-generated) or real (human-written).

⸻

 Live Demo

 https://fake-review-detector-pranav.streamlit.app

⸻

 Problem Statement

Fake reviews are a major issue in e-commerce platforms.
This project aims to classify reviews as Fake (CG) or Real (OR) using Natural Language Processing techniques.

⸻

 Features
	•	Text preprocessing (lowercasing, punctuation handling, stopword removal)
	•	TF-IDF vectorization with n-grams
	•	Machine Learning models:
	•	Logistic Regression
	•	Naive Bayes (final model)
	•	Model evaluation using accuracy, confusion matrix, and classification report
	•	Interactive web interface using Streamlit

⸻

 Model Performance
	•	Accuracy: ~89%
	•	Balanced precision and recall across both classes
	•	Handles real-world text inputs via web app

⸻

 Tech Stack
	•	Python
	•	Pandas, NumPy
	•	Scikit-learn
	•	NLTK
	•	Streamlit


fake-review-detector/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── notebook.ipynb
└── data/

Limitations
	•	Model is trained on computer-generated vs original reviews, not exaggerated/spammy reviews
	•	May classify highly emotional or repetitive human reviews as real
	•	Does not currently use advanced features like punctuation intensity or repetition scoring


Future Improvements
	•	Add punctuation and repetition-based features
	•	Use deep learning models (LSTM / Transformers)
	•	Improve UI/UX of the web app
	•	Deploy with API backend

Author

Pranav
