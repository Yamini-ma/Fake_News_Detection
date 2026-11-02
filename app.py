
import streamlit as st
import joblib
import re

# Load the trained machine learning model
model = joblib.load("models/best_model.joblib")

# Configure the webpage
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="centered"
)

# Application title
st.title("📰 Fake News Detection System")

st.write(
    "This application uses Machine Learning "
    "to classify news articles as FAKE or REAL."
)

st.divider()

# News article input
news_text = st.text_area(
    "Enter your news article below:",
    height=250,
    placeholder="Paste or type your news article here..."
)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Prediction button
if st.button("Detect News", type="primary"):

    if news_text.strip():

        cleaned_news = clean_text(news_text)

        prediction = model.predict([cleaned_news])[0]

        st.subheader("Prediction Result")

        if prediction == 0:
            st.error("⚠️ FAKE NEWS DETECTED")
            st.write("The model predicts that this article is FAKE.")
        else:
            st.success("✅ REAL NEWS DETECTED")
            st.write("The model predicts that this article is REAL.")

        st.caption(
            "Note: This is a machine-learning prediction, "
            "not independent verification of the news."
        )

    else:
        st.warning("Please enter a news article first.")
