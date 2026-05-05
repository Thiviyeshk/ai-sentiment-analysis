import streamlit as st
import pandas as pd
import pickle
from collections import Counter

st.title("AI Sentiment Analytics Platform")

# Load model
model = pickle.load(open("sentiment_model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

# Load dataset
df = pd.read_csv("cleaned_tweets.csv")

# ----------------------------
# Sentiment Dashboard
# ----------------------------

st.subheader("Sentiment Distribution")

st.bar_chart(df['sentiment'].value_counts())

# ----------------------------
# Keyword Analysis
# ----------------------------

st.subheader("Top Keywords")

words = " ".join(df['clean_text'])

word_list = words.split()

top_words = Counter(word_list).most_common(10)

top_words_df = pd.DataFrame(top_words, columns=["Keyword","Count"])

st.table(top_words_df)

# ----------------------------
# Interactive Prediction
# ----------------------------

st.subheader("Try Sentiment Prediction")

user_input = st.text_area("Enter a Tweet")

if st.button("Predict Sentiment"):

    text_vec = vectorizer.transform([user_input])

    prediction = model.predict(text_vec)

    st.write("Prediction:", prediction[0])

    if prediction[0] == "positive":
        st.success("😊 Positive Sentiment")

    elif prediction[0] == "negative":
        st.error("😡 Negative Sentiment")

    else:
        st.info("😐 Neutral Sentiment")

# ----------------------------
# AI Insights
# ----------------------------

if st.button("Generate AI Insights"):

    st.write("""
    • Most tweets show positive sentiment.
    
    • Negative sentiment mostly relates to product issues.
    
    • Improving customer support may increase positive sentiment.
    """)