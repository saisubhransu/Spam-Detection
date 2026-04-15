# import streamlit as st
# import pickle
# import string
# from nltk.corpus import stopwords
# import nltk
# from nltk.stem.porter import PorterStemmer

# ps = PorterStemmer()

# nltk.download('punkt')


# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)

#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         y.append(ps.stem(i))

#     return " ".join(y)

# tfidf = pickle.load(open('vectorizer.pkl','rb'))
# model = pickle.load(open('model.pkl','rb'))

# st.title("Email/SMS Spam Classifier")

# input_sms = st.text_area("Enter the message")

# if st.button('Predict'):

#     # 1. preprocess
#     transformed_sms = transform_text(input_sms)
#     # 2. vectorize
#     vector_input = tfidf.transform([transformed_sms])
#     # 3. predict
#     result = model.predict(vector_input)[0]
#     # 4. Display
#     if result == 1:
#         st.header("Spam")
#     else:
#         st.header("Not Spam")

import streamlit as st
import pickle
import re

# -------------------------------
# Load model and vectorizer
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -------------------------------
# Text preprocessing (NO NLTK)
# -------------------------------
def transform_text(text):
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove punctuation & special chars
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Tokenize
    words = text.split()
    
    # Remove very short words (optional but helpful)
    words = [word for word in words if len(word) > 2]
    
    return " ".join(words)

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Spam Classifier", page_icon="📩")

st.title("📩 Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter a message")
    else:
        # 1. preprocess
        transformed_sms = transform_text(input_sms)

        # 2. vectorize
        vector_input = vectorizer.transform([transformed_sms])

        # 3. predict
        result = model.predict(vector_input)[0]

        # 4. probability (if available)
        try:
            prob = model.predict_proba(vector_input)[0]
            confidence = max(prob)
        except:
            confidence = None

        # -------------------------------
        # Output
        # -------------------------------
        if result == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")

        if confidence:
            st.info(f"Confidence: {round(confidence * 100, 2)}%")
