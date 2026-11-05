import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import json
import random
import joblib

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Load trained model and files
@st.cache_resource
def load_chatbot():
    model = load_model("Chatbot.keras")
    classes = joblib.load("classes.pkl")
    vector = joblib.load("Vector.pkl")

    with open("data.json", "r") as file:
        data = json.load(file)

    return model, classes, vector, data

model, classes, vector, data = load_chatbot()

# Chatbot response function
def chatbot_response(user):
    user_vector = vector.transform([user]).toarray()
    prediction = model.predict(user_vector, verbose=0)[0]

    index = np.argmax(prediction)
    tag = classes[index]

    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, I don't understand."

# UI
st.title("🤖 AI Chatbot")
st.caption("Deep Learning Based Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    response = chatbot_response(user_input)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    st.rerun()
