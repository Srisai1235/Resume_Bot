import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Define intents
intents = [
    {
        "tag": "Career Objective",
        "patterns": ["objective", "self introduction", "introduction","self"],
        "responses": ["Passionate undergrad student of AI and Data Science, with skills in Python programming, machine learning and predictive modeling"]
    },
    {
        "tag": "leetcode",
        "patterns": ["LeetCode", "leetcode link", "coding practice", "leetcode profile"],
        "responses": ["Here is my LeetCode profile: https://leetcode.com/u/srisai_777/"]
    },
    {
        "tag": "github",
        "patterns": ["GitHub", "github link", "code repository"],
        "responses": ["Here is my GitHub profile: https://github.com/Srisai1235"]
    },
    {
        "tag": "linkedin",
        "patterns": ["LinkedIn", "linkedin link", "professional profile", "connect with you"],
        "responses": ["Here is my LinkedIn profile: https://www.linkedin.com/in/munipalle-sri-sai-5b6613235"]
    },
    {
        "tag": "email",
        "patterns": ["email", "contact email", "send you an email", "mail"],
        "responses": ["You can email me at: srisaimunipalle9335@gmail.com"]
    },
    {
        "tag": "Mini Project",
        "patterns": ["mini project", "academic project", "group project"],
        "responses": ["""Fake Currency Detection - DEEP LEARNING
        This Fake Indian Currency Detection project uses a pre-trained deep learning model to classify currency images as”Fake” or ”Real.” """]
    },
    {
        "tag": "Projects",
        "patterns": ["projects", "individual projects"],
        "responses": ["""**Projects Completed:**\n
        1) Fake Currency Detection - DEEP LEARNING\n
    2) Market Basket Analyzer - MACHINE LEARNING\n
    3) Air Canvas Traffic Light Colors - COMPUTER VISION\n
    4) Credit Card Fraud Detection -MACHINE LEARNING\n
    5) HousePrice Prediction - MACHINE LEARNING\n"""]
    },
    {
        "tag": "phone",
        "patterns": ["phone number", "contact number", "call you", "mobile"],
        "responses": ["You can reach me at: +91 6304447377"]
    },
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": ["Hi there!", "Hello!", "Hey! How can I assist you today?"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye!", "See you later!", "Take care and have a great day!"]
    },
    {
        "tag": "fallback",
        "patterns": [],
        "responses": ["I'm sorry, I didn't understand that. Can you rephrase?", "Could you clarify what you mean?"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help"]
    },
    {
        "tag": "education",
        "patterns": ["education", "current education", "qualification", "college","B.Tech"],
        "responses": ["I am currently pursuing B.Tech in CSE with a specialization in AI and Data Science at GNITC Ibrahimpatnam with an 8.9 CGPA."]
    },
    {
        "tag": "Schooling",
        "patterns": ["10th", "SSC", "School", "10", "ssc"],
        "responses": ["I have completed my schooling in Sri Krishnaveni Group of Schools, Zaheerabad with a CGPA of 9.7"]
    },
    {
        "tag": "Intermediate",
        "patterns": ["intermediate", "junior college", "+12", "inter", "tsbie"],
        "responses": ["I have completed my Intermediate in Narayana Junior College, Lingampally with a Percentage of 88%"]
    },
    {
        "tag": "Hobbies",
        "patterns": ["Hobbies", "interests", "free time"],
        "responses": [""" **Hobbies**\n
        • Exploring new technologies\n
    • Watching movie\n
    • Painting\n"""]
    },
     {
        "tag": "Certifications",
        "patterns": ["certifications", "certificates"],
        "responses": ["""**Certifications**\n
        • Completion of Salesforce Supported Virtual Internship Program(2024)\n
    • NPTEL-Ellite Grade Certificate- Python, Data Science, Data Analytics\n
    • Bharat Internship-Data Science(2023)\n
    • Machine Learning (2023)\n
    • OOPs using Java (2023)\n
    • Participation in STARTUP ECOSYSTEM AWARENESS PROGRAM\n"""]
    }

]

# Vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Function to get chatbot response
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm sorry, I didn't understand that. Can you please rephrase?"

# Custom CSS styling
def add_custom_css():
    st.markdown("""
    <style>
    .main {
        background-color: #6ED6F0;
    }
    h1 {
        color: #4B0082;
        text-align: center;
        font-size: 45px;
    }
    .response-box {
        background-color: #E6E6FA;
        color: #4B0082;
        border-radius: 15px;
        padding: 15px;
        margin: 15px 0px;
        font-size: 18px;
        font-weight: bold;
    }
    .input-box {
        background-color: #C3C3F7;
        border: 2px solid #D8BFD8;
        border-radius: 15px;
        padding: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit UI
def main():
    add_custom_css()
    st.markdown("<h1>🤖 M.Sri Sai Resumebot</h1>", unsafe_allow_html=True)
    st.markdown("#### Welcome! Feel free to ask me anything about my career, projects, or education! 🌟")

    # Input text
    user_input = st.text_input("Type your question here:", key="user_input")

    if user_input:
        response = chatbot(user_input)

        # Display chatbot response in a styled container
        st.markdown(f"<div class='response-box'>{response}</div>", unsafe_allow_html=True)

        # Conditional goodbye message
        if "goodbye" in response.lower():
            st.success("Thank you for chatting! Have a great day! 👋")
            st.stop()

if __name__ == '__main__':
    main()
