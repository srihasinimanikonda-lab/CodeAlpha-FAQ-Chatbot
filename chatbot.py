import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
<style>

/* Hide Streamlit Header */
header {visibility:hidden;}
[data-testid="stToolbar"] {display:none;}
[data-testid="stHeader"] {background:transparent;}
[data-testid="stDecoration"] {display:none;}
[data-testid="stStatusWidget"] {display:none;}

/* Background */
.stApp{
background: linear-gradient(
135deg,
#eef2ff 0%,
#dbeafe 35%,
#ddd6fe 70%,
#f5d0fe 100%
);
}

/* Main Card */
.block-container{
max-width:850px;
padding:2.5rem;
margin-top:20px;

background:rgba(255,255,255,0.60);

backdrop-filter:blur(20px);
-webkit-backdrop-filter:blur(20px);

border-radius:30px;

border:1px solid rgba(255,255,255,0.7);

box-shadow:
0 20px 60px rgba(0,0,0,0.08);
}

/* Title */
.title{
text-align:center;
font-size:52px;
font-weight:800;
color:#1e293b;
margin-bottom:5px;
}

.subtitle{
text-align:center;
font-size:18px;
color:#475569;
margin-bottom:30px;
}

/* Text Input */
.stTextInput input{
border-radius:15px !important;
padding:14px !important;
font-size:18px !important;
background:white !important;
}

/* Button */
.stButton > button{
width:100%;
border:none;
border-radius:15px;
padding:14px;

font-size:18px;
font-weight:700;
color:white;

background:linear-gradient(
90deg,
#38bdf8,
#6366f1
);

box-shadow:
0 10px 25px rgba(99,102,241,0.25);

transition:0.3s ease;
}

.stButton > button:hover{
transform:translateY(-2px);
}

/* User Message */
.user-box{
background:linear-gradient(
90deg,
#3b82f6,
#6366f1
);

color:white;

padding:16px;
border-radius:18px;

font-size:18px;

margin-top:20px;
margin-bottom:10px;

box-shadow:
0 8px 20px rgba(59,130,246,0.25);
}

/* Bot Message */
.bot-box{
background:white;

color:#1e293b;

padding:18px;

border-radius:18px;

font-size:18px;
font-weight:600;

box-shadow:
0 8px 20px rgba(0,0,0,0.06);

margin-bottom:15px;
}

/* Footer */
.footer{
text-align:center;
margin-top:25px;
color:#64748b;
font-size:14px;
}

</style>
""", unsafe_allow_html=True)

faq_data = {
    "question":[
        "What is CodeAlpha?",
        "How do I submit my project?",
        "How many tasks should I complete?",
        "Will I get a certificate?",
        "What is Python?",
        "What is Artificial Intelligence?",
        "What is Machine Learning?",
        "What is NLP?",
        "What is Streamlit?",
        "How do I upload code to GitHub?",
        "What is a chatbot?",
        "What is a translator tool?",
        "What is object detection?",
        "What is YOLO?",
        "How can I learn programming?"
    ],

    "answer":[
        "CodeAlpha is a software development and internship platform.",
        "Submit your project through the provided submission form.",
        "You should complete at least 2 or 3 tasks.",
        "Yes, after successful completion.",
        "Python is a popular programming language.",
        "Artificial Intelligence enables machines to mimic human intelligence.",
        "Machine Learning allows systems to learn from data.",
        "NLP stands for Natural Language Processing.",
        "Streamlit is a framework used to build Python web applications.",
        "Push your project repository to GitHub.",
        "A chatbot automatically responds to user questions.",
        "A translator tool converts text from one language to another.",
        "Object detection identifies objects inside images or videos.",
        "YOLO is a popular real-time object detection model.",
        "Practice consistently and build projects regularly."
    ]
}

df = pd.DataFrame(faq_data)

st.markdown("""
<div class="title">
🤖 AI FAQ Chatbot
</div>

<div class="subtitle">
Ask Questions and Get Instant Answers
</div>
""", unsafe_allow_html=True)

question = st.text_input(
    "Ask Your Question",
    placeholder="Example: What is NLP?"
)

if st.button("🚀 Ask AI"):

    if question.strip():

        vectorizer = TfidfVectorizer()

        vectors = vectorizer.fit_transform(
            df["question"].tolist() + [question]
        )

        similarity = cosine_similarity(
            vectors[-1],
            vectors[:-1]
        )

        index = similarity.argmax()

        answer = df.iloc[index]["answer"]

        st.markdown(
            f"""
            <div class="user-box">
            👤 <b>You:</b> {question}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="bot-box">
            🤖 <b>AI:</b><br><br>
            {answer}
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("""
<div class="footer">
✨ Powered by AI FAQ Assistant
</div>
""", unsafe_allow_html=True)