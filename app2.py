import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import torch

# ================================
# Configure Gemini API
# ================================
genai.configure(api_key="YOUR_GEMINI_API_EKY")  # <-- Replace with your real key

# ================================
# Load Models
# ================================
bert_model = pickle.load(open("bert_fake_news_model.pkl", "rb"))
device = torch.device("cpu")  # Change to "cuda" if GPU is available
bert_vectorizer = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
   # or "cuda"


# ================================
# Streamlit Page Config
# ================================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Custom CSS for better UI
st.markdown(
    """
    <style>
    .stTextArea textarea {
        font-size: 16px;
        border-radius: 10px;
        padding: 10px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .result-box {
        background: #2b2b2b;
        color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
        font-size: 16px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ================================
# App Title
# ================================
st.title("📰 Fake News Detection (BERT + Gemini)")
st.caption("Detect whether a news article is Fake or Real using Machine Learning + Generative AI")

# Input from user
news_text = st.text_area("✍️ Enter a news headline or article:", height=150)

if st.button("🔍 Check News"):
    if news_text.strip():
        # -------------------
        # 1. BERT Prediction
        # -------------------
        embedding = bert_vectorizer.encode([news_text])
        bert_pred = bert_model.predict(embedding)[0]
        bert_conf = max(bert_model.predict_proba(embedding)[0]) * 100
        bert_result = "Real" if bert_pred == 0 else "Fake"

        st.subheader("📊 BERT Model Result")
        st.progress(int(bert_conf))
        st.markdown(
            f"<div class='result-box'>Prediction: {bert_result}<br>Confidence: {bert_conf:.2f}%</div>",
            unsafe_allow_html=True
        )

        # -------------------
        # 2. Gemini Prediction
        # -------------------
        try:
            response = genai.GenerativeModel("gemini-1.5-flash").generate_content(
                f"""
                Analyze the following news article and classify it as 'Fake' or 'Real'.
                After giving Fake/Real, provide a short explanation in 2-3 sentences.

                Article:
                {news_text}
                """
            )
            gemini_full = response.text.strip()
            # Extract first word (Fake/Real) + explanation
            gemini_result = gemini_full.split()[0] if gemini_full else "Unknown"
            gemini_explanation = " ".join(gemini_full.split()[1:]).strip()
        except Exception as e:
            gemini_result = "Error"
            gemini_explanation = f"Gemini API Error: {e}"

        st.subheader("🤖 Gemini AI Result")
        st.markdown(
            f"<div class='result-box'>Prediction: {gemini_result}<br>Explanation: {gemini_explanation}</div>",
            unsafe_allow_html=True
        )

        # -------------------
        # 3. Final Decision (Prioritize Gemini)
        # -------------------
        if gemini_result.lower() == bert_result.lower():
            final_decision = f"✅ Final Decision: {bert_result} (Both Agree)"
        else:
            final_decision = f"⚖️ Final Decision: {gemini_result} (Gemini trusted, BERT disagrees)"

        st.subheader("🔎 Final Decision")
        st.success(final_decision)

    else:
        st.warning("⚠️ Please enter some news text.")
