
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Placement Predictor AI", page_icon="🎓", layout="centered")

# --- 2. LOAD MODELS (Cached for Speed) ---
@st.cache_resource
def load_all_models():
    svm = joblib.load('svm_model.joblib')
    tfidf = joblib.load('tfidf_vectorizer.joblib')
    le = joblib.load('label_encoder.joblib')
    lstm_model = tf.keras.models.load_model('lstm_model.keras')
    keras_tok = joblib.load('keras_tokenizer.joblib')
    r_tokenizer = AutoTokenizer.from_pretrained('Sanrachana/student-placement-roberta')
    r_model = AutoModelForSequenceClassification.from_pretrained('Sanrachana/student-placement-roberta')
    return svm, tfidf, le, lstm_model, keras_tok, r_tokenizer, r_model

svm, tfidf, le, lstm, k_tok, r_tok, r_mod = load_all_models()

# --- 3. TEXT CLEANING (matches training) ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# --- 4. UI DESIGN ---
st.title("🎓 Student Placement Prediction System")
st.markdown("This system uses a **Hybrid Ensemble AI** (SVM + LSTM + RoBERTa + Numerical Analysis) to predict placement probability.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    degree = st.selectbox("Degree", ["B.Tech", "M.Tech", "BCA", "MCA"])
    branch = st.text_input("Branch", "Computer Science")
    cgpa = st.slider("Current CGPA", 0.0, 10.0, 7.5)
    backlogs = st.number_input("Active Backlogs", 0, 10, 0)

with col2:
    interns = st.number_input("Internships Done", 0, 5, 0)
    projects = st.number_input("Major Projects", 0, 10, 1)
    coding = st.slider("Coding Skill (1-10)", 1, 10, 5)
    aptitude = st.slider("Aptitude Score (%)", 0, 100, 70)

# --- 5. PREDICTION LOGIC ---
if st.button("Predict Placement Status", type="primary"):

    # Hard block for high backlogs
    if backlogs > 2:
        st.error("Prediction: NOT PLACED ❌")
        st.warning("Reason: High academic risk due to active backlogs.")
    else:
        # Prepare text input (matches training format)
        input_text = clean_text(f"{degree} {branch} CGPA {cgpa}")

        # A. SVM Prediction
        p_svm = svm.predict_proba(tfidf.transform([input_text]))[0][:2]
        p_svm = p_svm / p_svm.sum()

        # B. LSTM Prediction
        seq = k_tok.texts_to_sequences([input_text])
        pad = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=100)
        p_lstm = lstm.predict(pad, verbose=0)[0][:2]
        p_lstm = p_lstm / p_lstm.sum()

        # C. RoBERTa Prediction
        inputs = r_tok(input_text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = r_mod(**inputs).logits
            p_roberta = torch.softmax(logits, dim=1)[0][:2].tolist()
            p_roberta = [x / sum(p_roberta) for x in p_roberta]

        # D. Numerical Score (based on your actual training features)
        cgpa_score     = (cgpa / 10) * 0.40
        intern_score   = (interns / 5) * 0.25
        coding_score   = (coding / 10) * 0.20
        aptitude_score = (aptitude / 100) * 0.15
        numerical_score = cgpa_score + intern_score + coding_score + aptitude_score

        # --- 6. ENSEMBLE WEIGHTING ---
        # RoBERTa(30%) + LSTM(20%) + SVM(10%) + Numerical(40%)
        final_prob = (
            p_svm[1]      * 0.10 +
            p_lstm[1]     * 0.20 +
            p_roberta[1]  * 0.30 +
            numerical_score * 0.40
        )

        # --- 7. FINAL DECISION & OUTPUT ---
        st.subheader("Results Analysis")

        if final_prob >= 0.50:
            st.success(f"Prediction: PLACED 🎉 (Confidence: {final_prob*100:.1f}%)")
            st.balloons()

            # Tips for placed students
            st.info("💡 Keep it up! Focus on interview prep and building your portfolio.")
        else:
            st.error(f"Prediction: NOT PLACED ❌ (Confidence Score: {final_prob*100:.1f}%)")
            st.info("💡 Recommendation: Improve CGPA, complete internships, and build coding skills.")

        # --- 8. SCORE BREAKDOWN ---
        st.divider()
        st.markdown("**Score Breakdown**")
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("SVM", f"{p_svm[1]*100:.0f}%")
        col_b.metric("LSTM", f"{p_lstm[1]*100:.0f}%")
        col_c.metric("RoBERTa", f"{p_roberta[1]*100:.0f}%")
        col_d.metric("Numerical", f"{numerical_score*100:.0f}%")

st.divider()
st.caption("Developed for Pre-Final Year Lab Assignment - 2026")
