import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Placement Predictor AI", page_icon="🎓", layout="centered")

# --- 2. LOAD MODELS (Cached for Speed) ---
@st.cache_resource
def load_all_models():
    # Preprocessors (Indented 4 spaces)
    svm = joblib.load('svm_model.joblib')
    tfidf = joblib.load('tfidf_vectorizer.joblib')
    le = joblib.load('label_encoder.joblib')
    
    # LSTM Deep Learning (Indented 4 spaces)
    lstm_model = tf.keras.models.load_model('lstm_model.keras')
    keras_tok = joblib.load('keras_tokenizer.joblib')
    
    # RoBERTa Transformer (Indented 4 spaces)
    r_tokenizer = AutoTokenizer.from_pretrained("Sanrachana/student-placement-roberta")
    r_model = AutoModelForSequenceClassification.from_pretrained("Sanrachana/student-placement-roberta")
    
    # The return statement MUST be inside the function (Indented 4 spaces)
    return svm, tfidf, le, lstm_model, keras_tok, r_tokenizer, r_model

# Initialize models (Outside the function, at the left margin)
svm, tfidf, le, lstm, k_tok, r_tok, r_mod = load_all_models()

# --- 3. UI DESIGN ---
st.title("🎓 Student Placement Prediction System")
st.markdown("This system uses a **Hybrid Ensemble AI** (SVM + LSTM + RoBERTa) to predict placement probability.")
st.divider()

# Input Columns
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

# --- 4. PREDICTION LOGIC ---
if st.button("Predict Placement Status", type="primary"):
    # Create the text string for the NLP models
    input_text = f"{degree} {branch} CGPA {cgpa}"
    
    # A. SVM Prediction
    p_svm = svm.predict_proba(tfidf.transform([input_text]))[0][:2]
    p_svm /= p_svm.sum() # Normalize
    
    # B. LSTM Prediction
    seq = k_tok.texts_to_sequences([input_text])
    pad = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=100)
    p_lstm = lstm.predict(pad, verbose=0)[0][:2]
    p_lstm /= p_lstm.sum() # Normalize
    
    # C. RoBERTa Prediction
    inputs = r_tok(input_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = r_mod(**inputs).logits
        p_roberta = torch.softmax(logits, dim=1).detach().cpu().numpy()[0][:2]
    p_roberta /= p_roberta.sum() # Normalize

    # --- 5. ENSEMBLE WEIGHTING ---
    # RoBERTa(50%), LSTM(30%), SVM(20%)
    final_prob = (p_svm[1] * 0.2) + (p_lstm[1] * 0.3) + (p_roberta[1] * 0.5)

    # --- 6. FINAL DECISION & OUTPUT ---
    st.subheader("Results Analysis")
    
    if backlogs > 2:
        st.error("Prediction: NOT PLACED ❌")
        st.warning("Reason: High academic risk due to active backlogs.")
    elif final_prob >= 0.40: 
        st.success(f"Prediction: PLACED 🎉 (Confidence: {final_prob*100:.1f}%)")
        st.balloons()
    else:
        st.error(f"Prediction: NOT PLACED ❌ (Confidence Score: {final_prob*100:.1f}%)")
        st.info("Recommendation: Focus on building technical projects and improving coding skills.")

st.divider()
st.caption("Developed for Pre-Final Year Lab Assignment - 2026")
