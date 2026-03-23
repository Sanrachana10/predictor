# 🎓 Student Placement Prediction System
### A Hybrid Ensemble Approach (SVM + LSTM + RoBERTa)

This project implements a multi-modal AI system to predict student placement outcomes based on academic and skill-based features.

## 🚀 Key Features
- **Hybrid Architecture**: Combines Classic ML (SVM), Deep Learning (LSTM), and Transformers (RoBERTa).
- **Weighted Soft Voting**: Uses a 20/30/50 weighting system to balance contextual intuition with hard data.
- **Bias Mitigation**: Calibrated thresholds (0.40) to ensure fair predictions for borderline candidates.

## 📁 Folder Structure
- `app.py`: Streamlit web interface.
- `roberta_model/`: Serialized Transformer weights.
- `lstm_model.h5`: Trained Deep Learning model.
- `svm_model.joblib`: Baseline SVM classifier.