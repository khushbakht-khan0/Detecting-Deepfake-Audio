import streamlit as st
import numpy as np
import librosa
import joblib
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer

# --- Load Models (Part 1 - Audio) ---
audio_models = {
    "SVM": joblib.load("models/svm_audio_model.pkl"),
    "Logistic Regression": joblib.load("models/lr_audio_model.pkl"),
    "Perceptron": joblib.load("models/perceptron_audio_model.pkl"),
    "DNN": tf.keras.models.load_model("models/dnn_audio_model.h5")
}

# --- Load Models & TF-IDF (Part 2 - Defects) ---
defect_models = {
    "SVM": joblib.load("models/svm_defect_model.pkl"),
    "Logistic Regression": joblib.load("models/lr_defect_model.pkl"),
    "Perceptron": joblib.load("models/perceptron_defect_model.pkl")
}
tfidf = joblib.load("models/tfidf.pkl")

# --- Streamlit App ---
st.set_page_config(page_title="Deepfake & Defect Predictor", layout="centered")
st.title("🔍 Deepfake Audio & 🐞 Software Defect Predictor")

tab1, tab2 = st.tabs(["🎧 Audio Prediction", "📝 Defect Prediction"])

# -------------------- TAB 1: AUDIO PREDICTION --------------------
with tab1:
    st.header("🎧 Upload an Audio File")
    audio_file = st.file_uploader("Upload a .wav file", type=["wav"])
    audio_model_choice = st.selectbox("Choose a model for audio prediction", list(audio_models.keys()))

    if audio_file and st.button("Predict Audio"):
        try:
            y, sr = librosa.load(audio_file, sr=16000)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features = np.mean(mfcc.T, axis=0).reshape(1, -1)
            model = audio_models[audio_model_choice]

            if audio_model_choice == "DNN":
                prob = model.predict(features)[0][0]
            else:
                prob = model.predict_proba(features)[0][1] if hasattr(model, "predict_proba") else model.predict(features)[0]
            
            label = "🟢 Bonafide" if prob < 0.5 else "🔴 Deepfake"
            st.markdown(f"### Prediction: {label}")
            st.write(f"**Confidence Score**: {prob:.2f}")
        except Exception as e:
            st.error(f"Error processing audio: {e}")

# -------------------- TAB 2: DEFECT PREDICTION --------------------
label_names = ['Security', 'Performance', 'Crash', 'UI', 'Functional', 'Usability', 'Compatibility']

with tab2:
    st.header("📝 Input Feature Text for Software Defect Prediction")
    input_text = st.text_area("Enter software description or relevant text features:")
    defect_model_choice = st.selectbox("Choose a model for defect prediction", list(defect_models.keys()))

    if input_text and st.button("Predict Defects"):
        try:
            X_transformed = tfidf.transform([input_text])
            model = defect_models[defect_model_choice]
            probs = model.predict_proba(X_transformed)[0]

            # Threshold-based binary prediction
            threshold = 0.5
            predicted = (probs > threshold).astype(int)

            st.markdown("### 🧠 Prediction Results")
            for i, p in enumerate(probs):
                label = label_names[i] if i < len(label_names) else f"Label {i+1}"
                st.write(f"{label}: {'✅' if p > threshold else '❌'} (Confidence: {p:.2f})")

        except Exception as e:
            st.error(f"Error in prediction: {e}")
