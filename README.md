# 🎙️ Deepfake Audio Detection & 🐞 Multi-Label Software Defect Prediction

This project showcases two machine learning tasks wrapped into one interactive Streamlit app:
- **Binary Classification** of Urdu deepfake audio (Bonafide vs Deepfake)
- **Multi-Label Classification** of software defect descriptions (e.g., Security, Performance, UI)

---

## 🚀 Features

### ✅ Part 1: Urdu Deepfake Audio Detection
- Dataset: [CSALT/deepfake_detection_dataset_urdu](https://huggingface.co/datasets/CSALT/deepfake_detection_dataset_urdu)
- Features: MFCC extraction using `librosa`
- Models Used:
  - Support Vector Machine (SVM)
  - Logistic Regression
  - Perceptron (Single Layer)
  - Deep Neural Network (DNN)

### ✅ Part 2: Software Defect Prediction
- Custom CSV dataset (multi-label with 7 defect types)
- Text preprocessing via TF-IDF
- Models Used:
  - Logistic Regression (One-vs-Rest)
  - Multi-label SVM
  - Perceptron (online learning)
  - DNN

### ✅ Part 3: Streamlit Web App
- Upload audio → predict Bonafide or Deepfake
- Enter software description → predict multiple defect labels
- Model selector for both tasks
- Shows confidence scores
![audio_Detection ](https://github.com/user-attachments/assets/708deb55-1b71-4c48-97f8-b3204e42a124)
![software_prediction](https://github.com/user-attachments/assets/39dc9aa3-2571-402f-b4ff-03973d2d2734)


---


## 🛠️ Installation

```bash
git clone https://github.com/yourusername/deepfake-defect-detector.git
cd deepfake-defect-detector
pip install -r requirements.txt
streamlit run streamlit_app.py



