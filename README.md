# 🚦 Traffic Sign Recognition (GTSRB)

**Neural Networks Course Project**

This project implements a Convolutional Neural Network (CNN) to classify German traffic signs using the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset.

---

## 📂 Dataset Source

The dataset was downloaded from **Kaggle**:

🔗 **GTSRB – German Traffic Sign Recognition Benchmark**
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/

Dataset contents:

- 43 traffic sign classes
- Train, Test, and Meta folders
- Thousands of labeled images

---

## 📌 Project Objectives

- Load and preprocess the GTSRB dataset
- Build and train a CNN model
- Evaluate model performance
- Develop a Streamlit-based web interface
- Run the model both locally and online (Streamlit Cloud)

---

## 🧠 Technologies Used

- **Python**
- **TensorFlow / Keras**
- **NumPy, Pandas**
- **OpenCV**
- **Streamlit**
- **Google Colab**

---

## 🏋️ Model Overview

- Model Type: **Convolutional Neural Network (CNN)**
- Layers: `Conv2D`, `MaxPooling2D`, `Dense`, `Dropout`
- Handles classification of **43 classes**
- Data Augmentation applied
- Trained model stored as: `MODEL_GERMAN_TRAFFIC_SIGN.h5`

---

## 📊 Model Results

- Test Accuracy: **XX%** *(replace with your result)*
- The model predicts the correct traffic sign label when an image is uploaded

---

## 🌐 Streamlit Web App

### 🔧 Run Locally

Install dependencies:

```bash
pip install streamlit tensorflow pillow numpy
```

Launch the app:

```bash
streamlit run app.py
```

Then open:

👉 http://localhost:8501

---

### ☁️ Deploy Online (Streamlit Cloud)

1. Upload repository to GitHub
2. Go to https://streamlit.io/cloud
3. Click **New app**
4. Select your GitHub repository
5. Choose `app.py` and deploy
6. You will receive an online public URL

---

## 📁 Project Structure

```
├── German Traffic Sign Recognition Benchmark.ipynb   # Training notebook
├── app.py                           # Streamlit web app
├── MODEL_GERMAN_TRAFFIC_SIGN.h5     # Trained model
├── README.md                        # Project documentation
```

---

## 👨‍💻 Author

Sirojiddin Ismoilov
Neural Networks Course Project
