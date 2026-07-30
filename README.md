# 😊 DeepFER: Facial Emotion Recognition Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-CNN-red?logo=keras)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Application-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Project Overview

DeepFER (Deep Facial Emotion Recognition) is a deep learning-based computer vision system that recognizes human emotions from facial expressions. The project uses **Convolutional Neural Networks (CNNs)** with **Transfer Learning** to classify facial images into seven different emotional categories.

The application can predict emotions from uploaded images and can also be deployed as an interactive Streamlit web application for real-time inference.

---

## 🎯 Objectives

- Develop an accurate facial emotion recognition model.
- Classify facial expressions into seven emotion classes.
- Compare deep learning techniques for emotion classification.
- Deploy the trained model using Streamlit.
- Build a user-friendly AI application for real-time emotion prediction.

---

## 😀 Emotion Classes

The model predicts the following emotions:

- Angry 😠
- Disgust 🤢
- Fear 😨
- Happy 😄
- Neutral 😐
- Sad 😢
- Surprise 😲

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| TensorFlow | Deep Learning |
| Keras | CNN Model |
| OpenCV | Image Processing |
| NumPy | Numerical Computing |
| Pandas | Data Handling |
| Matplotlib | Visualization |
| Streamlit | Web Application |
| Scikit-learn | Data Splitting & Evaluation |

---

## 📂 Project Structure

```text
DeepFER-Facial-Emotion-Recognition/
│
├── app.py
├── train.py
├── predict.py
├── requirements.txt
├── README.md
├── LICENSE
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── inference.py
│   └── utils.py
│
├── models/
│   └── emotion_model.keras
│
├── notebooks/
│   └── DeepFER.ipynb
│
├── dataset/
│
├── outputs/
│
└── images/
```

---

## 📊 Dataset

The project uses a facial expression dataset containing thousands of labeled facial images representing seven human emotions.

The dataset undergoes:

- Image preprocessing
- Face normalization
- Data augmentation
- Training-validation split

---

## 🧠 Deep Learning Architecture

The model consists of:

- Image Input Layer
- Data Augmentation
- Transfer Learning Backbone
- Global Average Pooling
- Dense Layers
- Dropout
- Softmax Output Layer

---

## 🚀 Features

- Facial Emotion Recognition
- Image Upload Prediction
- Deep Learning Classification
- Transfer Learning
- Streamlit Interface
- Easy Deployment
- Real-Time Prediction

---

## 📈 Model Workflow

```text
Input Image
      │
      ▼
Image Preprocessing
      │
      ▼
CNN + Transfer Learning
      │
      ▼
Feature Extraction
      │
      ▼
Emotion Classification
      │
      ▼
Prediction Result
```

---

## 💻 Installation

Clone the repository

```bash
git clone https://github.com/maha2806/DeepFER-Facial-Emotion-Recognition.git
```

Move into the project directory

```bash
cd DeepFER-Facial-Emotion-Recognition
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

## 📊 Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 📷 Screenshots

### Home Page

(Add Screenshot)

---

### Prediction Page

(Add Screenshot)

---

### Emotion Detection

(Add Screenshot)

---

## 🔮 Future Enhancements

- Live Webcam Detection
- Video Emotion Recognition
- Emotion Analytics Dashboard
- Mobile Application
- REST API Deployment
- Multi-face Detection
- Cloud Deployment

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Push the branch
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License.

---

## 👩‍💻 Author

**Sita Bharatula**

MCA (Artificial Intelligence & Machine Learning)

Chandigarh University

GitHub: https://github.com/maha2806

---

⭐ If you found this project useful, consider giving it a Star.
