import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="DeepFER", page_icon="😊")

st.title("😊 DeepFER - Facial Emotion Recognition")

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

model_name = st.selectbox(
    "Select Model",
    [
        "CNN_Model.keras",
        "Improved_CNN_Model.keras",
        "Final_Optimized_CNN_Model.keras"
    ]
)

@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

model = load_model(model_name)

uploaded_file = st.file_uploader(
    "Upload a Face Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", width=250)

    image = image.resize((48, 48))

    img = np.array(image)

    img = img.astype("float32") / 255.0

    img = img.reshape(1, 48, 48, 1)

    prediction = model.predict(img)

    predicted_class = np.argmax(prediction)

    confidence = prediction[0][predicted_class] * 100

    st.success(f"Predicted Emotion: {emotion_labels[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}%")

    st.subheader("Prediction Scores")

    for i in range(len(emotion_labels)):
        st.write(f"{emotion_labels[i]} : {prediction[0][i]*100:.2f}%")
