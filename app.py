import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# ── Config ────────────────────────────────────────────────────────────────
IMG_SIZE = (48, 48)
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
DEFAULT_MODEL_PATH = "models/DeepFER_Best_Model.keras"


@st.cache_resource
def get_model(model_path: str):
    return load_model(model_path)


def preprocess(img: Image.Image):
    img = img.convert("L").resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))
    return arr, img


def main():
    st.set_page_config(page_title="DeepFER - Facial Emotion Recognition", page_icon="🙂")
    st.title("🙂 DeepFER: Facial Emotion Recognition")
    st.write(
        "Upload a face image (ideally cropped and front-facing) and the CNN model "
        "will predict the emotion expressed."
    )

    model_path = st.sidebar.text_input("Model path", value=DEFAULT_MODEL_PATH)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}. Train a model first.")
            return

        model = get_model(model_path)
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        arr, processed_img = preprocess(image)
        predictions = model.predict(arr)[0]
        predicted_idx = int(np.argmax(predictions))
        predicted_label = CLASS_NAMES[predicted_idx]
        confidence = predictions[predicted_idx] * 100

        with col2:
            st.image(processed_img, caption="Preprocessed (48x48 grayscale)", use_container_width=True)

        st.subheader(f"Predicted Emotion: **{predicted_label.upper()}** ({confidence:.1f}%)")

        scores_df = pd.DataFrame(
            {"Emotion": CLASS_NAMES, "Confidence (%)": predictions * 100}
        ).sort_values("Confidence (%)", ascending=False)

        st.bar_chart(scores_df.set_index("Emotion"))


if __name__ == "__main__":
    main()
