import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="DeepFER",
    page_icon="😊",
    layout="wide"
)

# -------------------------------------------------
# Custom CSS
# -------------------------------------------------

st.markdown("""
<style>

.main{
    background-color:#f7f7f7;
}

h1,h2,h3{
    color:#0E1117;
}

.stButton>button{
    width:100%;
    border-radius:10px;
    height:3em;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Load Model
# -------------------------------------------------

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/emotion_model.keras")
    return model

model = load_model()

# -------------------------------------------------
# Emotion Labels
# -------------------------------------------------

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# -------------------------------------------------
# Sidebar
# -------------------------------------------------

st.sidebar.title("😊 DeepFER")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📷 Predict Emotion",
        "🧠 About"
    ]
)

# -------------------------------------------------
# HOME
# -------------------------------------------------

if page == "🏠 Home":

    st.title("😊 DeepFER")

    st.subheader("Facial Emotion Recognition Using Deep Learning")

    st.markdown("---")

    st.write("""
DeepFER is a Deep Learning based facial emotion recognition system.

The model classifies a face into one of the following emotions:

- 😠 Angry
- 🤢 Disgust
- 😨 Fear
- 😄 Happy
- 😐 Neutral
- 😢 Sad
- 😲 Surprise
""")

    st.info("👈 Select **Predict Emotion** from the sidebar to begin.")

# -------------------------------------------------
# Prediction
# -------------------------------------------------

elif page == "📷 Predict Emotion":

    st.title("📷 Emotion Detection")

    uploaded_file = st.file_uploader(
        "Upload a Face Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("L")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        # -----------------------------
        # Preprocess
        # -----------------------------

        img = image.resize((48,48))

        img = np.array(img)

        img = img.astype("float32")/255.0

        img = np.expand_dims(img, axis=-1)

        img = np.expand_dims(img, axis=0)

        # -----------------------------
        # Prediction
        # -----------------------------

        prediction = model.predict(img, verbose=0)

        predicted_class = np.argmax(prediction)

        predicted_emotion = emotion_labels[predicted_class]

        confidence = prediction[0]

        with col2:

            st.success(f"### 😊 Prediction")

            st.metric(
                "Detected Emotion",
                predicted_emotion
            )

            st.metric(
                "Confidence",
                f"{confidence[predicted_class]*100:.2f}%"
            )

        st.markdown("---")

        st.subheader("Prediction Probabilities")

        fig, ax = plt.subplots(figsize=(9,4))

        bars = ax.bar(
            emotion_labels,
            confidence*100
        )

        ax.set_ylabel("Probability (%)")
        ax.set_ylim(0,100)

        for bar in bars:

            y = bar.get_height()

            ax.text(
                bar.get_x()+bar.get_width()/2,
                y+1,
                f"{y:.1f}",
                ha='center'
            )

        st.pyplot(fig)

# -------------------------------------------------
# About
# -------------------------------------------------

elif page == "🧠 About":

    st.title("🧠 About")

    st.write("""
### Project Name

DeepFER: Facial Emotion Recognition Using Deep Learning

---

### Technologies

- Python
- TensorFlow
- Keras
- NumPy
- OpenCV
- Streamlit

---

### Model

Deep Learning CNN trained on grayscale facial images (48×48).

---

### Emotion Classes

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

---

### Developed By

**Sita Bharatula**

MCA (Artificial Intelligence & Machine Learning)

Chandigarh University
""")
