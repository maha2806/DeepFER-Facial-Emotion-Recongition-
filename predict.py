import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "models/emotion_model.keras"

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

model = tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")
    image = image.resize((48, 48))

    image = np.array(image).astype("float32") / 255.0

    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)

    return image


def predict_emotion(image_path):
    image = preprocess_image(image_path)

    prediction = model.predict(image, verbose=0)

    idx = np.argmax(prediction)

    return emotion_labels[idx], prediction[0]


if __name__ == "__main__":

    image_path = input("Enter image path: ")

    emotion, confidence = predict_emotion(image_path)

    print(f"Predicted Emotion: {emotion}")

    print(confidence)
