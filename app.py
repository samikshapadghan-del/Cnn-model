import os
import json
import subprocess
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

st.set_page_config(page_title="Coffee Mug Classifier", page_icon="☕")

IMAGE_SIZE = (150, 150)
MODEL_DIR = "model_folder"

@st.cache_resource
def load_my_model():
    os.makedirs("/home/appuser/.kaggle", exist_ok=True)

    kaggle_json = {
        "username": st.secrets["KAGGLE_USERNAME"],
        "key": st.secrets["KAGGLE_KEY"]
    }

    with open("/home/appuser/.kaggle/kaggle.json", "w") as f:
        json.dump(kaggle_json, f)

    os.chmod("/home/appuser/.kaggle/kaggle.json", 0o600)

    if not os.path.exists(MODEL_DIR):
        subprocess.run(
            [
                "kaggle",
                "models",
                "instances",
                "versions",
                "download",
                "samikshapadghan/cnn/tensorFlow2/default/1",
                "-p",
                MODEL_DIR,
                "--unzip"
            ],
            check=True
        )

    model = tf.keras.models.load_model(MODEL_DIR)
    return model

model = load_my_model()

st.title("Coffee Mug vs Tea Cup Classifier")
st.write("Upload an image to classify it.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize(IMAGE_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    class_names = ["coffee mug", "tea cup"]

    if prediction[0][0] > 0.5:
        predicted_class = class_names[1]
        confidence = prediction[0][0]
    else:
        predicted_class = class_names[0]
        confidence = 1 - prediction[0][0]

    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence * 100:.2f}%")

    if predicted_class == "coffee mug":
        st.success("It's a Coffee Mug!")
    else:
        st.info("It's a Tea Cup!")
