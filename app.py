import os
import json
import zipfile
import subprocess
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

st.set_page_config(page_title="Coffee Mug Classifier", page_icon="☕")

IMAGE_SIZE = (150, 150)
MODEL_DIR = "model_folder"
ZIP_FILE = "model.zip"

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
        os.makedirs(MODEL_DIR, exist_ok=True)

        result = subprocess.run(
            [
                "kaggle",
                "models",
                "instances",
                "versions",
                "download",
                "samikshapadghan/cnn/tensorFlow2/default/1",
                "-p",
                MODEL_DIR
            ],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            st.error(result.stderr)
            st.stop()

        # unzip downloaded zip file
        for file in os.listdir(MODEL_DIR):
            if file.endswith(".zip"):
                zip_path = os.path.join(MODEL_DIR, file)
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(MODEL_DIR)

    st.write("Files:", os.listdir(MODEL_DIR))

    model = tf.keras.models.load_model(MODEL_DIR)
    return model

model = load_my_model()

st.title("☕ Coffee Mug vs Tea Cup Classifier")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image)

    img = image.resize(IMAGE_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    pred = model.predict(img_array)

    if pred[0][0] > 0.5:
        st.success("Tea Cup 🍵")
    else:
        st.success("Coffee Mug ☕")
