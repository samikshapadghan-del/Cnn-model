import os
import json
import zipfile
import subprocess
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Coffee Mug Classifier", page_icon="☕")

# ---------------- SETTINGS ----------------
IMAGE_SIZE = (150, 150)
MODEL_DIR = "model_folder"
KAGGLE_MODEL = "samikshapadghan/cnn/tensorFlow2/default/1"

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_my_model():
    os.makedirs("/home/appuser/.kaggle", exist_ok=True)

    # Kaggle API credentials from Streamlit Secrets
    kaggle_json = {
        "username": st.secrets["KAGGLE_USERNAME"],
        "key": st.secrets["KAGGLE_KEY"]
    }

    with open("/home/appuser/.kaggle/kaggle.json", "w") as f:
        json.dump(kaggle_json, f)

    os.chmod("/home/appuser/.kaggle/kaggle.json", 0o600)

    # Download model if not already downloaded
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)

        result = subprocess.run(
            [
                "kaggle",
                "models",
                "instances",
                "versions",
                "download",
                KAGGLE_MODEL,
                "-p",
                MODEL_DIR
            ],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            st.error("Kaggle model download failed")
            st.code(result.stderr)
            st.stop()

        # Extract zip files
        for file in os.listdir(MODEL_DIR):
            if file.endswith(".zip"):
                zip_path = os.path.join(MODEL_DIR, file)
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(MODEL_DIR)

    # Find model file automatically
    model_path = None

    for root, dirs, files in os.walk(MODEL_DIR):
        for file in files:
            if file.endswith(".keras") or file.endswith(".h5"):
                model_path = os.path.join(root, file)
                break

    if model_path is None:
        st.error("No model file (.keras or .h5) found")
        for root, dirs, files in os.walk(MODEL_DIR):
            st.write(root, files)
        st.stop()

    st.success(f"Loading model: {model_path}")

    model = tf.keras.models.load_model(model_path)
    return model

# Load model
model = load_my_model()

# ---------------- UI ----------------
st.title("☕ Coffee Mug vs Tea Cup Classifier")
st.write("Upload an image and get prediction.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Preprocess
    img = image.resize(IMAGE_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
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
        st.success("It's a Coffee Mug ☕")
    else:
        st.info("It's a Tea Cup 🍵")
