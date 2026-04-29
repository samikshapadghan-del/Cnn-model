
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import os

# --- Configuration ---
MODEL_PATH = 'https://www.kaggle.com/models/samikshapadghan/cnn//tensorFlow2/default'
IMAGE_SIZE = (150, 150) # Must match the size used during training

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# --- Streamlit App ---f
st.title("Coffee Mug vs. Tea Cup Classifier")
st.write("Upload an image to classify if it's a coffee mug or a tea cup.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Preprocess the image
    img = image.resize(IMAGE_SIZE) # Resize the PIL Image
    img_array = img_to_array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = img_array / 255.0 # Rescale pixels

    # Make prediction
    prediction = model.predict(img_array)

    # Class labels (ensure these match your training labels)
    # Assuming 'coffee mug' was 0 and 'tea cup' was 1 based on previous output
    class_names = ['coffee mug', 'tea cup']

    # Determine predicted class and confidence
    if prediction[0][0] > 0.5:
        predicted_class_name = class_names[1] # tea cup
        confidence = prediction[0][0]
    else:
        predicted_class_name = class_names[0] # coffee mug
        confidence = 1 - prediction[0][0]

    st.subheader(f"Prediction: {predicted_class_name}")
    st.write(f"Confidence: {confidence*100:.2f}%")

    if predicted_class_name == 'coffee mug':
        st.success("It's a Coffee Mug!")
    else:
        st.info("It's a Tea Cup!")
