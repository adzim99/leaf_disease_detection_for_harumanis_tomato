import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import streamlit as st

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Load the trained models
MODEL_PATH_1 = "tomato_model.keras"
MODEL_PATH_2 = "harumanis_model.keras"

tomato_model = load_model(MODEL_PATH_1)
harumanis_model = load_model(MODEL_PATH_2)

# Streamlit Web Application
st.title("Dual Leaf Disease Detection System")
st.write("Upload an image to detect diseases using either the Tomato Model or Harumanis Model.")

# Dropdown for model selection
model_option = st.selectbox(
    "Select a model to use:",
    ("Tomato Model", "Harumanis Model")
)

# File uploader
uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    st.write("Processing...")

    # Get the target size based on the selected model's input shape
    if model_option == "Tomato Model":
        target_size = tomato_model.input_shape[1:3]  # (150, 150) expected
        model = tomato_model
        class_indices = {
            0: "Bacterial Spot",
            1: "Early Blight",
            2: "Late Blight",
            3: "Leaf Mold",
            4: "Septoria Leaf Spot",
            5: "Spider Mite",
            6: "Target Spot",
            7: "Yellow Leaf Curl Virus",
            8: "Mosaic Virus",
            9: "Healthy"
        }
    else:
        target_size = harumanis_model.input_shape[1:3]  # (224, 224) expected
        model = harumanis_model
        class_indices = {
            0: "Anthracnose",
            1: "Black Sooty Mold",
            2: "Healthy"
        }

    # Preprocess the image
    image = load_img(uploaded_file, target_size=target_size)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array /= 255.0  # Normalize

    # Make predictions
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Display prediction and confidence
    confidence = np.max(predictions) * 100
    prediction_label = class_indices.get(predicted_class, 'Unknown')

    st.write(f"**Prediction:** {prediction_label}")
    st.write(f"**Confidence:** {confidence:.2f}%")
else:
    st.write("Please upload an image to classify.")
