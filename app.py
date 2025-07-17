# B2R
import streamlit as st
import numpy as np
import cv2
import os
import gdown
from tensorflow.keras.models import load_model
from PIL import Image

# Constants
IMAGE_SIZE = 64
MODEL_PATH = "model.h5"
DRIVE_LINK = "https://drive.google.com/uc?id=1-QNakGepRJEW4EiIKmaC6TKy_FEwCb2k"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Downloading model..."):
        gdown.download(DRIVE_LINK, MODEL_PATH, quiet=False)
        st.success("✅ Model downloaded!")

# Load trained model
@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

model = None
if st.button("🔄 Load Model"):
    if not model:
        model = load_cnn_model()
        st.success("✅ Model Loaded")

# Streamlit UI
st.title("🧠 Brain Tumor Detection from MRI")
st.markdown("Upload a brain MRI image (JPG/PNG) to detect if it shows signs of a tumor.")

# Image uploader
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if not model:
        model = load_cnn_model()

    pred = np.argmax(model.predict(img_array))
    result = "🧠 Tumor Detected" if pred == 1 else "✅ No Tumor Detected"
    st.subheader("🔍 Prediction Result:")
    st.success(result)
