import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import gdown
import os

# Constants
IMAGE_SIZE = 64
MODEL_PATH = "model.h5"

# Google Drive Model Link
MODEL_ID = "1-QNakGepRJEW4EiIKmaC6TKy_FEwCb2k"  # 👈 your model's file ID
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"

@st.cache_resource
def load_cnn_model():
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading model from Google Drive...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = None
if st.button("🔄 Load Model"):
    model = load_cnn_model()
    st.success("✅ Model Loaded")

st.title("🧠 Brain Tumor Detection from MRI")
st.markdown("Upload a brain MRI image (JPG/PNG) to detect if it shows signs of a tumor.")

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
