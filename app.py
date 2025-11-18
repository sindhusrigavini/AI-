import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="AI Blur Detector", page_icon="📷", layout="centered")

st.title("📷 AI Image Blur Detector")
st.write("Upload an image, and the app will detect whether it is **Sharp** or **Blurry**.")

# Function to compute blur using Laplacian variance
def detect_blur(pil_img):
    gray = np.array(pil_img.convert("L"))
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Default threshold slider
threshold = st.slider(
    "Blur Threshold (lower = blurry, higher = sharp)",
    min_value=10,
    max_value=500,
    value=100
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    lap_value = detect_blur(image)

    st.subheader("🔍 Blur Analysis")
    st.write(f"**Laplacian Variance:** `{lap_value:.2f}`")

    # Blur confidence score between 0–1
    blur_score = min(max(lap_value / 500, 0.0), 1.0)
    st.progress(blur_score)

    # Classification
    if lap_value < threshold:
        st.error(f"Result: **BLURRY IMAGE** ❌ (Variance: {lap_value:.2f})")
    else:
        st.success(f"Result: **SHARP IMAGE** ✅ (Variance: {lap_value:.2f})")
