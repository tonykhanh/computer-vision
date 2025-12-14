import streamlit as st
import cv2
import numpy as np
from PIL import Image
from core import ObjectDetector
import config

# Page Config
st.set_page_config(
    page_title="AI Fruit Vision",
    page_icon="🍎",
    layout="centered"
)

# Custom CSS implementation via markdown
st.markdown("""
<style>
    .main {
        background: linear-gradient(to right, #24243e, #302b63, #0f0c29);
        color: white;
    }
    h1 {
        text-align: center;
        background: linear-gradient(to right, #fff, #a5b4fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stCamera {
        border-radius: 20px;
        overflow: hidden;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Title and Description
st.title("AI Fruit Vision")
st.write("Real-time object detection powered by MobileNetV2")

# Initialize Model (Cached to avoid reloading)
@st.cache_resource
def load_detector():
    try:
        return ObjectDetector()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

detector = load_detector()

if not detector:
    st.warning("Model could not be loaded. Please check if model files exist.")

# Camera Input
img_file_buffer = st.camera_input("Capture an image")

if img_file_buffer is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    if detector:
        # Process frame
        # process_frame takes BGR image (OpenCV default), draws on it, returns BGR
        processed_frame = detector.process_frame(frame)
        
        # Convert BGR to RGB for Streamlit display
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        st.image(rgb_frame, caption="Processed Image", use_container_width=True)
    else:
        # Fallback display if model failed
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(rgb_frame, caption="Original Image (Model missing)", use_container_width=True)
