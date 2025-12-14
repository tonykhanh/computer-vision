import streamlit as st
import cv2
import numpy as np
from PIL import Image
from core import ObjectDetector

# Page Config
st.set_page_config(
    page_title="VisionAI - YOLO World",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Cyberpunk/Glassmorphism look
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(0, 0, 0) 0%, rgb(30, 30, 30) 90.2%);
        color: #fff;
    }
    
    /* Header Styling */
    h1 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        background: linear-gradient(to right, #00dbde, #fc00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding-bottom: 20px;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Camera Input Styling */
    .stCamera {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Processed Image Container */
    .processed-image {
        border-radius: 15px;
        border: 2px solid #00dbde;
        box-shadow: 0 0 20px rgba(0, 219, 222, 0.3);
    }
    
    /* Info Box */
    .info-box {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #fc00ff;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/artificial-intelligence.png", width=80)
    st.title("Settings")
    
    st.markdown("### Model Config")
    conf_threshold = st.slider("Confidence Threshold", 0.01, 1.0, 0.15, 0.05)
    iou_threshold = st.slider("IoU Threshold (NMS)", 0.01, 1.0, 0.45, 0.05)
    
    st.write("---")
    st.markdown("### Info")
    st.info("Utilizing **YOLO-World (v8s)** for open-vocabulary object detection.")

# Main Layout
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.title("VisionAI: Object Detection")
    st.markdown("<div style='text-align: center; color: #aaa; margin-bottom: 30px;'>Powered by Ultralytics YOLO-World & Streamlit</div>", unsafe_allow_html=True)

# Initialize Detector
@st.cache_resource
def load_detector():
    return ObjectDetector(model_path="yolov8s-world.pt")

detector = load_detector()

if not detector or detector.model is None:
    st.error("Model Failed to Load.")
    if detector and detector.error_msg:
         st.error(f"Detailed Error: {detector.error_msg}")
    st.warning("Please check the logs or ensure the model file can be downloaded.")
    # Do not stop completely, let the UI render so they can see the error


# Camera Input
st.markdown("### 📸 Capture Image")
img_file_buffer = st.camera_input("Take a photo", key="camera")

if img_file_buffer is not None:
    # Read Image
    file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    
    # Process
    with st.spinner("AI Processing..."):
        processed_frame = detector.process_frame(frame, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
        
        # Display
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        st.markdown("### 🔍 Analysis Result")
        st.image(rgb_frame, caption="Detected Objects", use_container_width=True, output_format="PNG")
        
        # Optional: Show detection stats if possible (would need `core.py` data return change, keeping simple for now)
        
# Footer
st.markdown("""
<div style='position: fixed; bottom: 20px; right: 20px; color: #555; font-size: 12px;'>
    v2.0.0 | AI-Powered
</div>
""", unsafe_allow_html=True)
