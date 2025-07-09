import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Helmet Detection System",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

def show_tensorflow_warning():
    """Show warning about TensorFlow compatibility"""
    st.markdown("""
    <div class="warning-box">
        <h3>⚠️ TensorFlow Compatibility Issue</h3>
        <p>This deployment is currently experiencing TensorFlow compatibility issues with Python 3.13.</p>
        <p><strong>Status:</strong> Working on a solution using alternative ML frameworks.</p>
        <p><strong>Current Features:</strong> Image upload and display functionality available.</p>
    </div>
    """, unsafe_allow_html=True)

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for display"""
    # Convert PIL image to numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Resize image
    image = cv2.resize(image, target_size)
    
    return image

def handle_image_upload():
    """Handle image upload and display"""
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image to analyze",
        key="image_uploader"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.subheader("🔍 Image Analysis")
            
            # Show image info
            st.write(f"**Image Size:** {image.size}")
            st.write(f"**Image Mode:** {image.mode}")
            st.write(f"**File Size:** {uploaded_file.size} bytes")
            
            # Show preprocessed image
            processed_image = preprocess_image(image)
            st.image(processed_image, caption="Preprocessed Image (224x224)", use_column_width=True)
            
            st.info("🔄 Model prediction temporarily unavailable due to TensorFlow compatibility issues.")

def handle_camera_capture():
    """Handle camera capture"""
    st.subheader("📸 Camera Capture")
    st.info("Camera functionality will be available once TensorFlow compatibility is resolved.")
    
    # Placeholder for camera functionality
    if st.button("📷 Open Camera (Coming Soon)"):
        st.warning("Camera capture feature is being updated for Python 3.13 compatibility.")

def show_safety_alert():
    """Show safety information"""
    st.subheader("🛡️ Safety Information")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### ✅ With Helmet
        - **Safety Compliance:** Proper PPE usage
        - **Risk Level:** Low
        - **Action:** Continue safe practices
        """)
    
    with col2:
        st.markdown("""
        ### ⚠️ Without Helmet
        - **Safety Violation:** Missing PPE
        - **Risk Level:** High
        - **Action:** Immediate correction required
        """)

def main():
    # Header
    st.markdown('<h1 class="main-header">🪖 Helmet Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered safety compliance monitoring</p>', unsafe_allow_html=True)
    
    # Show TensorFlow warning
    show_tensorflow_warning()
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📷 Image Upload", "📸 Camera", "ℹ️ About", "🛡️ Safety"])
    
    with tab1:
        st.header("📷 Image Upload")
        handle_image_upload()
    
    with tab2:
        st.header("📸 Camera Capture")
        handle_camera_capture()
    
    with tab3:
        st.header("ℹ️ About This System")
        
        st.markdown("""
        ### 🎯 Purpose
        This system uses artificial intelligence to detect whether individuals in images are wearing safety helmets.
        
        ### 🔧 Technical Details
        - **Framework:** TensorFlow (currently being updated for Python 3.13)
        - **Model:** Convolutional Neural Network (CNN)
        - **Input:** 224x224 RGB images
        - **Output:** Binary classification (With/Without Helmet)
        
        ### 🚧 Current Status
        - ✅ UI and image processing working
        - 🔄 ML model compatibility being resolved
        - 📱 Mobile-friendly interface
        """)
    
    with tab4:
        st.header("🛡️ Safety Guidelines")
        show_safety_alert()
        
        st.markdown("""
        ### 📋 Safety Standards
        - Always wear appropriate PPE in construction zones
        - Regular safety training and compliance checks
        - Report safety violations immediately
        - Maintain safety equipment in good condition
        """)

if __name__ == "__main__":
    main() 