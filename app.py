import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from model_loader import SavedModelPredictor

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
    .result-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .with-helmet {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .without-helmet {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
    .confidence-bar {
        background-color: #e9ecef;
        border-radius: 10px;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained TensorFlow model"""
    try:
        predictor = SavedModelPredictor("model.savedmodel")
        if predictor.model is not None:
            st.success("✅ Model loaded successfully!")
            return predictor
        else:
            st.error("❌ Failed to load model")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model input"""
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
    
    # Normalize pixel values (using Teachable Machine normalization)
    image = image.astype(np.float32)
    image = (image / 127.5) - 1
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

def predict_helmet(predictor, image):
    """Predict helmet presence using the model"""
    try:
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction using the predictor
        prediction = predictor.predict(processed_image)
        
        if prediction is None:
            st.error("❌ Prediction failed")
            return None, None, None
        
        # Ensure prediction is numpy array
        if hasattr(prediction, 'numpy'):
            prediction = prediction.numpy()
        
        # Get class and confidence
        class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][class_idx])
        
        # Map class index to label
        labels = ["With Helmet", "Without Helmet"]
        predicted_class = labels[class_idx]
        
        return predicted_class, confidence, prediction[0]
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

def display_results(predicted_class, confidence, all_probabilities):
    """Display prediction results with styling"""
    labels = ["With Helmet", "Without Helmet"]
    
    # Convert all tensor values to Python floats
    confidence = float(confidence)
    all_probabilities = [float(prob) for prob in all_probabilities]
    
    # Create columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎯 Detection Result")
        
        # Display result with appropriate styling
        if predicted_class == "With Helmet":
            st.markdown(f"""
            <div class="result-box with-helmet">
                <h3>✅ {predicted_class}</h3>
                <p>Safety compliance detected!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box without-helmet">
                <h3>⚠️ {predicted_class}</h3>
                <p>Safety violation detected!</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📊 Confidence Score")
        
        # Display confidence percentage
        confidence_percent = confidence * 100
        st.metric("Confidence", f"{confidence_percent:.1f}%")
        
        # Progress bar for confidence
        st.progress(confidence)
        
        # Display all class probabilities
        st.subheader("📈 Class Probabilities")
        for i, (label, prob) in enumerate(zip(labels, all_probabilities)):
            prob_percent = prob * 100
            st.write(f"{label}: {prob_percent:.1f}%")
            
            # Color-coded progress bar
            if i == 0:  # With Helmet
                st.progress(prob, text=f"🪖 {prob_percent:.1f}%")
            else:  # Without Helmet
                st.progress(prob, text=f"🚫 {prob_percent:.1f}%")

def handle_image_upload(predictor):
    """Handle image upload and processing"""
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image to detect helmet presence",
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
            st.subheader("🔍 Analysis")
            
            # Add a button to trigger prediction
            if st.button("🚀 Detect Helmet", type="primary", key="detect_upload"):
                with st.spinner("Analyzing image..."):
                    # Make prediction
                    predicted_class, confidence, all_probabilities = predict_helmet(predictor, image)
                    
                    if predicted_class is not None:
                        # Display results
                        st.success("✅ Analysis complete!")
                        
                        # Show results in main area
                        st.markdown("---")
                        display_results(predicted_class, confidence, all_probabilities)
                        
                        # Show safety alert if no helmet detected
                        if predicted_class == "Without Helmet":
                            show_safety_alert()
                        
                        # Additional safety recommendations
                        st.markdown("---")
                        st.subheader("💡 Safety Recommendations")
                        
                        if predicted_class == "With Helmet":
                            st.success("""
                            ✅ **Good Safety Practice Detected!**
                            - Continue following safety protocols
                            - Ensure helmet is properly fitted
                            - Check for any damage to PPE
                            - Set a good example for others
                            """)
                        else:
                            st.warning("""
                            ⚠️ **Safety Violation Detected!**
                            - Immediately put on appropriate helmet
                            - Check helmet for proper fit
                            - Ensure helmet meets safety standards
                            - Report to supervisor if PPE is damaged
                            """)
                    else:
                        st.error("❌ Failed to analyze image. Please try again.")
    
    else:
        # Instructions when no image is uploaded
        st.info("👆 Please upload an image to start helmet detection analysis.")

def handle_camera_capture(predictor):
    """Handle camera capture and processing"""
    st.info("📷 Use your camera to capture an image for real-time helmet detection")
    
    # Camera input
    camera_photo = st.camera_input("Take a photo for helmet detection", key="camera_capture")
    
    if camera_photo is not None:
        # Display captured image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Captured Image")
            image = Image.open(camera_photo)
            st.image(image, caption="Captured Image", use_column_width=True)
        
        with col2:
            st.subheader("🔍 Analysis")
            
            # Add a button to trigger prediction
            if st.button("🚀 Detect Helmet", type="primary", key="detect_camera"):
                with st.spinner("Analyzing captured image..."):
                    # Make prediction
                    predicted_class, confidence, all_probabilities = predict_helmet(predictor, image)
                    
                    if predicted_class is not None:
                        # Display results
                        st.success("✅ Analysis complete!")
                        
                        # Show results in main area
                        st.markdown("---")
                        display_results(predicted_class, confidence, all_probabilities)
                        
                        # Show safety alert if no helmet detected
                        if predicted_class == "Without Helmet":
                            show_safety_alert()
                        
                        # Additional safety recommendations
                        st.markdown("---")
                        st.subheader("💡 Safety Recommendations")
                        
                        if predicted_class == "With Helmet":
                            st.success("""
                            ✅ **Good Safety Practice Detected!**
                            - Continue following safety protocols
                            - Ensure helmet is properly fitted
                            - Check for any damage to PPE
                            - Set a good example for others
                            """)
                        else:
                            st.warning("""
                            ⚠️ **Safety Violation Detected!**
                            - Immediately put on appropriate helmet
                            - Check helmet for proper fit
                            - Ensure helmet meets safety standards
                            - Report to supervisor if PPE is damaged
                            """)
                    else:
                        st.error("❌ Failed to analyze image. Please try again.")

def show_safety_alert():
    """Display safety alert for helmet violation"""
    st.markdown("""
    <div style="background-color: #ffebee; border: 2px solid #f44336; border-radius: 10px; padding: 1.5rem; margin: 1rem 0;">
        <h3 style="color: #d32f2f; margin: 0; font-size: 1.3rem;">🚨 SAFETY ALERT - MACHINE STOPPED</h3>
        <p style="color: #d32f2f; font-weight: bold; margin: 0.8rem 0; font-size: 1.1rem;">
            ⚠️ HELMET NOT DETECTED - SAFETY VIOLATION
        </p>
        <p style="margin: 0.8rem 0; color: #2c3e50; line-height: 1.5;">
            <strong style="color: #d32f2f;">Action Taken:</strong> All manufacturing machines have been automatically stopped due to safety violation.
        </p>
        <p style="margin: 0.8rem 0; color: #2c3e50; line-height: 1.5;">
            <strong style="color: #d32f2f;">Required Action:</strong> Please put on appropriate safety helmet before resuming operations.
        </p>
        <p style="margin: 0.8rem 0; color: #2c3e50; line-height: 1.5;">
            <strong style="color: #d32f2f;">Contact:</strong> Notify supervisor immediately for safety clearance.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # JavaScript alert (if possible)
    st.markdown("""
    <script>
        alert("🚨 SAFETY ALERT: Helmet not detected! Machines have been stopped. Please put on safety helmet immediately.");
    </script>
    """, unsafe_allow_html=True)

def main():
    # Header - Moved up for better screenshot capture
    st.markdown('<h1 class="main-header">🪖 Helmet Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered PPE Detection for Manufacturing Safety</p>', unsafe_allow_html=True)
    
    # Project information from problem statement
    st.markdown("---")
    st.markdown("""
    <div style="background-color: #e3f2fd; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid #2196f3;">
        <h3 style="color: #1565c0; margin-bottom: 1rem;">📋 Project Overview</h3>
        <p style="color: #2c3e50; margin-bottom: 0.8rem;"><strong style="color: #1565c0;">Project:</strong> AI based accident prevention in MMS (modular manufacturing system)</p>
        <p style="color: #2c3e50; line-height: 1.6;"><strong style="color: #1565c0;">Description:</strong> Accidents in MMS often occur due to delayed human response or oversight. 
        By implementing an AI-powered camera vision system, activities can be continuously monitored and machines 
        can be auto-stopped in real-time upon detecting potential hazards. This proactive approach can significantly 
        reduce accidents, prevent injuries and ensure workplace safety.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model information
        st.subheader("📋 Model Information")
        st.write("**Model Type:** TensorFlow SavedModel")
        st.write("**Training:** Teachable Machine")
        st.write("**Classes:** 2 (With/Without Helmet)")
        st.write("**Purpose:** PPE Detection")
        
        # Safety guidelines
        st.subheader("🛡️ Safety Guidelines")
        st.info("""
        - Always wear appropriate PPE in manufacturing areas
        - Helmets protect against head injuries
        - Regular safety audits are essential
        - Report any safety violations immediately
        """)
        
        # About section
        st.subheader("ℹ️ About")
        st.write("""
        This AI system helps prevent accidents in MMS by monitoring PPE compliance in real-time.
        """)
    
    # Main content
    st.header("📸 Detection Methods")
    
    # Load model
    predictor = load_model()
    
    if predictor is None:
        st.error("❌ Model could not be loaded. Please check the model files.")
        return
    
    # Create tabs for different detection methods
    tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Camera Capture"])
    
    with tab1:
        st.subheader("Upload Image for Detection")
        handle_image_upload(predictor)
    
    with tab2:
        st.subheader("Real-time Camera Detection")
        handle_camera_capture(predictor)
    
    # Example usage and technical details
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 How to Use")
        st.write("""
        1. **Upload Image**: Click 'Browse files' to select an image
        2. **Camera Capture**: Use your camera for real-time detection
        3. **Analyze**: Click 'Detect Helmet' to run the AI analysis
        4. **Review Results**: Check the detection results and confidence scores
        5. **Follow Guidelines**: Adhere to safety recommendations provided
        """)
    
    with col2:
        st.subheader("🔧 Technical Details")
        st.write("""
        - **Model Architecture**: TensorFlow SavedModel
        - **Input Size**: 224x224 pixels
        - **Preprocessing**: RGB normalization (0-1)
        - **Output**: Binary classification with confidence scores
        - **Real-time Processing**: Camera integration for immediate detection
        """)
    
    # Footer with author information
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background-color: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
        <h3 style="color: #495057; margin-bottom: 1rem;">👨‍💻 Project Developer</h3>
        <p style="font-size: 1.2rem; font-weight: bold; color: #007bff; margin-bottom: 0.5rem;">
            Made by Ansh Kumar Tripathi
        </p>
        <p style="color: #6c757d; margin-bottom: 0.5rem;">
            <strong>Project Type:</strong> Internship/Industrial Project
        </p>
        <p style="color: #6c757d; margin-bottom: 0.5rem;">
            <strong>Technology Stack:</strong> Python, TensorFlow, Streamlit, OpenCV
        </p>
        <p style="color: #6c757d; font-style: italic;">
            Datset from Kaggle and trained using Teachable Machine
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 