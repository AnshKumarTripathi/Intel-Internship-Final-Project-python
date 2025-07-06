# 🪖 Helmet Detection System

An AI-powered Streamlit application for detecting helmet presence in manufacturing and construction environments. This system helps prevent accidents in MMS (Modular Manufacturing System) by monitoring PPE (Personal Protective Equipment) compliance in real-time.

## 🎯 Project Overview

**Project**: AI based accident prevention in MMS (modular manufacturing system)

**Problem Statement**: Accidents in MMS often occur due to delayed human response or oversight. By implementing an AI-powered camera vision system, activities can be continuously monitored and machines can be auto-stopped in real-time upon detecting potential hazards.

**Solution**: This proactive approach can significantly reduce accidents, prevent injuries, and ensure workplace safety through automated helmet detection.

**Developer**: Ansh Kumar Tripathi  
**Project Type**: Internship/Industrial Project  
**Technology Stack**: Python, TensorFlow, Streamlit, OpenCV  
**Dataset**: Kaggle dataset trained using Teachable Machine

## 🚀 Features

### **Core Detection Features**

- **Image Upload**: Upload images in PNG, JPG, or JPEG formats
- **Real-time Camera Capture**: Direct camera integration for immediate detection
- **AI-powered Detection**: Helmet detection using TensorFlow SavedModel
- **Confidence Scoring**: Detailed confidence scores for predictions
- **Dual Interface**: Tab-based system for upload and camera methods

### **Safety & Alert System**

- **Safety Alert System**: Automatic machine stop simulation when no helmet detected
- **Real-time Alerts**: Prominent safety violation notifications
- **Safety Recommendations**: Contextual guidelines based on detection results
- **Compliance Tracking**: Monitor safety protocol adherence

### **User Interface**

- **Modern Dashboard**: Professional Streamlit interface
- **Responsive Design**: Works on different screen sizes
- **Visual Feedback**: Color-coded results and progress bars
- **Professional Styling**: High contrast, readable text for reports
- **Screenshot-Ready**: Optimized layout for documentation

## 📋 Requirements

- Python 3.8 or higher
- TensorFlow >= 2.16.0
- Streamlit 1.28.1
- OpenCV >= 4.8.0
- Pillow >= 10.0.0
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0

## 🛠️ Installation

1. **Clone or download the project files**

   ```bash
   # Ensure you have the following files in your project directory:
   # - app.py (main Streamlit application)
   # - model_loader.py (SavedModel handler)
   # - test_model.py (model testing script)
   # - opencv_demo.py (camera demo)
   # - requirements.txt (dependencies)
   # - model.savedmodel/ (directory with trained model)
   # - labels.txt (class labels)
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Verify model files**
   Ensure the `model.savedmodel` directory contains:

   - `saved_model.pb`
   - `variables/` directory
   - `assets/` directory

4. **Test the model**
   ```bash
   python test_model.py
   ```

## 🎮 Usage

1. **Start the application**

   ```bash
   streamlit run app.py
   ```

2. **Open your browser**

   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in the terminal

3. **Choose detection method**

   - **Upload Image**: Click "Browse files" to select an image
   - **Camera Capture**: Use your camera for real-time detection
   - Supported formats: PNG, JPG, JPEG

4. **Run detection**

   - Click "🚀 Detect Helmet" button
   - Wait for the analysis to complete

5. **Review results**
   - Check the detection result (With/Without Helmet)
   - Review confidence scores
   - Follow safety recommendations
   - Check safety alerts if violation detected

## 📊 Model Information

- **Model Type**: TensorFlow SavedModel
- **Training Platform**: Teachable Machine
- **Dataset Source**: Kaggle
- **Classes**: 2 (With Helmet, Without Helmet)
- **Input Size**: 224x224 pixels
- **Preprocessing**: RGB normalization using Teachable Machine format
- **Output**: Binary classification with confidence scores
- **Model Loading**: Compatible with TensorFlow >= 2.16.0

## 🏗️ Project Structure

```
project/
├── app.py                 # Main Streamlit application
├── model_loader.py        # SavedModel handler for TensorFlow compatibility
├── test_model.py          # Model testing and validation script
├── opencv_demo.py         # OpenCV camera demo (standalone)
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── labels.txt            # Class labels
├── problem-statment.txt  # Project problem statement
└── model.savedmodel/     # Trained TensorFlow model
    ├── saved_model.pb
    ├── variables/
    └── assets/
```

## 🔧 Technical Details

### Model Architecture

- **Framework**: TensorFlow/Keras
- **Format**: SavedModel
- **Input**: RGB images (224x224)
- **Output**: 2-class probabilities

### Image Preprocessing

1. Convert to RGB format
2. Resize to 224x224 pixels
3. Normalize pixel values using Teachable Machine format: `(image / 127.5) - 1`
4. Add batch dimension

### Detection Logic

- **Class 0**: "With Helmet" (Safety compliance)
- **Class 1**: "Without Helmet" (Safety violation)
- **Confidence Threshold**: Displayed as percentage

## 🛡️ Safety Guidelines

### When Helmet is Detected ✅

- Continue following safety protocols
- Ensure helmet is properly fitted
- Check for any damage to PPE
- Set a good example for others

### When No Helmet is Detected ⚠️

- **Safety Alert**: Automatic machine stop simulation activated
- Immediately put on appropriate safety helmet
- Check helmet for proper fit
- Ensure helmet meets safety standards
- Report to supervisor if PPE is damaged
- Wait for safety clearance before resuming operations

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Error**

   - Ensure `model.savedmodel` directory exists
   - Check file permissions
   - Verify TensorFlow version compatibility

2. **Image Upload Issues**

   - Use supported formats (PNG, JPG, JPEG)
   - Check file size (recommended < 10MB)
   - Ensure image is not corrupted

3. **Prediction Errors**
   - Verify image preprocessing
   - Check model input requirements
   - Ensure proper image format

### Performance Tips

- Use images with clear visibility of the person
- Ensure good lighting conditions
- Avoid heavily blurred or distorted images
- For best results, use images with helmet clearly visible

## 🔮 Future Enhancements

- **Real-time Video Processing**: Live camera feed analysis
- **Multiple PPE Detection**: Gloves, safety glasses, etc.
- **Advanced Alert System**: Email/SMS notifications for violations
- **Analytics Dashboard**: Historical compliance tracking
- **Database Integration**: Store detection results and compliance data
- **Mobile App**: iOS/Android companion app
- **API Integration**: Connect with existing safety management systems
- **Multi-language Support**: International deployment capabilities

## 🎯 Additional Features

### **Camera Integration**

- **Real-time Capture**: Direct camera access for immediate detection
- **Tab-based Interface**: Separate tabs for upload and camera methods
- **Instant Analysis**: Quick processing of captured images

### **Safety Alert System**

- **Machine Stop Simulation**: Realistic manufacturing environment alerts
- **Visual Alerts**: Prominent red-bordered safety violation notifications
- **Action Instructions**: Clear guidance on required safety actions
- **Professional Styling**: High contrast text for optimal readability

---

**⚠️ Important**: This system is designed to assist in safety monitoring but should not replace human supervision or safety protocols. Always follow workplace safety guidelines and regulations.

**👨‍💻 Developer**: Ansh Kumar Tripathi  
**📧 Contact**: For technical support or collaboration inquiries
