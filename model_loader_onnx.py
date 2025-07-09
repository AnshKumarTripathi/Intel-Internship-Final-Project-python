#!/usr/bin/env python3
"""
Model loader for ONNX format compatible with Python 3.13
"""

import onnxruntime as ort
import numpy as np
import os

class ONNXPredictor:
    """Wrapper class for ONNX model prediction"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self.load_model()
    
    def load_model(self):
        """Load the ONNX model"""
        try:
            # Create ONNX Runtime session
            self.session = ort.InferenceSession(self.model_path)
            
            # Get input and output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            print("✅ ONNX model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to load ONNX model: {e}")
            return False
    
    def predict(self, image):
        """Make prediction with ONNX model"""
        try:
            # Run inference
            result = self.session.run([self.output_name], {self.input_name: image})
            return result[0]
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return None

def create_predictor(model_path="model.onnx"):
    """Create an ONNX predictor instance"""
    return ONNXPredictor(model_path)

def test_model_loading():
    """Test if the ONNX model can be loaded and used"""
    print("🔍 Testing ONNX model loading...")
    
    try:
        predictor = create_predictor()
        
        if predictor.session is None:
            print("❌ ONNX model could not be loaded")
            return False
        
        # Test with dummy input
        print("\n🧪 Testing with dummy input...")
        dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        # Apply Teachable Machine normalization
        dummy_input = (dummy_input / 127.5) - 1
        
        prediction = predictor.predict(dummy_input)
        
        if prediction is not None:
            print(f"✅ Prediction shape: {prediction.shape}")
            print(f"✅ Prediction values: {prediction[0]}")
            print(f"✅ Sum of probabilities: {np.sum(prediction[0]):.4f}")
            
            # Check if probabilities sum to ~1
            if 0.99 <= np.sum(prediction[0]) <= 1.01:
                print("✅ Probabilities sum to 1 (as expected)")
            else:
                print("⚠️ Probabilities don't sum to 1")
            
            return True
        else:
            print("❌ Prediction failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_model_loading() 