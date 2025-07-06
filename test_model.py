#!/usr/bin/env python3
"""
Test script to verify model loading and basic functionality
"""

import numpy as np
import os
from model_loader import SavedModelPredictor

def test_model_loading():
    """Test if the model can be loaded successfully"""
    print("🔍 Testing model loading...")
    
    try:
        # Check if model directory exists
        model_path = "model.savedmodel"
        if not os.path.exists(model_path):
            print("❌ Model directory not found!")
            return False
        
        # Use the new model loader
        predictor = SavedModelPredictor(model_path)
        
        if predictor.model is None:
            print("❌ Model could not be loaded")
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
        print(f"❌ Error loading model: {str(e)}")
        return False

def test_labels():
    """Test if labels file exists and is readable"""
    print("\n📝 Testing labels file...")
    
    try:
        with open("labels.txt", "r") as f:
            labels = f.read().strip().split("\n")
        
        print(f"✅ Labels loaded: {labels}")
        
        if len(labels) == 2:
            print("✅ Correct number of labels (2)")
        else:
            print(f"⚠️ Expected 2 labels, got {len(labels)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading labels: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting Helmet Detection Model Tests\n")
    
    # Test model loading
    model_ok = test_model_loading()
    
    # Test labels
    labels_ok = test_labels()
    
    # Summary
    print("\n" + "="*50)
    print("📋 TEST SUMMARY")
    print("="*50)
    
    if model_ok and labels_ok:
        print("✅ All tests passed! The model is ready to use.")
        print("\n🎉 You can now run the Streamlit app with:")
        print("   streamlit run app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        
        if not model_ok:
            print("\n🔧 Model issues to check:")
            print("   - Ensure model.savedmodel directory exists")
            print("   - Check if saved_model.pb is present")
            print("   - Verify TensorFlow version compatibility")
        
        if not labels_ok:
            print("\n🔧 Labels issues to check:")
            print("   - Ensure labels.txt file exists")
            print("   - Check file permissions")

if __name__ == "__main__":
    main() 