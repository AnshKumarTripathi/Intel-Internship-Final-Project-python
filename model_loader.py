#!/usr/bin/env python3
"""
Model loader for SavedModel format compatible with newer TensorFlow versions
"""

import tensorflow as tf
import numpy as np
import os

class SavedModelPredictor:
    """Wrapper class for SavedModel prediction"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the SavedModel with multiple fallback methods"""
        try:
            # Method 1: Try loading as SavedModel
            self.model = tf.saved_model.load(self.model_path)
            print("✅ Model loaded using tf.saved_model.load")
            return True
        except Exception as e1:
            print(f"⚠️ Method 1 failed: {e1}")
            try:
                # Method 2: Try with keras load_model
                self.model = tf.keras.models.load_model(self.model_path)
                print("✅ Model loaded using tf.keras.models.load_model")
                return True
            except Exception as e2:
                print(f"⚠️ Method 2 failed: {e2}")
                try:
                    # Method 3: Try with compile=False
                    self.model = tf.keras.models.load_model(self.model_path, compile=False)
                    print("✅ Model loaded with compile=False")
                    return True
                except Exception as e3:
                    print(f"❌ All loading methods failed: {e3}")
                    return False
    
    def predict(self, image):
        """Make prediction with multiple fallback methods"""
        try:
            # Method 1: Try standard predict
            if hasattr(self.model, 'predict'):
                result = self.model.predict(image, verbose=0)
                # Ensure we return numpy array
                if hasattr(result, 'numpy'):
                    return result.numpy()
                return result
        except:
            pass
        
        try:
            # Method 2: Try direct call
            if callable(self.model):
                result = self.model(image)
                # Ensure we return numpy array
                if hasattr(result, 'numpy'):
                    return result.numpy()
                return result
        except:
            pass
        
        try:
            # Method 3: Try SavedModel signature
            if hasattr(self.model, 'signatures'):
                # Get the serving signature
                signature = self.model.signatures['serving_default']
                # Convert numpy to tensor
                input_tensor = tf.constant(image, dtype=tf.float32)
                # Make prediction
                result = signature(input_tensor)
                # Get the output (first key)
                output_key = list(result.keys())[0]
                return result[output_key].numpy()
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return None

def create_predictor(model_path="model.savedmodel"):
    """Create a predictor instance"""
    return SavedModelPredictor(model_path)

def test_model_loading():
    """Test if the model can be loaded and used"""
    print("🔍 Testing model loading...")
    
    try:
        predictor = create_predictor()
        
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
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_model_loading() 