#!/usr/bin/env python3
"""
OpenCV Demo for Helmet Detection using SavedModel
Adapted from official Teachable Machine code
"""

import tensorflow as tf
import cv2
import numpy as np
import os

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

def load_model_savedmodel():
    """Load SavedModel format"""
    model_path = "model.savedmodel"
    
    try:
        # Try different loading methods
        try:
            model = tf.keras.models.load_model(model_path)
            print("✅ Model loaded using tf.keras.models.load_model")
            return model
        except:
            try:
                model = tf.saved_model.load(model_path)
                print("✅ Model loaded using tf.saved_model.load")
                return model
            except:
                model = tf.keras.models.load_model(model_path, compile=False)
                print("✅ Model loaded with compile=False")
                return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def predict_with_savedmodel(model, image):
    """Predict using SavedModel"""
    try:
        # Try different prediction methods
        try:
            prediction = model.predict(image, verbose=0)
        except:
            try:
                prediction = model(image)
            except:
                prediction = model.signatures['serving_default'](tf.constant(image))
                output_key = list(prediction.keys())[0]
                prediction = prediction[output_key].numpy()
        
        return prediction
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None

def main():
    # Load the model
    model = load_model_savedmodel()
    if model is None:
        print("❌ Failed to load model")
        return
    
    # Load the labels
    try:
        with open("labels.txt", "r") as f:
            class_names = f.readlines()
        print(f"✅ Labels loaded: {[name.strip() for name in class_names]}")
    except Exception as e:
        print(f"❌ Error loading labels: {e}")
        return
    
    # CAMERA can be 0 or 1 based on default camera of your computer
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("❌ Could not open camera")
        return
    
    print("🎥 Camera opened successfully")
    print("📱 Press 'ESC' to exit")
    
    while True:
        # Grab the webcamera's image.
        ret, image = camera.read()
        
        if not ret:
            print("❌ Failed to grab frame")
            break
        
        # Resize the raw image into (224-height,224-width) pixels
        image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        
        # Show the image in a window
        cv2.imshow("Webcam Image", image_resized)
        
        # Make the image a numpy array and reshape it to the models input shape.
        image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
        
        # Normalize the image array (using the same normalization as Teachable Machine)
        image_normalized = (image_array / 127.5) - 1
        
        # Predicts the model
        prediction = predict_with_savedmodel(model, image_normalized)
        
        if prediction is not None:
            index = np.argmax(prediction)
            class_name = class_names[index].strip()
            confidence_score = prediction[0][index]
            
            # Print prediction and confidence score
            print(f"Class: {class_name}, Confidence Score: {confidence_score * 100:.1f}%")
            
            # Display on image
            text = f"{class_name}: {confidence_score * 100:.1f}%"
            cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Listen to the keyboard for presses.
        keyboard_input = cv2.waitKey(1)
        
        # 27 is the ASCII for the esc key on your keyboard.
        if keyboard_input == 27:
            break
    
    camera.release()
    cv2.destroyAllWindows()
    print("👋 Demo ended")

if __name__ == "__main__":
    main() 