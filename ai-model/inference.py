"""
Inference script for the trained skin condition model
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import os
from pathlib import Path

class SkinConditionPredictor:
    def __init__(self, model_path):
        """Initialize the predictor with a trained model"""
        self.model = tf.keras.models.load_model(model_path)
        self.img_size = (224, 224)
        
    def preprocess_image(self, image_path):
        """Preprocess an image for prediction"""
        # Load and resize image
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.img_size)
        
        # Convert to array and normalize
        img_array = np.array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image_path):
        """Predict skin condition for an image"""
        try:
            # Preprocess image
            img_array = self.preprocess_image(image_path)
            
            # Make prediction
            prediction = self.model.predict(img_array, verbose=0)
            probability = prediction[0][0]
            
            # Determine result
            if probability > 0.5:
                result = "ACNE_DETECTED"
                confidence = probability
            else:
                result = "NO_ACNE"
                confidence = 1 - probability
            
            return {
                'result': result,
                'confidence': float(confidence),
                'probability': float(probability)
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'result': 'ERROR',
                'confidence': 0.0
            }
    
    def predict_batch(self, image_paths):
        """Predict for multiple images"""
        results = []
        for image_path in image_paths:
            result = self.predict(image_path)
            result['image_path'] = str(image_path)
            results.append(result)
        return results

def main():
    # Model path
    model_path = Path(__file__).parent / 'saved_models' / 'skin_condition_model.h5'
    
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Please train the model first using train_skin_model.py")
        return
    
    # Initialize predictor
    predictor = SkinConditionPredictor(str(model_path))
    
    # Test on some sample images
    test_dir = Path(__file__).parent.parent / 'acne_test_dataset' / 'train'
    
    if test_dir.exists():
        # Get a few sample images
        image_files = list(test_dir.glob('*.jpg'))[:5]
        
        print("Testing model on sample images...")
        print("=" * 50)
        
        for image_path in image_files:
            result = predictor.predict(image_path)
            print(f"Image: {image_path.name}")
            print(f"Result: {result['result']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Probability: {result['probability']:.3f}")
            print("-" * 30)
    else:
        print("No test images found. You can test the model by calling:")
        print("predictor = SkinConditionPredictor('path/to/model.h5')")
        print("result = predictor.predict('path/to/image.jpg')")

if __name__ == "__main__":
    main()
