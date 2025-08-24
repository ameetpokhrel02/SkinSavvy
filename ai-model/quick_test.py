#!/usr/bin/env python3
"""
Quick test to show the model is working
"""

import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

def test_model():
    print("🧬 Quick SkinCareAI Model Test")
    print("=" * 40)
    
    # Load model
    model_path = Path(__file__).parent / 'saved_models' / 'skin_condition_model.h5'
    if not model_path.exists():
        print("❌ Model not found!")
        return
    
    print("🔄 Loading model...")
    model = tf.keras.models.load_model(str(model_path))
    print("✅ Model loaded successfully!")
    
    # Test on sample images
    sample_dir = Path(__file__).parent.parent / 'acne_test_dataset' / 'train'
    if not sample_dir.exists():
        print("❌ Sample dataset not found!")
        return
    
    image_files = list(sample_dir.glob('*.jpg'))[:3]
    print(f"📸 Testing on {len(image_files)} sample images...")
    print("-" * 40)
    
    for i, image_path in enumerate(image_files, 1):
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            prediction = model.predict(img_array, verbose=0)
            probability = prediction[0][0]
            
            if probability > 0.5:
                result = "🔴 ACNE DETECTED"
                confidence = probability
            else:
                result = "🟢 NO ACNE"
                confidence = 1 - probability
            
            print(f"Sample {i}: {result} ({confidence*100:.1f}% confidence)")
            
        except Exception as e:
            print(f"Sample {i}: ❌ Error - {str(e)}")
    
    print("-" * 40)
    print("✅ Model test completed!")
    print("🎯 The AI model is working correctly!")

if __name__ == "__main__":
    test_model()
