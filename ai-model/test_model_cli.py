"""
Command Line Interface to Test Skin Condition Model
Quick testing without GUI
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys
from pathlib import Path
import argparse

class SkinConditionTester:
    def __init__(self):
        self.model = None
        self.img_size = (224, 224)
        self.model_path = Path(__file__).parent / 'saved_models' / 'skin_condition_model.h5'
        
    def load_model(self):
        """Load the trained model"""
        try:
            print("🔄 Loading model...")
            if not self.model_path.exists():
                print("❌ Model file not found! Please train the model first.")
                return False
                
            self.model = tf.keras.models.load_model(str(self.model_path))
            print("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def preprocess_image(self, image_path):
        """Preprocess image for model prediction"""
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.img_size)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
        except Exception as e:
            raise Exception(f"Error preprocessing image: {str(e)}")
    
    def predict_image(self, image_path):
        """Predict skin condition for an image"""
        if self.model is None:
            raise Exception("Model not loaded!")
            
        try:
            img_array = self.preprocess_image(image_path)
            prediction = self.model.predict(img_array, verbose=0)
            probability = prediction[0][0]
            
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
            raise Exception(f"Error during prediction: {str(e)}")
    
    def test_single_image(self, image_path):
        """Test a single image"""
        try:
            print(f"🔍 Analyzing: {os.path.basename(image_path)}")
            
            result = self.predict_image(image_path)
            
            status = "🔴 ACNE DETECTED" if result['result'] == "ACNE_DETECTED" else "🟢 NO ACNE"
            confidence_pct = result['confidence'] * 100
            
            print(f"📊 Results:")
            print(f"   Status: {status}")
            print(f"   Confidence: {confidence_pct:.2f}%")
            print(f"   Probability: {result['probability']:.4f}")
            print(f"   Recommendation: {self.get_recommendation(result)}")
            print()
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    def test_sample_images(self, num_samples=5):
        """Test on sample images from dataset"""
        try:
            print(f"🧪 Testing on {num_samples} sample images...")
            
            sample_dir = Path(__file__).parent.parent / 'acne_test_dataset' / 'train'
            if not sample_dir.exists():
                print("❌ Sample dataset not found!")
                return
            
            image_files = list(sample_dir.glob('*.jpg'))[:num_samples]
            
            if not image_files:
                print("❌ No sample images found!")
                return
            
            print(f"📸 Found {len(image_files)} sample images")
            print("-" * 50)
            
            for i, image_path in enumerate(image_files, 1):
                try:
                    result = self.predict_image(image_path)
                    status = "🔴 ACNE" if result['result'] == "ACNE_DETECTED" else "🟢 NO ACNE"
                    confidence = result['confidence'] * 100
                    
                    print(f"Sample {i}: {status} ({confidence:.1f}%)")
                    
                except Exception as e:
                    print(f"❌ Error with sample {i}: {str(e)}")
            
            print("-" * 50)
            print("✅ Sample testing completed!")
            
        except Exception as e:
            print(f"❌ Error during sample testing: {str(e)}")
    
    def get_recommendation(self, result):
        """Get recommendation based on prediction"""
        if result['result'] == "ACNE_DETECTED":
            if result['confidence'] > 0.8:
                return "High confidence acne detection. Consider consulting a dermatologist."
            elif result['confidence'] > 0.6:
                return "Moderate acne detected. Try gentle skincare products."
            else:
                return "Low confidence acne detection. Monitor skin condition."
        else:
            if result['confidence'] > 0.8:
                return "No acne detected with high confidence. Maintain current skincare routine."
            else:
                return "No significant acne detected. Continue monitoring skin health."

def main():
    parser = argparse.ArgumentParser(description="Test SkinCareAI Model")
    parser.add_argument("--image", "-i", help="Path to image file to test")
    parser.add_argument("--samples", "-s", type=int, default=5, help="Number of sample images to test")
    parser.add_argument("--test-samples", action="store_true", help="Test on sample images from dataset")
    
    args = parser.parse_args()
    
    print("🧬 SkinCareAI Model Tester")
    print("=" * 40)
    
    tester = SkinConditionTester()
    
    if not tester.load_model():
        sys.exit(1)
    
    if args.image:
        if os.path.exists(args.image):
            tester.test_single_image(args.image)
        else:
            print(f"❌ Image file not found: {args.image}")
    
    elif args.test_samples:
        tester.test_sample_images(args.samples)
    
    else:
        print("Usage examples:")
        print("  python test_model_cli.py --image path/to/image.jpg")
        print("  python test_model_cli.py --test-samples")
        print("  python test_model_cli.py --test-samples --samples 10")

if __name__ == "__main__":
    main()
