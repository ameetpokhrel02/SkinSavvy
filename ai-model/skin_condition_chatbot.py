"""
Simple Chatbot to Test Skin Condition Model
Tests the trained model with image uploads and provides analysis results
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading

class SkinConditionChatbot:
    def __init__(self):
        self.model = None
        self.img_size = (224, 224)
        self.model_path = Path(__file__).parent / 'saved_models' / 'skin_condition_model.h5'
        
        # Initialize GUI
        self.root = tk.Tk()
        self.root.title("SkinCareAI - Model Testing Chatbot")
        self.root.geometry("600x500")
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI components"""
        # Title
        title_label = tk.Label(self.root, text="🧬 SkinCareAI Model Tester", 
                              font=("Arial", 16, "bold"), fg="#2E86AB")
        title_label.pack(pady=10)
        
        # Status frame
        status_frame = tk.Frame(self.root)
        status_frame.pack(pady=5, padx=10, fill="x")
        
        self.status_label = tk.Label(status_frame, text="Status: Loading model...", 
                                    font=("Arial", 10), fg="#666")
        self.status_label.pack(side="left")
        
        # Buttons frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        self.load_model_btn = tk.Button(button_frame, text="Load Model", 
                                       command=self.load_model, bg="#4CAF50", fg="white")
        self.load_model_btn.pack(side="left", padx=5)
        
        self.upload_btn = tk.Button(button_frame, text="Upload Image", 
                                   command=self.upload_image, bg="#2196F3", fg="white", state="disabled")
        self.upload_btn.pack(side="left", padx=5)
        
        self.test_sample_btn = tk.Button(button_frame, text="Test Sample Images", 
                                        command=self.test_sample_images, bg="#FF9800", fg="white", state="disabled")
        self.test_sample_btn.pack(side="left", padx=5)
        
        # Chat display
        chat_frame = tk.Frame(self.root)
        chat_frame.pack(pady=10, padx=10, fill="both", expand=True)
        
        chat_label = tk.Label(chat_frame, text="Chat Log:", font=("Arial", 12, "bold"))
        chat_label.pack(anchor="w")
        
        self.chat_display = scrolledtext.ScrolledText(chat_frame, height=15, width=70)
        self.chat_display.pack(fill="both", expand=True)
        
        # Initialize model loading
        self.load_model()
        
    def log_message(self, message, message_type="info"):
        """Add message to chat display"""
        colors = {
            "info": "black",
            "success": "green",
            "error": "red",
            "warning": "orange"
        }
        
        self.chat_display.insert(tk.END, f"{message}\n")
        self.chat_display.see(tk.END)
        self.root.update()
        
    def load_model(self):
        """Load the trained model"""
        def load():
            try:
                self.log_message("🔄 Loading model...", "info")
                self.status_label.config(text="Status: Loading model...")
                
                if not self.model_path.exists():
                    self.log_message("❌ Model file not found! Please train the model first.", "error")
                    self.status_label.config(text="Status: Model not found")
                    return
                
                self.model = tf.keras.models.load_model(str(self.model_path))
                self.log_message("✅ Model loaded successfully!", "success")
                self.status_label.config(text="Status: Model ready")
                
                # Enable buttons
                self.upload_btn.config(state="normal")
                self.test_sample_btn.config(state="normal")
                
            except Exception as e:
                self.log_message(f"❌ Error loading model: {str(e)}", "error")
                self.status_label.config(text="Status: Error loading model")
        
        threading.Thread(target=load, daemon=True).start()
        
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
    
    def upload_image(self):
        """Upload and analyze an image"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if file_path:
            self.analyze_image(file_path)
    
    def analyze_image(self, image_path):
        """Analyze a specific image"""
        def analyze():
            try:
                self.log_message(f"🔍 Analyzing image: {os.path.basename(image_path)}", "info")
                
                result = self.predict_image(image_path)
                
                # Format the result
                status = "🔴 ACNE DETECTED" if result['result'] == "ACNE_DETECTED" else "🟢 NO ACNE"
                confidence_pct = result['confidence'] * 100
                
                analysis_text = f"""
📊 Analysis Results:
   Status: {status}
   Confidence: {confidence_pct:.2f}%
   Probability: {result['probability']:.4f}
   
💡 Recommendation: {self.get_recommendation(result)}
"""
                
                self.log_message(analysis_text, "success")
                
            except Exception as e:
                self.log_message(f"❌ Error analyzing image: {str(e)}", "error")
        
        threading.Thread(target=analyze, daemon=True).start()
    
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
    
    def test_sample_images(self):
        """Test the model on sample images from the dataset"""
        def test_samples():
            try:
                self.log_message("🧪 Testing model on sample images...", "info")
                
                # Find sample images
                sample_dir = Path(__file__).parent.parent / 'acne_test_dataset' / 'train'
                if not sample_dir.exists():
                    self.log_message("❌ Sample dataset not found!", "error")
                    return
                
                # Get a few sample images
                image_files = list(sample_dir.glob('*.jpg'))[:5]
                
                if not image_files:
                    self.log_message("❌ No sample images found!", "error")
                    return
                
                self.log_message(f"📸 Found {len(image_files)} sample images", "info")
                
                for i, image_path in enumerate(image_files, 1):
                    try:
                        result = self.predict_image(image_path)
                        status = "🔴 ACNE" if result['result'] == "ACNE_DETECTED" else "🟢 NO ACNE"
                        confidence = result['confidence'] * 100
                        
                        self.log_message(f"Sample {i}: {status} ({confidence:.1f}%)", "info")
                        
                    except Exception as e:
                        self.log_message(f"❌ Error with sample {i}: {str(e)}", "error")
                
                self.log_message("✅ Sample testing completed!", "success")
                
            except Exception as e:
                self.log_message(f"❌ Error during sample testing: {str(e)}", "error")
        
        threading.Thread(target=test_samples, daemon=True).start()
    
    def run(self):
        """Start the chatbot"""
        self.root.mainloop()

def main():
    print("🚀 Starting SkinCareAI Model Testing Chatbot...")
    chatbot = SkinConditionChatbot()
    chatbot.run()

if __name__ == "__main__":
    main()