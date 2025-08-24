"""
Simple Web-based Chatbot to Test Skin Condition Model
Uses Flask for web interface - no tkinter required
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify, send_file
import base64
import io
import json

app = Flask(__name__)

class SkinConditionModel:
    def __init__(self):
        self.model = None
        self.img_size = (224, 224)
        self.model_path = Path(__file__).parent / 'saved_models' / 'skin_condition_model.h5'
        self.load_model()
        
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
    
    def preprocess_image(self, image_data):
        """Preprocess image for model prediction"""
        try:
            img = Image.open(io.BytesIO(image_data)).convert('RGB')
            img = img.resize(self.img_size)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
        except Exception as e:
            raise Exception(f"Error preprocessing image: {str(e)}")
    
    def predict_image(self, image_data):
        """Predict skin condition for an image"""
        if self.model is None:
            raise Exception("Model not loaded!")
            
        try:
            img_array = self.preprocess_image(image_data)
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

# Initialize the model
model = SkinConditionModel()

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>SkinCareAI - Model Tester</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2E86AB;
            text-align: center;
            margin-bottom: 30px;
        }
        .upload-section {
            text-align: center;
            margin: 20px 0;
            padding: 20px;
            border: 2px dashed #ccc;
            border-radius: 10px;
        }
        .upload-section:hover {
            border-color: #2E86AB;
        }
        .btn {
            background: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
        }
        .btn:hover {
            background: #45a049;
        }
        .btn-secondary {
            background: #2196F3;
        }
        .btn-secondary:hover {
            background: #1976D2;
        }
        .result {
            margin: 20px 0;
            padding: 15px;
            border-radius: 5px;
            display: none;
        }
        .result.acne {
            background: #ffebee;
            border-left: 4px solid #f44336;
        }
        .result.no-acne {
            background: #e8f5e8;
            border-left: 4px solid #4caf50;
        }
        .chat-log {
            background: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            height: 300px;
            overflow-y: auto;
            margin-top: 20px;
        }
        .chat-message {
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
        }
        .chat-message.user {
            background: #e3f2fd;
            margin-left: 20px;
        }
        .chat-message.bot {
            background: #f1f8e9;
            margin-right: 20px;
        }
        .status {
            text-align: center;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .status.success {
            background: #d4edda;
            color: #155724;
        }
        .status.error {
            background: #f8d7da;
            color: #721c24;
        }
        .status.info {
            background: #d1ecf1;
            color: #0c5460;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧬 SkinCareAI Model Tester</h1>
        
        <div id="status" class="status info">
            Status: Ready to test images
        </div>
        
        <div class="upload-section">
            <h3>Upload Image for Analysis</h3>
            <input type="file" id="imageInput" accept="image/*" style="display: none;">
            <button class="btn" onclick="document.getElementById('imageInput').click()">Choose Image</button>
            <button class="btn btn-secondary" onclick="testSampleImages()">Test Sample Images</button>
        </div>
        
        <div id="result" class="result"></div>
        
        <div class="chat-log" id="chatLog">
            <div class="chat-message bot">
                🤖 Hello! I'm your SkinCareAI assistant. Upload an image to analyze for acne detection.
            </div>
        </div>
    </div>

    <script>
        document.getElementById('imageInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                analyzeImage(file);
            }
        });

        function analyzeImage(file) {
            const formData = new FormData();
            formData.append('image', file);
            
            updateStatus('Analyzing image...', 'info');
            addChatMessage('You uploaded: ' + file.name, 'user');
            
            fetch('/analyze', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    displayResult(data.result);
                    addChatMessage(formatResult(data.result), 'bot');
                } else {
                    updateStatus('Error: ' + data.error, 'error');
                    addChatMessage('❌ Error analyzing image', 'bot');
                }
            })
            .catch(error => {
                updateStatus('Error: ' + error.message, 'error');
                addChatMessage('❌ Network error', 'bot');
            });
        }

        function testSampleImages() {
            updateStatus('Testing sample images...', 'info');
            addChatMessage('Testing on sample images...', 'user');
            
            fetch('/test-samples')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    updateStatus('Sample testing completed!', 'success');
                    data.results.forEach(result => {
                        addChatMessage(`Sample: ${result.status} (${result.confidence}%)`, 'bot');
                    });
                } else {
                    updateStatus('Error: ' + data.error, 'error');
                    addChatMessage('❌ Error testing samples', 'bot');
                }
            })
            .catch(error => {
                updateStatus('Error: ' + error.message, 'error');
                addChatMessage('❌ Network error', 'bot');
            });
        }

        function displayResult(result) {
            const resultDiv = document.getElementById('result');
            const status = result.result === 'ACNE_DETECTED' ? 'acne' : 'no-acne';
            const icon = result.result === 'ACNE_DETECTED' ? '🔴' : '🟢';
            
            resultDiv.className = `result ${status}`;
            resultDiv.style.display = 'block';
            resultDiv.innerHTML = `
                <h3>${icon} Analysis Results</h3>
                <p><strong>Status:</strong> ${result.result}</p>
                <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(2)}%</p>
                <p><strong>Probability:</strong> ${result.probability.toFixed(4)}</p>
                <p><strong>Recommendation:</strong> ${result.recommendation}</p>
            `;
        }

        function formatResult(result) {
            const status = result.result === 'ACNE_DETECTED' ? '🔴 ACNE DETECTED' : '🟢 NO ACNE';
            const confidence = (result.confidence * 100).toFixed(1);
            return `📊 ${status} (${confidence}% confidence) - ${result.recommendation}`;
        }

        function updateStatus(message, type) {
            const statusDiv = document.getElementById('status');
            statusDiv.textContent = message;
            statusDiv.className = `status ${type}`;
        }

        function addChatMessage(message, sender) {
            const chatLog = document.getElementById('chatLog');
            const messageDiv = document.createElement('div');
            messageDiv.className = `chat-message ${sender}`;
            messageDiv.textContent = message;
            chatLog.appendChild(messageDiv);
            chatLog.scrollTop = chatLog.scrollHeight;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image uploaded'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Read image data
        image_data = file.read()
        
        # Make prediction
        result = model.predict_image(image_data)
        result['recommendation'] = model.get_recommendation(result)
        
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/test-samples')
def test_samples():
    try:
        # Find sample images
        sample_dir = Path(__file__).parent.parent / 'acne_test_dataset' / 'train'
        if not sample_dir.exists():
            return jsonify({'success': False, 'error': 'Sample dataset not found'})
        
        # Get sample images
        image_files = list(sample_dir.glob('*.jpg'))[:5]
        
        if not image_files:
            return jsonify({'success': False, 'error': 'No sample images found'})
        
        results = []
        for image_path in image_files:
            try:
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                
                result = model.predict_image(image_data)
                results.append({
                    'filename': image_path.name,
                    'status': '🔴 ACNE' if result['result'] == 'ACNE_DETECTED' else '🟢 NO ACNE',
                    'confidence': f"{result['confidence'] * 100:.1f}%"
                })
                
            except Exception as e:
                results.append({
                    'filename': image_path.name,
                    'status': '❌ ERROR',
                    'confidence': str(e)
                })
        
        return jsonify({'success': True, 'results': results})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("🚀 Starting SkinCareAI Web Chatbot...")
    print("📱 Open your browser and go to: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
