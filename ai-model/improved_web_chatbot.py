"""
Improved Web-based Chatbot for Skin Condition Model
Works with categorical classification and provides better acne detection
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

class ImprovedSkinConditionModel:
    def __init__(self):
        self.model = None
        self.img_size = (224, 224)
        self.model_path = Path(__file__).parent / 'saved_models' / 'skin_condition_model.h5'
        self.class_names = ['no_acne', 'acne']  # Categorical classification
        self.load_model()
        
    def load_model(self):
        """Load the trained model"""
        try:
            print("🔄 Loading improved model...")
            if not self.model_path.exists():
                print("❌ Model file not found! Please train the model first.")
                return False
                
            self.model = tf.keras.models.load_model(str(self.model_path))
            print("✅ Improved model loaded successfully!")
            print(f"Model input shape: {self.model.input_shape}")
            print(f"Model output shape: {self.model.output_shape}")
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
            predictions = self.model.predict(img_array, verbose=0)
            
            # Get class probabilities
            acne_probability = predictions[0][1]  # Probability of acne class
            no_acne_probability = predictions[0][0]  # Probability of no_acne class
            
            # Determine result based on higher probability
            if acne_probability > no_acne_probability:
                result = "ACNE_DETECTED"
                confidence = acne_probability
                predicted_class = 1
            else:
                result = "NO_ACNE"
                confidence = no_acne_probability
                predicted_class = 0
            
            return {
                'result': result,
                'confidence': float(confidence),
                'acne_probability': float(acne_probability),
                'no_acne_probability': float(no_acne_probability),
                'predicted_class': predicted_class,
                'class_name': self.class_names[predicted_class]
            }
            
        except Exception as e:
            raise Exception(f"Error during prediction: {str(e)}")
    
    def get_recommendation(self, result):
        """Get detailed recommendation based on prediction"""
        if result['result'] == "ACNE_DETECTED":
            confidence = result['confidence']
            if confidence > 0.9:
                return "🔴 HIGH CONFIDENCE: Significant acne detected. Consider consulting a dermatologist for professional treatment."
            elif confidence > 0.7:
                return "🟠 MODERATE-HIGH: Acne detected. Use gentle cleansers and avoid harsh products. Consider over-the-counter treatments."
            elif confidence > 0.5:
                return "🟡 MODERATE: Some acne signs detected. Monitor your skin and maintain a gentle skincare routine."
            else:
                return "🟢 LOW-MODERATE: Minor acne signs detected. Continue with gentle skincare and monitor for changes."
        else:
            confidence = result['confidence']
            if confidence > 0.9:
                return "🟢 EXCELLENT: No acne detected with high confidence. Your skin appears healthy. Maintain your current routine."
            elif confidence > 0.7:
                return "🟢 GOOD: No significant acne detected. Continue with your skincare routine and maintain good hygiene."
            else:
                return "🟡 UNCERTAIN: No clear acne detection. Consider retaking the photo with better lighting or consult a professional."

# Initialize the model
model = ImprovedSkinConditionModel()

# HTML template for the improved web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>SkinCareAI - Improved Acne Detection</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        h1 {
            color: #2E86AB;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .upload-section {
            text-align: center;
            margin: 30px 0;
            padding: 30px;
            border: 3px dashed #667eea;
            border-radius: 15px;
            background: linear-gradient(45deg, #f8f9fa, #e9ecef);
            transition: all 0.3s ease;
        }
        .upload-section:hover {
            border-color: #764ba2;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .btn {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            padding: 12px 25px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            margin: 8px;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        .btn-secondary {
            background: linear-gradient(45deg, #11998e, #38ef7d);
        }
        .btn-danger {
            background: linear-gradient(45deg, #ff416c, #ff4b2b);
        }
        .result {
            margin: 25px 0;
            padding: 20px;
            border-radius: 10px;
            display: none;
            animation: fadeIn 0.5s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .result.acne {
            background: linear-gradient(135deg, #ffebee, #ffcdd2);
            border-left: 5px solid #f44336;
        }
        .result.no-acne {
            background: linear-gradient(135deg, #e8f5e8, #c8e6c9);
            border-left: 5px solid #4caf50;
        }
        .chat-log {
            background: #f8f9fa;
            border: 2px solid #dee2e6;
            border-radius: 10px;
            padding: 20px;
            height: 400px;
            overflow-y: auto;
            margin-top: 25px;
            box-shadow: inset 0 2px 5px rgba(0,0,0,0.1);
        }
        .chat-message {
            margin: 15px 0;
            padding: 12px 15px;
            border-radius: 10px;
            animation: slideIn 0.3s ease-out;
        }
        @keyframes slideIn {
            from { transform: translateX(-20px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        .chat-message.user {
            background: linear-gradient(135deg, #e3f2fd, #bbdefb);
            margin-left: 30px;
            border-left: 4px solid #2196f3;
        }
        .chat-message.bot {
            background: linear-gradient(135deg, #f1f8e9, #c8e6c9);
            margin-right: 30px;
            border-left: 4px solid #4caf50;
        }
        .status {
            text-align: center;
            padding: 15px;
            margin: 15px 0;
            border-radius: 10px;
            font-weight: bold;
        }
        .status.success {
            background: linear-gradient(135deg, #d4edda, #c3e6cb);
            color: #155724;
            border: 2px solid #28a745;
        }
        .status.error {
            background: linear-gradient(135deg, #f8d7da, #f5c6cb);
            color: #721c24;
            border: 2px solid #dc3545;
        }
        .status.info {
            background: linear-gradient(135deg, #d1ecf1, #bee5eb);
            color: #0c5460;
            border: 2px solid #17a2b8;
        }
        .probability-bar {
            background: #e9ecef;
            border-radius: 10px;
            height: 20px;
            margin: 10px 0;
            overflow: hidden;
        }
        .probability-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 0.5s ease;
        }
        .probability-fill.acne {
            background: linear-gradient(90deg, #ff416c, #ff4b2b);
        }
        .probability-fill.no-acne {
            background: linear-gradient(90deg, #11998e, #38ef7d);
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border: 2px solid #dee2e6;
        }
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
            margin: 5px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧬 SkinCareAI - Improved Acne Detection</h1>
        
        <div id="status" class="status info">
            Status: Ready to analyze skin images with improved accuracy
        </div>
        
        <div class="upload-section">
            <h3>📸 Upload Image for Advanced Analysis</h3>
            <p>Get detailed acne detection with confidence scores and recommendations</p>
            <input type="file" id="imageInput" accept="image/*" style="display: none;">
            <button class="btn" onclick="document.getElementById('imageInput').click()">Choose Image</button>
            <button class="btn btn-secondary" onclick="testSampleImages()">Test Sample Images</button>
            <button class="btn btn-danger" onclick="clearChat()">Clear Chat</button>
        </div>
        
        <div id="result" class="result"></div>
        
        <div class="chat-log" id="chatLog">
            <div class="chat-message bot">
                🤖 Welcome to SkinCareAI! I'm your advanced skin analysis assistant. 
                Upload an image to get detailed acne detection with confidence scores and personalized recommendations.
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
            
            updateStatus('🔍 Analyzing image with improved model...', 'info');
            addChatMessage('📤 You uploaded: ' + file.name, 'user');
            
            fetch('/analyze', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    displayResult(data.result);
                    addChatMessage(formatResult(data.result), 'bot');
                    updateStatus('✅ Analysis completed successfully!', 'success');
                } else {
                    updateStatus('❌ Error: ' + data.error, 'error');
                    addChatMessage('❌ Error analyzing image: ' + data.error, 'bot');
                }
            })
            .catch(error => {
                updateStatus('❌ Network error: ' + error.message, 'error');
                addChatMessage('❌ Network error occurred', 'bot');
            });
        }

        function testSampleImages() {
            updateStatus('🧪 Testing on sample images...', 'info');
            addChatMessage('🧪 Testing improved model on sample images...', 'user');
            
            fetch('/test-samples')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    updateStatus('✅ Sample testing completed!', 'success');
                    data.results.forEach(result => {
                        addChatMessage(`📊 Sample: ${result.status} (${result.confidence}% confidence)`, 'bot');
                    });
                } else {
                    updateStatus('❌ Error: ' + data.error, 'error');
                    addChatMessage('❌ Error testing samples: ' + data.error, 'bot');
                }
            })
            .catch(error => {
                updateStatus('❌ Network error: ' + error.message, 'error');
                addChatMessage('❌ Network error during testing', 'bot');
            });
        }

        function displayResult(result) {
            const resultDiv = document.getElementById('result');
            const status = result.result === 'ACNE_DETECTED' ? 'acne' : 'no-acne';
            const icon = result.result === 'ACNE_DETECTED' ? '🔴' : '🟢';
            
            resultDiv.className = `result ${status}`;
            resultDiv.style.display = 'block';
            
            const acnePercent = (result.acne_probability * 100).toFixed(1);
            const noAcnePercent = (result.no_acne_probability * 100).toFixed(1);
            
            resultDiv.innerHTML = `
                <h3>${icon} Advanced Analysis Results</h3>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div>Status</div>
                        <div class="stat-value">${result.result}</div>
                    </div>
                    <div class="stat-card">
                        <div>Confidence</div>
                        <div class="stat-value">${(result.confidence * 100).toFixed(1)}%</div>
                    </div>
                    <div class="stat-card">
                        <div>Class</div>
                        <div class="stat-value">${result.class_name}</div>
                    </div>
                </div>
                
                <h4>📊 Probability Breakdown:</h4>
                <div>
                    <div>Acne Probability: ${acnePercent}%</div>
                    <div class="probability-bar">
                        <div class="probability-fill acne" style="width: ${acnePercent}%"></div>
                    </div>
                </div>
                <div>
                    <div>No Acne Probability: ${noAcnePercent}%</div>
                    <div class="probability-bar">
                        <div class="probability-fill no-acne" style="width: ${noAcnePercent}%"></div>
                    </div>
                </div>
                
                <h4>💡 Recommendation:</h4>
                <p><strong>${result.recommendation}</strong></p>
            `;
        }

        function formatResult(result) {
            const status = result.result === 'ACNE_DETECTED' ? '🔴 ACNE DETECTED' : '🟢 NO ACNE';
            const confidence = (result.confidence * 100).toFixed(1);
            const acneProb = (result.acne_probability * 100).toFixed(1);
            const noAcneProb = (result.no_acne_probability * 100).toFixed(1);
            
            return `📊 ${status} (${confidence}% confidence)
                    📈 Acne: ${acneProb}% | No Acne: ${noAcneProb}%
                    💡 ${result.recommendation}`;
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

        function clearChat() {
            const chatLog = document.getElementById('chatLog');
            chatLog.innerHTML = `
                <div class="chat-message bot">
                    🤖 Chat cleared! Ready for new analysis.
                </div>
            `;
            updateStatus('Chat cleared. Ready for new analysis.', 'info');
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
    print("🚀 Starting Improved SkinCareAI Web Chatbot...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🔬 This version uses improved model with better acne detection")
    app.run(host='0.0.0.0', port=5000, debug=False)
