"""
Skin Classification - Multi-Class Skin Condition Detection
Main script for inference and web interface
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify
import base64
import io
import json
from datetime import datetime

app = Flask(__name__)

# --- CLASS DEFINITIONS ---
CLASS_NAMES = {
    0: 'Acne',
    1: 'Pimple', 
    2: 'Spots',
    3: 'Mole1',
    4: 'Mole2',
    5: 'Scar'
}

CLASS_DESCRIPTIONS = {
    0: 'Bumpy, small-big, white-brown cysts with no redness',
    1: 'Bumpy, small-big, red-white cysts, any red big bump (must be red)',
    2: 'Flat, small, circular, dark brown',
    3: 'Flat/black/small/concentrated',
    4: 'Bumpy/black/dark-brown',
    5: 'Flat/any shape/any shade'
}

class MultiClassSkinClassifier:
    def __init__(self):
        self.model = None
        self.img_size = (380, 380)  # EfficientNetB4 optimal size
        self.model_path = Path(__file__).parent / 'models' / 'skin_classification_model.h5'
        self.load_model()
        
    def load_model(self):
        """Load the trained model"""
        try:
            print("🔄 Loading multi-class skin classification model...")
            
            # Try to find the latest model file
            models_dir = Path(__file__).parent / 'models'
            if models_dir.exists():
                model_files = list(models_dir.glob('skin_classification_model_*.h5'))
                if model_files:
                    # Get the latest model file
                    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                    self.model_path = latest_model
                    print(f"📁 Found model: {latest_model.name}")
                elif self.model_path.exists():
                    print(f"📁 Using default model: {self.model_path.name}")
                else:
                    print("❌ No trained model found! Please train the model first.")
                    return False
            else:
                print("❌ Models directory not found!")
                return False
                
            self.model = tf.keras.models.load_model(str(self.model_path))
            print("✅ Multi-class model loaded successfully!")
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
            class_probabilities = predictions[0]
            predicted_class = np.argmax(class_probabilities)
            confidence = class_probabilities[predicted_class]
            
            # Get top 3 predictions
            top_3_indices = np.argsort(class_probabilities)[::-1][:3]
            top_3_predictions = []
            
            for idx in top_3_indices:
                top_3_predictions.append({
                    'class_id': int(idx),
                    'class_name': CLASS_NAMES[idx],
                    'probability': float(class_probabilities[idx]),
                    'description': CLASS_DESCRIPTIONS[idx]
                })
            
            return {
                'predicted_class': int(predicted_class),
                'class_name': CLASS_NAMES[predicted_class],
                'confidence': float(confidence),
                'description': CLASS_DESCRIPTIONS[predicted_class],
                'all_probabilities': {CLASS_NAMES[i]: float(class_probabilities[i]) for i in range(len(CLASS_NAMES))},
                'top_3_predictions': top_3_predictions
            }
            
        except Exception as e:
            raise Exception(f"Error during prediction: {str(e)}")
    
    def get_recommendation(self, result):
        """Get detailed recommendation based on prediction"""
        class_name = result['class_name']
        confidence = result['confidence']
        
        recommendations = {
            'Acne': {
                'high': "🔴 HIGH CONFIDENCE: Significant acne detected. Consider consulting a dermatologist for professional treatment. Avoid touching or popping pimples.",
                'medium': "🟠 MODERATE: Acne detected. Use gentle cleansers, avoid harsh products, and consider over-the-counter treatments with salicylic acid or benzoyl peroxide.",
                'low': "🟡 LOW: Some acne signs detected. Monitor your skin and maintain a gentle skincare routine. Consider non-comedogenic products."
            },
            'Pimple': {
                'high': "🔴 HIGH CONFIDENCE: Pimple detected. Apply warm compress, use spot treatment, and avoid picking. Consider topical treatments.",
                'medium': "🟠 MODERATE: Pimple signs detected. Use gentle spot treatments and maintain good hygiene. Avoid touching the affected area.",
                'low': "🟡 LOW: Minor pimple signs. Continue with gentle skincare and monitor for changes."
            },
            'Spots': {
                'high': "🔴 HIGH CONFIDENCE: Dark spots detected. Consider treatments with vitamin C, niacinamide, or consult a dermatologist for professional treatments.",
                'medium': "🟠 MODERATE: Spot signs detected. Use brightening products and protect skin from sun exposure with SPF.",
                'low': "🟡 LOW: Minor spot signs. Continue with sun protection and consider brightening skincare products."
            },
            'Mole1': {
                'high': "🔴 HIGH CONFIDENCE: Flat mole detected. Monitor for changes in size, shape, or color. Consider professional evaluation if concerning.",
                'medium': "🟠 MODERATE: Mole signs detected. Regular monitoring recommended. Protect from sun exposure.",
                'low': "🟡 LOW: Minor mole signs. Continue monitoring and sun protection."
            },
            'Mole2': {
                'high': "🔴 HIGH CONFIDENCE: Raised mole detected. Professional evaluation recommended, especially if new or changing.",
                'medium': "🟠 MODERATE: Raised mole signs detected. Monitor closely and consider professional consultation.",
                'low': "🟡 LOW: Minor raised mole signs. Continue monitoring for changes."
            },
            'Scar': {
                'high': "🔴 HIGH CONFIDENCE: Scar tissue detected. Consider professional treatments like laser therapy or consult a dermatologist.",
                'medium': "🟠 MODERATE: Scar signs detected. Use scar treatment products and protect from sun exposure.",
                'low': "🟡 LOW: Minor scar signs. Continue with gentle skincare and sun protection."
            }
        }
        
        if confidence > 0.8:
            confidence_level = 'high'
        elif confidence > 0.6:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'
        
        return recommendations[class_name][confidence_level]

# Initialize the model
classifier = MultiClassSkinClassifier()

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>SkinCareAI - Multi-Class Skin Condition Detection</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
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
        .result.acne { background: linear-gradient(135deg, #ffebee, #ffcdd2); border-left: 5px solid #f44336; }
        .result.pimple { background: linear-gradient(135deg, #fff3e0, #ffe0b2); border-left: 5px solid #ff9800; }
        .result.spots { background: linear-gradient(135deg, #f3e5f5, #e1bee7); border-left: 5px solid #9c27b0; }
        .result.mole1 { background: linear-gradient(135deg, #e8f5e8, #c8e6c9); border-left: 5px solid #4caf50; }
        .result.mole2 { background: linear-gradient(135deg, #e0f2f1, #b2dfdb); border-left: 5px solid #009688; }
        .result.scar { background: linear-gradient(135deg, #fce4ec, #f8bbd9); border-left: 5px solid #e91e63; }
        
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
        .status.success { background: linear-gradient(135deg, #d4edda, #c3e6cb); color: #155724; border: 2px solid #28a745; }
        .status.error { background: linear-gradient(135deg, #f8d7da, #f5c6cb); color: #721c24; border: 2px solid #dc3545; }
        .status.info { background: linear-gradient(135deg, #d1ecf1, #bee5eb); color: #0c5460; border: 2px solid #17a2b8; }
        
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
        .probability-fill.acne { background: linear-gradient(90deg, #ff416c, #ff4b2b); }
        .probability-fill.pimple { background: linear-gradient(90deg, #ff9800, #ff5722); }
        .probability-fill.spots { background: linear-gradient(90deg, #9c27b0, #673ab7); }
        .probability-fill.mole1 { background: linear-gradient(90deg, #4caf50, #8bc34a); }
        .probability-fill.mole2 { background: linear-gradient(90deg, #009688, #00bcd4); }
        .probability-fill.scar { background: linear-gradient(90deg, #e91e63, #f06292); }
        
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
        .top-predictions {
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        .prediction-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            margin: 5px 0;
            background: white;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧬 SkinCareAI - Multi-Class Skin Condition Detection</h1>
        
        <div id="status" class="status info">
            Status: Ready to analyze skin images for 6 different conditions
        </div>
        
        <div class="upload-section">
            <h3>📸 Upload Image for Multi-Class Analysis</h3>
            <p>Detect: Acne, Pimple, Spots, Mole1, Mole2, Scar</p>
            <input type="file" id="imageInput" accept="image/*" style="display: none;">
            <button class="btn" onclick="document.getElementById('imageInput').click()">Choose Image</button>
            <button class="btn btn-secondary" onclick="testSampleImages()">Test Sample Images</button>
        </div>
        
        <div id="result" class="result"></div>
        
        <div class="chat-log" id="chatLog">
            <div class="chat-message bot">
                🤖 Welcome to SkinCareAI Multi-Class Detection! I can identify 6 different skin conditions:
                <br>• Acne: Bumpy, white-brown cysts
                <br>• Pimple: Red bumps and cysts
                <br>• Spots: Flat, dark brown circles
                <br>• Mole1: Flat, black, concentrated
                <br>• Mole2: Bumpy, dark brown
                <br>• Scar: Flat, any shape/shade
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
            
            updateStatus('🔍 Analyzing image for 6 skin conditions...', 'info');
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
            addChatMessage('🧪 Testing multi-class model on sample images...', 'user');
            
            fetch('/test-samples')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    updateStatus('✅ Sample testing completed!', 'success');
                    data.results.forEach(result => {
                        addChatMessage(`📊 Sample: ${result.class_name} (${result.confidence}% confidence)`, 'bot');
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
            const classLower = result.class_name.toLowerCase();
            const icon = getClassIcon(result.class_name);
            
            resultDiv.className = `result ${classLower}`;
            resultDiv.style.display = 'block';
            
            // Create probability bars
            let probabilityBars = '';
            Object.entries(result.all_probabilities).forEach(([className, prob]) => {
                const percent = (prob * 100).toFixed(1);
                const classLower = className.toLowerCase();
                probabilityBars += `
                    <div>
                        <div>${className}: ${percent}%</div>
                        <div class="probability-bar">
                            <div class="probability-fill ${classLower}" style="width: ${percent}%"></div>
                        </div>
                    </div>
                `;
            });
            
            // Create top 3 predictions
            let topPredictions = '';
            result.top_3_predictions.forEach((pred, index) => {
                const percent = (pred.probability * 100).toFixed(1);
                topPredictions += `
                    <div class="prediction-item">
                        <div>
                            <strong>${index + 1}. ${pred.class_name}</strong>
                            <br><small>${pred.description}</small>
                        </div>
                        <div><strong>${percent}%</strong></div>
                    </div>
                `;
            });
            
            resultDiv.innerHTML = `
                <h3>${icon} Multi-Class Analysis Results</h3>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div>Primary Detection</div>
                        <div class="stat-value">${result.class_name}</div>
                    </div>
                    <div class="stat-card">
                        <div>Confidence</div>
                        <div class="stat-value">${(result.confidence * 100).toFixed(1)}%</div>
                    </div>
                    <div class="stat-card">
                        <div>Class ID</div>
                        <div class="stat-value">${result.predicted_class}</div>
                    </div>
                </div>
                
                <h4>📊 All Class Probabilities:</h4>
                ${probabilityBars}
                
                <h4>🏆 Top 3 Predictions:</h4>
                <div class="top-predictions">
                    ${topPredictions}
                </div>
                
                <h4>💡 Recommendation:</h4>
                <p><strong>${result.recommendation}</strong></p>
            `;
        }

        function getClassIcon(className) {
            const icons = {
                'Acne': '🔴',
                'Pimple': '🟠',
                'Spots': '🟣',
                'Mole1': '🟢',
                'Mole2': '🔵',
                'Scar': '🟡'
            };
            return icons[className] || '🔍';
        }

        function formatResult(result) {
            const icon = getClassIcon(result.class_name);
            const confidence = (result.confidence * 100).toFixed(1);
            return `📊 ${icon} ${result.class_name.toUpperCase()} DETECTED (${confidence}% confidence)
                    📝 ${result.description}
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
            messageDiv.innerHTML = message;
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
        result = classifier.predict_image(image_data)
        result['recommendation'] = classifier.get_recommendation(result)
        
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/test-samples')
def test_samples():
    try:
        # Find sample images from any class directory
        sample_dir = Path(__file__).parent / 'data' / 'train'
        if not sample_dir.exists():
            return jsonify({'success': False, 'error': 'Sample dataset not found'})
        
        # Get sample images from first available class
        class_dirs = list(sample_dir.glob('*'))
        if not class_dirs:
            return jsonify({'success': False, 'error': 'No class directories found'})
        
        first_class_dir = class_dirs[0]
        image_files = list(first_class_dir.glob('*.jpg'))[:3] + list(first_class_dir.glob('*.png'))[:3]
        
        if not image_files:
            return jsonify({'success': False, 'error': 'No sample images found'})
        
        results = []
        for image_path in image_files:
            try:
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                
                result = classifier.predict_image(image_data)
                results.append({
                    'filename': image_path.name,
                    'class_name': result['class_name'],
                    'confidence': f"{result['confidence'] * 100:.1f}%"
                })
                
            except Exception as e:
                results.append({
                    'filename': image_path.name,
                    'class_name': 'ERROR',
                    'confidence': str(e)
                })
        
        return jsonify({'success': True, 'results': results})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("🚀 Starting Multi-Class SkinCareAI...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🔬 This version detects 6 different skin conditions")
    app.run(host='0.0.0.0', port=5000, debug=False)
