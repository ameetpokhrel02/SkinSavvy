# 🧬 Multi-Class Skin Condition Classification

## 📋 Project Overview

This project implements a comprehensive multi-class skin condition classification system using **EfficientNetB4** to identify 6 different skin conditions with high accuracy and detailed analysis.

## 🎯 Skin Conditions Detected

| Class ID | Condition | Description |
|----------|-----------|-------------|
| 0 | **Acne** | Bumpy, small-big, white-brown cysts with no redness |
| 1 | **Pimple** | Bumpy, small-big, red-white cysts, any red big bump (must be red) |
| 2 | **Spots** | Flat, small, circular, dark brown |
| 3 | **Mole1** | Flat/black/small/concentrated |
| 4 | **Mole2** | Bumpy/black/dark-brown |
| 5 | **Scar** | Flat/any shape/any shade |

## 🏗️ Project Structure

```
ai-model/
├── data/
│   ├── train/
│   │   ├── Acne/
│   │   ├── Pimple/
│   │   ├── Spots/
│   │   ├── Mole1/
│   │   ├── Mole2/
│   │   └── Scar/
│   ├── val/
│   │   ├── Acne/
│   │   ├── Pimple/
│   │   ├── Spots/
│   │   ├── Mole1/
│   │   ├── Mole2/
│   │   └── Scar/
│   └── test/
│       ├── Acne/
│       ├── Pimple/
│       ├── Spots/
│       ├── Mole1/
│       ├── Mole2/
│       └── Scar/
├── models/
│   ├── skin_classification_model_*.h5
│   ├── evaluation_report_*.txt
│   └── logs/
├── utils/
├── multi_class_skin_classifier.py
├── skin_classification.py
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

Organize your images in the following structure:
```
data/
├── train/
│   ├── Acne/          # Training images for acne
│   ├── Pimple/        # Training images for pimples
│   ├── Spots/         # Training images for spots
│   ├── Mole1/         # Training images for flat moles
│   ├── Mole2/         # Training images for raised moles
│   └── Scar/          # Training images for scars
├── val/
│   ├── Acne/          # Validation images for acne
│   ├── Pimple/        # Validation images for pimples
│   ├── Spots/         # Validation images for spots
│   ├── Mole1/         # Validation images for flat moles
│   ├── Mole2/         # Validation images for raised moles
│   └── Scar/          # Validation images for scars
└── test/
    ├── Acne/          # Test images for acne
    ├── Pimple/        # Test images for pimples
    ├── Spots/         # Test images for spots
    ├── Mole1/         # Test images for flat moles
    ├── Mole2/         # Test images for raised moles
    └── Scar/          # Test images for scars
```

### 3. Train the Model

```bash
# Start training
python multi_class_skin_classifier.py
```

### 4. Run Web Interface

```bash
# Start the web application
python skin_classification.py

# Open browser to http://localhost:5000
```

## 🔧 Technical Specifications

### Model Architecture
- **Base Model**: EfficientNetB4 (pre-trained on ImageNet)
- **Input Size**: 380×380×3 (optimal for EfficientNetB4)
- **Output**: 6-class softmax probabilities
- **Training Strategy**: Two-phase transfer learning

### Training Configuration
- **Batch Size**: 16 (memory-optimized)
- **Learning Rate**: 0.001 (Phase 1), 0.0001 (Phase 2)
- **Epochs**: 20 (Phase 1) + 10 (Phase 2)
- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Categorical Crossentropy
- **Class Weights**: Automatically calculated for imbalance handling

### Data Augmentation
```python
# Comprehensive augmentation for training
rotation_range=30,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
brightness_range=[0.7, 1.3],
contrast_range=[0.8, 1.2],
channel_shift_range=20.0
```

## 📊 Evaluation Metrics

The system provides comprehensive evaluation including:

- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class precision scores
- **Recall**: Per-class recall scores
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under ROC curve for each class
- **Confusion Matrix**: Visual representation of predictions
- **Training History**: Accuracy, loss, precision, recall plots

## 🎨 Web Interface Features

### Advanced UI
- **Modern Design**: Gradient backgrounds and smooth animations
- **Real-time Analysis**: Instant image processing and results
- **Visual Probability Bars**: Color-coded probability visualization
- **Top 3 Predictions**: Shows confidence for all classes
- **Detailed Recommendations**: Condition-specific skincare advice

### Result Display
- **Primary Detection**: Main predicted condition with confidence
- **All Probabilities**: Complete breakdown for all 6 classes
- **Top 3 Rankings**: Best matches with descriptions
- **Personalized Recommendations**: Based on confidence levels

## 🔍 Model Performance

### Expected Results
- **Overall Accuracy**: >85%
- **Per-Class F1-Score**: >0.80 for each class
- **ROC AUC**: >0.90 for most classes
- **Balanced Predictions**: No bias towards specific classes

### Performance Optimization
- **GPU Acceleration**: CUDA support for faster training
- **Memory Management**: Optimized batch sizes and model loading
- **Early Stopping**: Prevents overfitting with patience=10
- **Model Checkpointing**: Saves best model during training

## 🛠️ Advanced Features

### Two-Phase Training
1. **Phase 1**: Train classification head with frozen base model
2. **Phase 2**: Fine-tune last 30 layers of EfficientNetB4

### Class Imbalance Handling
- **Automatic Class Weights**: Calculated based on dataset distribution
- **Balanced Sampling**: Ensures fair representation of all classes
- **Augmentation**: Enhanced data augmentation for minority classes

### Comprehensive Evaluation
- **Multiple Metrics**: Accuracy, precision, recall, F1-score, ROC AUC
- **Visualization**: Training curves, confusion matrix, per-class performance
- **Detailed Reports**: Text-based evaluation reports with timestamps

## 📈 Usage Examples

### Training
```bash
# Start comprehensive training
python multi_class_skin_classifier.py

# Monitor training progress
tensorboard --logdir models/logs/
```

### Inference
```python
from skin_classification import MultiClassSkinClassifier

# Load model
classifier = MultiClassSkinClassifier()

# Predict single image
with open('test_image.jpg', 'rb') as f:
    result = classifier.predict_image(f.read())
    print(f"Predicted: {result['class_name']}")
    print(f"Confidence: {result['confidence']:.2%}")
```

### Web Interface
```bash
# Start web server
python skin_classification.py

# Access interface
# Open http://localhost:5000 in browser
```

## 🔧 Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size to 8
2. **Training Too Slow**: Use GPU acceleration
3. **Poor Results**: Check dataset quality and balance
4. **Model Not Found**: Ensure training completed successfully

### Performance Tips
- **GPU Training**: Use CUDA-enabled GPU for 5-10x speedup
- **Data Quality**: Ensure images are properly labeled and high quality
- **Memory Management**: Monitor GPU memory usage during training
- **Regular Evaluation**: Check validation metrics during training

## 📝 Dependencies

### Core Requirements
- **TensorFlow**: 2.10.0+ (with GPU support recommended)
- **NumPy**: 1.21.0+
- **Pillow**: 9.0.0+ (image processing)
- **Matplotlib**: 3.5.0+ (visualization)
- **Seaborn**: 0.11.0+ (advanced plotting)
- **Scikit-learn**: 1.1.0+ (metrics and utilities)
- **Pandas**: 1.4.0+ (data handling)
- **Flask**: 2.2.0+ (web interface)

### Optional Dependencies
- **TensorBoard**: For training visualization
- **OpenCV**: For additional image processing
- **Albumentations**: For advanced data augmentation

## 🎉 Expected Outcomes

After successful training and deployment, you'll have:

1. **High-Accuracy Model**: >85% accuracy across all 6 skin conditions
2. **Balanced Detection**: Fair performance across all classes
3. **User-Friendly Interface**: Modern web UI with detailed results
4. **Comprehensive Analysis**: Multiple metrics and visualizations
5. **Production Ready**: Optimized for real-world deployment

## 📄 License

This project is designed for educational and research purposes. Please ensure proper medical validation before clinical use.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve the system.

---

**Note**: This system is designed for educational and research purposes. For medical applications, please consult with healthcare professionals and ensure proper validation.
