"""
Multi-Class Skin Condition Classification with EfficientNetB4
Classifies 6 different skin conditions with comprehensive evaluation
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from pathlib import Path
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
from datetime import datetime

# --- CONFIG ---
IMG_SIZE = (380, 380)  # EfficientNetB4 optimal size
BATCH_SIZE = 16  # Reduced for memory constraints
NUM_CLASSES = 6
EPOCHS = 30
LEARNING_RATE = 0.001

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

# --- DATASET PATHS ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
TRAIN_DIR = DATA_DIR / 'train'
VAL_DIR = DATA_DIR / 'val'
TEST_DIR = DATA_DIR / 'test'
MODEL_DIR = BASE_DIR / 'models'
UTILS_DIR = BASE_DIR / 'utils'

# Create directories
MODEL_DIR.mkdir(parents=True, exist_ok=True)
UTILS_DIR.mkdir(parents=True, exist_ok=True)

print("🧬 Multi-Class Skin Condition Classification")
print("=" * 60)
print(f"Classes: {NUM_CLASSES}")
print(f"Image Size: {IMG_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")

# Check if directories exist
if not TRAIN_DIR.exists():
    raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")
if not VAL_DIR.exists():
    raise FileNotFoundError(f"Validation directory not found: {VAL_DIR}")

# --- DATA AUGMENTATION ---
print("\n📊 Setting up data generators...")

# Comprehensive data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=[0.7, 1.3],
    contrast_range=[0.8, 1.2],
    fill_mode='nearest',
    channel_shift_range=20.0,
    validation_split=0.0
)

# Simple preprocessing for validation and test
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_gen = train_datagen.flow_from_directory(
    str(TRAIN_DIR),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

val_gen = val_datagen.flow_from_directory(
    str(VAL_DIR),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Print dataset information
print(f"\n📁 Dataset Information:")
print(f"Class indices: {train_gen.class_indices}")
print(f"Training samples: {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}")

# Count samples per class
class_counts = {}
for class_name in CLASS_NAMES.values():
    class_path = TRAIN_DIR / class_name
    if class_path.exists():
        count = len(list(class_path.glob('*.jpg')) + list(class_path.glob('*.png')))
        class_counts[class_name] = count
        print(f"  {class_name}: {count} samples")

# Calculate class weights to handle imbalance
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weight_dict = dict(zip(range(len(class_weights)), class_weights))
print(f"\n⚖️ Class weights: {class_weight_dict}")

# --- MODEL ARCHITECTURE ---
def create_efficientnet_model():
    """Create EfficientNetB4 based model for multi-class classification"""
    print("\n🏗️ Building EfficientNetB4 model...")
    
    # Load pre-trained EfficientNetB4
    base_model = EfficientNetB4(
        weights='imagenet',
        include_top=False,
        input_shape=IMG_SIZE + (3,),
        pooling='avg'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Create custom classification head
    model = models.Sequential([
        base_model,
        layers.Dropout(0.3),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model, base_model

# --- TRAINING FUNCTION ---
def train_model(model, base_model, train_gen, val_gen, class_weight_dict):
    """Train the model with two-phase approach"""
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')]
    )
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(MODEL_DIR / f'best_model_{timestamp}.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        TensorBoard(
            log_dir=str(MODEL_DIR / 'logs' / timestamp),
            histogram_freq=1
        )
    ]
    
    # Phase 1: Train with frozen base model
    print("\n🔄 Phase 1: Training with frozen base model...")
    history1 = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // BATCH_SIZE,
        validation_data=val_gen,
        validation_steps=val_gen.samples // BATCH_SIZE,
        epochs=20,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Phase 2: Fine-tune base model
    print("\n🔄 Phase 2: Fine-tuning base model...")
    base_model.trainable = True
    
    # Freeze early layers, train later layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')]
    )
    
    history2 = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // BATCH_SIZE,
        validation_data=val_gen,
        validation_steps=val_gen.samples // BATCH_SIZE,
        epochs=10,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Combine histories
    combined_history = {}
    for key in history1.history.keys():
        combined_history[key] = history1.history[key] + history2.history[key]
    
    return model, combined_history

# --- EVALUATION FUNCTION ---
def evaluate_model(model, val_gen, model_name="EfficientNetB4"):
    """Comprehensive model evaluation"""
    print(f"\n📊 Evaluating {model_name}...")
    
    # Get predictions
    val_gen.reset()
    predictions = model.predict(val_gen, steps=val_gen.samples // BATCH_SIZE + 1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_gen.classes[:len(predicted_classes)]
    
    # Calculate metrics
    accuracy = np.mean(predicted_classes == true_classes)
    
    # Per-class metrics
    class_report = classification_report(
        true_classes, 
        predicted_classes, 
        target_names=list(CLASS_NAMES.values()),
        output_dict=True
    )
    
    # ROC AUC for each class
    roc_auc_scores = {}
    for i, class_name in enumerate(CLASS_NAMES.values()):
        if len(np.unique(true_classes)) > 1:
            roc_auc_scores[class_name] = roc_auc_score(
                (true_classes == i).astype(int), 
                predictions[:, i]
            )
    
    # Overall ROC AUC
    overall_roc_auc = roc_auc_score(
        to_categorical(true_classes, num_classes=NUM_CLASSES),
        predictions,
        multi_class='ovr'
    )
    
    print(f"\n📈 {model_name} Results:")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Overall ROC AUC: {overall_roc_auc:.4f}")
    
    # Print per-class metrics
    print(f"\n📊 Per-Class Performance:")
    for class_name in CLASS_NAMES.values():
        if class_name in class_report:
            precision = class_report[class_name]['precision']
            recall = class_report[class_name]['recall']
            f1 = class_report[class_name]['f1-score']
            auc = roc_auc_scores.get(class_name, 0)
            print(f"  {class_name}:")
            print(f"    Precision: {precision:.4f}")
            print(f"    Recall: {recall:.4f}")
            print(f"    F1-Score: {f1:.4f}")
            print(f"    ROC AUC: {auc:.4f}")
    
    return {
        'accuracy': accuracy,
        'overall_roc_auc': overall_roc_auc,
        'class_report': class_report,
        'roc_auc_scores': roc_auc_scores,
        'predictions': predictions,
        'predicted_classes': predicted_classes,
        'true_classes': true_classes
    }

# --- VISUALIZATION FUNCTIONS ---
def plot_training_history(history, model_name):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot accuracy
    axes[0, 0].plot(history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title(f'{model_name} - Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    
    # Plot loss
    axes[0, 1].plot(history['loss'], label='Training Loss')
    axes[0, 1].plot(history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title(f'{model_name} - Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    
    # Plot precision
    axes[1, 0].plot(history['precision'], label='Training Precision')
    axes[1, 0].plot(history['val_precision'], label='Validation Precision')
    axes[1, 0].set_title(f'{model_name} - Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    
    # Plot recall
    axes[1, 1].plot(history['recall'], label='Training Recall')
    axes[1, 1].plot(history['val_recall'], label='Validation Recall')
    axes[1, 1].set_title(f'{model_name} - Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(str(MODEL_DIR / f'{model_name.lower()}_training_history.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(true_classes, predicted_classes, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(true_classes, predicted_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(CLASS_NAMES.values()),
                yticklabels=list(CLASS_NAMES.values()))
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(str(MODEL_DIR / f'{model_name.lower()}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_class_performance(class_report, roc_auc_scores, model_name):
    """Plot per-class performance metrics"""
    classes = list(CLASS_NAMES.values())
    precision = [class_report[cls]['precision'] for cls in classes]
    recall = [class_report[cls]['recall'] for cls in classes]
    f1_scores = [class_report[cls]['f1-score'] for cls in classes]
    auc_scores = [roc_auc_scores.get(cls, 0) for cls in classes]
    
    x = np.arange(len(classes))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.bar(x - width*1.5, precision, width, label='Precision', alpha=0.8)
    ax.bar(x - width*0.5, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width*0.5, f1_scores, width, label='F1-Score', alpha=0.8)
    ax.bar(x + width*1.5, auc_scores, width, label='ROC AUC', alpha=0.8)
    
    ax.set_xlabel('Classes')
    ax.set_ylabel('Score')
    ax.set_title(f'{model_name} - Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(MODEL_DIR / f'{model_name.lower()}_class_performance.png'), dpi=300, bbox_inches='tight')
    plt.show()

# --- MAIN TRAINING ---
def main():
    """Main training function"""
    print("\n🚀 Starting Multi-Class Skin Condition Classification Training...")
    
    # Create model
    model, base_model = create_efficientnet_model()
    model.summary()
    
    # Train model
    trained_model, history = train_model(
        model, base_model, train_gen, val_gen, class_weight_dict
    )
    
    # Evaluate model
    results = evaluate_model(trained_model, val_gen)
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    final_model_path = MODEL_DIR / f'skin_classification_model_{timestamp}.h5'
    trained_model.save(str(final_model_path))
    print(f"\n✅ Model saved to: {final_model_path}")
    
    # Create visualizations
    plot_training_history(history, "EfficientNetB4")
    plot_confusion_matrix(
        results['true_classes'], 
        results['predicted_classes'], 
        "EfficientNetB4"
    )
    plot_class_performance(
        results['class_report'], 
        results['roc_auc_scores'], 
        "EfficientNetB4"
    )
    
    # Save evaluation report
    save_evaluation_report(results, timestamp)
    
    print("\n🎉 Training completed successfully!")
    print(f"📊 Results saved to: {MODEL_DIR}")

def save_evaluation_report(results, timestamp):
    """Save detailed evaluation report"""
    report_path = MODEL_DIR / f'evaluation_report_{timestamp}.txt'
    
    with open(report_path, 'w') as f:
        f.write("MULTI-CLASS SKIN CONDITION CLASSIFICATION - EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("PROJECT OVERVIEW:\n")
        f.write("-" * 20 + "\n")
        f.write("Model: EfficientNetB4\n")
        f.write("Classes: 6 skin conditions\n")
        f.write("Image Size: 380x380\n")
        f.write("Training Strategy: Two-phase (frozen + fine-tuned)\n\n")
        
        f.write("CLASS DEFINITIONS:\n")
        f.write("-" * 20 + "\n")
        for class_id, (name, desc) in enumerate(zip(CLASS_NAMES.values(), CLASS_DESCRIPTIONS.values())):
            f.write(f"{class_id}. {name}: {desc}\n")
        f.write("\n")
        
        f.write("DATASET INFORMATION:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Training samples: {train_gen.samples}\n")
        f.write(f"Validation samples: {val_gen.samples}\n")
        f.write(f"Class weights: {class_weight_dict}\n\n")
        
        f.write("OVERALL PERFORMANCE:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Overall ROC AUC: {results['overall_roc_auc']:.4f}\n\n")
        
        f.write("PER-CLASS PERFORMANCE:\n")
        f.write("-" * 20 + "\n")
        for class_name in CLASS_NAMES.values():
            if class_name in results['class_report']:
                report = results['class_report'][class_name]
                f.write(f"\n{class_name}:\n")
                f.write(f"  Precision: {report['precision']:.4f}\n")
                f.write(f"  Recall: {report['recall']:.4f}\n")
                f.write(f"  F1-Score: {report['f1-score']:.4f}\n")
                f.write(f"  Support: {report['support']}\n")
        
        f.write(f"\n\nReport generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"📄 Evaluation report saved to: {report_path}")

if __name__ == '__main__':
    main()
