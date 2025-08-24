"""
Improved Skin Condition Detection Model - Acne Classifier
Addresses class imbalance and implements better model selection
Uses multiple architectures and advanced techniques for better performance
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet50V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from pathlib import Path
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

# --- CONFIG ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 2  # Binary classification with categorical encoding
EPOCHS = 25
LEARNING_RATE = 0.001

# --- DATASET PATHS ---
BASE_DIR = Path(__file__).parent
TRAIN_DIR = BASE_DIR / 'data_organized' / 'train'
VAL_DIR = BASE_DIR / 'data_organized' / 'val'
MODEL_DIR = BASE_DIR / 'saved_models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(f"Training data directory: {TRAIN_DIR}")
print(f"Validation data directory: {VAL_DIR}")
print(f"Model save directory: {MODEL_DIR}")

# Check if directories exist
if not TRAIN_DIR.exists():
    raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")
if not VAL_DIR.exists():
    raise FileNotFoundError(f"Validation directory not found: {VAL_DIR}")

# --- DATA AUGMENTATION ---
print("Setting up data generators...")

# Enhanced data augmentation for better generalization
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
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators with categorical mode
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

# Print class information
print(f"Class indices: {train_gen.class_indices}")
print(f"Training samples: {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}")

# Calculate class weights to address imbalance
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weight_dict = dict(zip(range(len(class_weights)), class_weights))
print(f"Class weights: {class_weight_dict}")

# --- MODEL ARCHITECTURES ---
def create_mobilenet_model():
    """Create MobileNetV2 based model"""
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=IMG_SIZE + (3,),
        pooling='avg'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model, base_model

def create_efficientnet_model():
    """Create EfficientNetB0 based model"""
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=IMG_SIZE + (3,),
        pooling='avg'
    )
    
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model, base_model

def create_resnet_model():
    """Create ResNet50V2 based model"""
    base_model = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=IMG_SIZE + (3,),
        pooling='avg'
    )
    
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model, base_model

# --- TRAINING FUNCTION ---
def train_model(model_name, model, base_model, train_gen, val_gen, class_weight_dict):
    """Train a model with fine-tuning"""
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(MODEL_DIR / f'best_{model_name.lower()}.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    
    # Phase 1: Train with frozen base model
    print("Phase 1: Training with frozen base model...")
    history1 = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // BATCH_SIZE,
        validation_data=val_gen,
        validation_steps=val_gen.samples // BATCH_SIZE,
        epochs=15,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Phase 2: Fine-tune base model
    print("Phase 2: Fine-tuning base model...")
    base_model.trainable = True
    
    # Freeze early layers, train later layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
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
def evaluate_model(model, val_gen, model_name):
    """Evaluate model performance"""
    print(f"\nEvaluating {model_name}...")
    
    # Get predictions
    val_gen.reset()
    predictions = model.predict(val_gen, steps=val_gen.samples // BATCH_SIZE + 1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_gen.classes[:len(predicted_classes)]
    
    # Calculate metrics
    accuracy = np.mean(predicted_classes == true_classes)
    precision = tf.keras.metrics.Precision()(true_classes, predicted_classes).numpy()
    recall = tf.keras.metrics.Recall()(true_classes, predicted_classes).numpy()
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    # ROC AUC for binary classification
    if NUM_CLASSES == 2:
        roc_auc = roc_auc_score(true_classes, predictions[:, 1])
    else:
        roc_auc = roc_auc_score(true_classes, predictions, multi_class='ovr')
    
    print(f"{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'roc_auc': roc_auc,
        'predictions': predictions,
        'predicted_classes': predicted_classes,
        'true_classes': true_classes
    }

# --- MAIN TRAINING ---
def main():
    # Define models to train
    models_to_train = [
        ('MobileNetV2', create_mobilenet_model()),
        ('EfficientNetB0', create_efficientnet_model()),
        ('ResNet50V2', create_resnet_model())
    ]
    
    results = {}
    
    for model_name, (model, base_model) in models_to_train:
        try:
            # Train model
            trained_model, history = train_model(
                model_name, model, base_model, train_gen, val_gen, class_weight_dict
            )
            
            # Evaluate model
            result = evaluate_model(trained_model, val_gen, model_name)
            results[model_name] = result
            
            # Save model
            model_path = MODEL_DIR / f'{model_name.lower()}_final.h5'
            trained_model.save(str(model_path))
            print(f"Model saved to: {model_path}")
            
            # Plot training history
            plot_training_history(history, model_name)
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            continue
    
    # Select best model
    if results:
        best_model = select_best_model(results)
        print(f"\n{'='*50}")
        print(f"BEST MODEL: {best_model}")
        print(f"{'='*50}")
        
        # Save best model as the main model
        best_model_path = MODEL_DIR / f'{best_model.lower()}_final.h5'
        main_model_path = MODEL_DIR / 'skin_condition_model.h5'
        shutil.copy(str(best_model_path), str(main_model_path))
        print(f"Best model copied to: {main_model_path}")
        
        # Create detailed evaluation report
        create_evaluation_report(results, best_model)

def select_best_model(results):
    """Select the best model based on F1-score and ROC AUC"""
    best_score = 0
    best_model = None
    
    for model_name, result in results.items():
        # Combined score (F1 + ROC AUC)
        combined_score = result['f1_score'] + result['roc_auc']
        
        if combined_score > best_score:
            best_score = combined_score
            best_model = model_name
    
    return best_model

def plot_training_history(history, model_name):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history['accuracy'], label='Training Accuracy')
    ax1.plot(history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title(f'{model_name} - Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history['loss'], label='Training Loss')
    ax2.plot(history['val_loss'], label='Validation Loss')
    ax2.set_title(f'{model_name} - Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(str(MODEL_DIR / f'{model_name.lower()}_training_history.png'))
    plt.close()

def create_evaluation_report(results, best_model):
    """Create detailed evaluation report"""
    report_path = MODEL_DIR / 'evaluation_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("SKIN CONDITION DETECTION - MODEL EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("DATASET INFORMATION:\n")
        f.write(f"Training samples: {train_gen.samples}\n")
        f.write(f"Validation samples: {val_gen.samples}\n")
        f.write(f"Class weights: {class_weight_dict}\n\n")
        
        f.write("MODEL COMPARISON:\n")
        f.write("-" * 40 + "\n")
        
        for model_name, result in results.items():
            f.write(f"\n{model_name}:\n")
            f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"  Precision: {result['precision']:.4f}\n")
            f.write(f"  Recall: {result['recall']:.4f}\n")
            f.write(f"  F1-Score: {result['f1_score']:.4f}\n")
            f.write(f"  ROC AUC: {result['roc_auc']:.4f}\n")
        
        f.write(f"\nBEST MODEL: {best_model}\n")
        f.write(f"Best model saved as: skin_condition_model.h5\n")
    
    print(f"Evaluation report saved to: {report_path}")

if __name__ == '__main__':
    print("🚀 Starting Improved Skin Condition Model Training...")
    print("📊 This will train multiple models and select the best one")
    main()
    print("✅ Training completed successfully!")
