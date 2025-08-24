"""
Skin Condition Detection Model - Acne Classifier
Uses transfer learning with MobileNetV2 (TensorFlow/Keras)
Enhanced with better monitoring, callbacks, and model validation
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from pathlib import Path

# --- CONFIG ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 16  # Reduced for memory constraints
NUM_CLASSES = 1  # Binary classification: acne vs no_acne
EPOCHS = 15
LEARNING_RATE = 0.001

# --- DATASET PATHS ---
# Use the organized dataset paths
BASE_DIR = Path(__file__).parent
TRAIN_DIR = BASE_DIR / 'data_organized' / 'train'
VAL_DIR = BASE_DIR / 'data_organized' / 'val'

# Create model directory if it doesn't exist
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

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_gen = train_datagen.flow_from_directory(
    str(TRAIN_DIR),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    seed=42
)

val_gen = val_datagen.flow_from_directory(
    str(VAL_DIR),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Print class indices and sample counts
print(f"Class indices: {train_gen.class_indices}")
print(f"Training samples: {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}")

# --- MODEL ARCHITECTURE ---
print("Building model...")

# Load pre-trained MobileNetV2
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=IMG_SIZE + (3,),
    pooling='avg'
)

# Freeze base model layers
base_model.trainable = False

# Create custom head
model = models.Sequential([
    base_model,
    layers.Dropout(0.2),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Display model architecture
model.summary()

# --- CALLBACKS ---
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=str(MODEL_DIR / 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
]

# --- TRAINING ---
print("Starting training...")
print(f"Steps per epoch: {train_gen.samples // BATCH_SIZE}")
print(f"Validation steps: {val_gen.samples // BATCH_SIZE}")

history = model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // BATCH_SIZE,
    validation_data=val_gen,
    validation_steps=val_gen.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# --- EVALUATION ---
print("Evaluating model on validation set...")
val_loss, val_accuracy, val_precision, val_recall = model.evaluate(val_gen)
print(f"\nValidation Results:")
print(f"Accuracy: {val_accuracy:.4f}")
print(f"Precision: {val_precision:.4f}")
print(f"Recall: {val_recall:.4f}")
print(f"F1-Score: {2 * (val_precision * val_recall) / (val_precision + val_recall):.4f}")

# --- SAVE FINAL MODEL ---
final_model_path = MODEL_DIR / 'skin_condition_model.h5'
model.save(str(final_model_path))
print(f"\nModel saved to: {final_model_path}")

# --- PLOT TRAINING HISTORY ---
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(str(MODEL_DIR / 'training_history.png'))
    plt.show()

plot_training_history(history)


print("Training completed successfully!")
print(f"Model saved to: {final_model_path}")
print(f"Training history plot saved to: {MODEL_DIR / 'training_history.png'}")