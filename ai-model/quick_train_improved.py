"""
Quick Training Script for Improved Skin Condition Model
Focuses on the best performing architecture for faster training
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

# --- CONFIG ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 2
EPOCHS = 20
LEARNING_RATE = 0.001

# --- PATHS ---
BASE_DIR = Path(__file__).parent
TRAIN_DIR = BASE_DIR / 'data_organized' / 'train'
VAL_DIR = BASE_DIR / 'data_organized' / 'val'
MODEL_DIR = BASE_DIR / 'saved_models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("🚀 Quick Training - Improved Skin Condition Model")
print(f"Training data: {TRAIN_DIR}")
print(f"Validation data: {VAL_DIR}")

# --- DATA GENERATORS ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

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

print(f"Class indices: {train_gen.class_indices}")
print(f"Training samples: {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}")

# --- CLASS WEIGHTS ---
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weight_dict = dict(zip(range(len(class_weights)), class_weights))
print(f"Class weights: {class_weight_dict}")

# --- MODEL ---
print("Building EfficientNetB0 model...")

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

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

model.summary()

# --- CALLBACKS ---
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=6,
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
        filepath=str(MODEL_DIR / 'best_quick_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
]

# --- TRAINING ---
print("Starting training...")

# Phase 1: Train with frozen base model
print("Phase 1: Training with frozen base model...")
history1 = model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // BATCH_SIZE,
    validation_data=val_gen,
    validation_steps=val_gen.samples // BATCH_SIZE,
    epochs=12,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# Phase 2: Fine-tune base model
print("Phase 2: Fine-tuning base model...")
base_model.trainable = True

# Freeze early layers, train later layers
for layer in base_model.layers[:-15]:
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
    epochs=8,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# --- EVALUATION ---
print("Evaluating model...")
val_gen.reset()
predictions = model.predict(val_gen, steps=val_gen.samples // BATCH_SIZE + 1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = val_gen.classes[:len(predicted_classes)]

accuracy = np.mean(predicted_classes == true_classes)
precision = tf.keras.metrics.Precision()(true_classes, predicted_classes).numpy()
recall = tf.keras.metrics.Recall()(true_classes, predicted_classes).numpy()
f1_score = 2 * (precision * recall) / (precision + recall)
roc_auc = roc_auc_score(true_classes, predictions[:, 1])

print(f"\n📊 Final Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# --- SAVE MODEL ---
final_model_path = MODEL_DIR / 'skin_condition_model.h5'
model.save(str(final_model_path))
print(f"\n✅ Model saved to: {final_model_path}")

# --- PLOT HISTORY ---
def plot_history(history1, history2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Combine histories
    combined_acc = history1.history['accuracy'] + history2.history['accuracy']
    combined_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    combined_loss = history1.history['loss'] + history2.history['loss']
    combined_val_loss = history1.history['val_loss'] + history2.history['val_loss']
    
    # Plot accuracy
    ax1.plot(combined_acc, label='Training Accuracy')
    ax1.plot(combined_val_acc, label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(combined_loss, label='Training Loss')
    ax2.plot(combined_val_loss, label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(str(MODEL_DIR / 'quick_training_history.png'))
    plt.show()

plot_history(history1, history2)

print("🎉 Quick training completed successfully!")
print(f"📈 Training history saved to: {MODEL_DIR / 'quick_training_history.png'}")
print(f"🤖 Model ready for use in improved_web_chatbot.py")
