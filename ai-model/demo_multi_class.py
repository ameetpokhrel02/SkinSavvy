"""
Demo Multi-Class Skin Classification System
This script demonstrates the system with sample data or existing data
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
from pathlib import Path
import os
import shutil
import random

# --- CONFIG ---
IMG_SIZE = (224, 224)  # MobileNetV2 optimal size
BATCH_SIZE = 8
NUM_CLASSES = 6
EPOCHS = 5  # Short training for demo
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

# --- PATHS ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
TRAIN_DIR = DATA_DIR / 'train'
VAL_DIR = DATA_DIR / 'val'
MODEL_DIR = BASE_DIR / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("🧬 Demo Multi-Class Skin Classification System")
print("=" * 60)
print(f"Classes: {NUM_CLASSES}")
print(f"Image Size: {IMG_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")

def create_sample_data():
    """Create sample data for demonstration"""
    print("\n📁 Creating sample data structure...")
    
    # Create directories if they don't exist
    for split in ['train', 'val']:
        for class_name in CLASS_NAMES.values():
            class_dir = DATA_DIR / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy existing images to create sample data
    existing_data = BASE_DIR / 'data_organized'
    if existing_data.exists():
        print("📂 Using existing organized data...")
        
        # Copy acne images to Acne class
        acne_source = existing_data / 'train' / 'acne'
        if acne_source.exists():
            acne_files = list(acne_source.glob('*.jpg'))[:20]  # Take 20 samples
            for i, file_path in enumerate(acne_files):
                target_path = TRAIN_DIR / 'Acne' / f'acne_sample_{i}.jpg'
                shutil.copy2(file_path, target_path)
        
        # Copy no_acne images to other classes (for demo purposes)
        no_acne_source = existing_data / 'train' / 'no_acne'
        if no_acne_source.exists():
            no_acne_files = list(no_acne_source.glob('*.jpg'))
            
            # Distribute to different classes for demo
            classes = ['Pimple', 'Spots', 'Mole1', 'Mole2', 'Scar']
            for i, file_path in enumerate(no_acne_files[:25]):  # Take 25 samples
                class_name = classes[i % len(classes)]
                target_path = TRAIN_DIR / class_name / f'{class_name.lower()}_sample_{i}.jpg'
                shutil.copy2(file_path, target_path)
        
        # Create validation data (smaller subset)
        for class_name in CLASS_NAMES.values():
            train_class_dir = TRAIN_DIR / class_name
            val_class_dir = VAL_DIR / class_name
            
            if train_class_dir.exists():
                train_files = list(train_class_dir.glob('*.jpg'))
                val_files = train_files[:5]  # Take 5 for validation
                
                for file_path in val_files:
                    target_path = val_class_dir / file_path.name
                    shutil.copy2(file_path, target_path)
        
        print("✅ Sample data created successfully!")
        return True
    else:
        print("⚠️  No existing data found. Please organize your dataset first.")
        return False

def create_model():
    """Create MobileNetV2 model for multi-class classification"""
    print("\n🏗️ Building MobileNetV2 model...")
    
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
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

def train_model(model, base_model, train_gen, val_gen):
    """Train the model with two-phase approach"""
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(MODEL_DIR / 'demo_best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    
    # Phase 1: Train with frozen base model
    print("\n🔄 Phase 1: Training with frozen base model...")
    history1 = model.fit(
        train_gen,
        steps_per_epoch=max(1, train_gen.samples // BATCH_SIZE),
        validation_data=val_gen,
        validation_steps=max(1, val_gen.samples // BATCH_SIZE),
        epochs=3,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tune base model
    print("\n🔄 Phase 2: Fine-tuning base model...")
    base_model.trainable = True
    
    # Freeze early layers, train later layers
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    
    history2 = model.fit(
        train_gen,
        steps_per_epoch=max(1, train_gen.samples // BATCH_SIZE),
        validation_data=val_gen,
        validation_steps=max(1, val_gen.samples // BATCH_SIZE),
        epochs=2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Combine histories
    combined_history = {}
    for key in history1.history.keys():
        combined_history[key] = history1.history[key] + history2.history[key]
    
    return model, combined_history

def evaluate_model(model, val_gen):
    """Evaluate model performance"""
    print(f"\n📊 Evaluating model...")
    
    # Get predictions
    val_gen.reset()
    predictions = model.predict(val_gen, steps=max(1, val_gen.samples // BATCH_SIZE))
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_gen.classes[:len(predicted_classes)]
    
    # Calculate metrics
    accuracy = np.mean(predicted_classes == true_classes)
    
    print(f"\n📈 Model Results:")
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    # Per-class accuracy
    for i, class_name in enumerate(CLASS_NAMES.values()):
        class_mask = (true_classes == i)
        if np.any(class_mask):
            class_accuracy = np.mean(predicted_classes[class_mask] == true_classes[class_mask])
            print(f"  {class_name}: {class_accuracy:.4f}")
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'predicted_classes': predicted_classes,
        'true_classes': true_classes
    }

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history['accuracy'], label='Training Accuracy')
    ax1.plot(history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history['loss'], label='Training Loss')
    ax2.plot(history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(str(MODEL_DIR / 'demo_training_history.png'))
    plt.show()

def main():
    """Main demo function"""
    print("\n🚀 Starting Demo Multi-Class Skin Classification...")
    
    # Create sample data
    if not create_sample_data():
        print("❌ Failed to create sample data. Exiting.")
        return
    
    # Check if we have data
    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        print("❌ Training or validation directories not found!")
        return
    
    # Count samples
    total_train = 0
    total_val = 0
    for class_name in CLASS_NAMES.values():
        train_count = len(list((TRAIN_DIR / class_name).glob('*.jpg')))
        val_count = len(list((VAL_DIR / class_name).glob('*.jpg')))
        total_train += train_count
        total_val += val_count
        print(f"  {class_name}: {train_count} train, {val_count} val")
    
    if total_train == 0 or total_val == 0:
        print("❌ No training or validation data found!")
        return
    
    print(f"\n📊 Total samples: {total_train} train, {total_val} val")
    
    # Setup data generators
    print("\n📊 Setting up data generators...")
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
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
    
    # Create and train model
    model, base_model = create_model()
    model.summary()
    
    trained_model, history = train_model(model, base_model, train_gen, val_gen)
    
    # Evaluate model
    results = evaluate_model(trained_model, val_gen)
    
    # Save final model
    final_model_path = MODEL_DIR / 'demo_skin_classification_model.h5'
    trained_model.save(str(final_model_path))
    print(f"\n✅ Demo model saved to: {final_model_path}")
    
    # Create visualizations
    plot_training_history(history)
    
    print("\n🎉 Demo completed successfully!")
    print(f"📊 Results saved to: {MODEL_DIR}")
    print(f"🤖 Model ready for use in skin_classification.py")

if __name__ == '__main__':
    main()
