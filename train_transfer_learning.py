"""
Transfer Learning: Fine-tune ImageNet models for Cat vs Dog classification
Uses pre-trained MobileNetV2 and fine-tunes on cat/dog dataset
"""

import os
import numpy as np
import cv2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Configuration
DATA_DIR = 'data'
MODELS_DIR = 'models'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001

os.makedirs(MODELS_DIR, exist_ok=True)

def load_dataset():
    """Load images from data/cats and data/dogs directories"""
    print("=" * 60)
    print("📂 Loading Cat vs Dog Dataset")
    print("=" * 60)
    
    images = []
    labels = []
    
    categories = {'cats': 0, 'dogs': 1}
    
    for category, label in categories.items():
        category_path = os.path.join(DATA_DIR, category)
        
        if not os.path.exists(category_path):
            print(f"⚠️  {category_path} not found!")
            continue
        
        image_files = [f for f in os.listdir(category_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
        
        print(f"  Loading {len(image_files)} {category} images...")
        
        for img_name in image_files:
            img_path = os.path.join(category_path, img_name)
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, IMG_SIZE)
                    images.append(img)
                    labels.append(label)
            except Exception as e:
                print(f"  ⚠️  Error loading {img_path}: {e}")
    
    if len(images) == 0:
        print("\n❌ No images found!")
        print("\nPlease organize your data:")
        print("data/")
        print("├── cats/")
        print("│   ├── cat1.jpg")
        print("│   └── ...")
        print("└── dogs/")
        print("    ├── dog1.jpg")
        print("    └── ...")
        return None, None
    
    print(f"\n✅ Loaded {len(images)} total images")
    print(f"   Cats: {np.sum(np.array(labels) == 0)}")
    print(f"   Dogs: {np.sum(np.array(labels) == 1)}\n")
    
    return np.array(images), np.array(labels)

def create_transfer_learning_model():
    """Create model using pre-trained MobileNetV2"""
    print("🏗️  Building Transfer Learning Model")
    print("-" * 60)
    
    # Load pre-trained MobileNetV2 (trained on ImageNet)
    print("  📦 Loading MobileNetV2 with ImageNet weights...")
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,  # Exclude final classification layer
        weights='imagenet'   # Use ImageNet pre-trained weights
    )
    
    # Freeze base model layers (use ImageNet features)
    print("  🔒 Freezing base model layers...")
    base_model.trainable = False
    
    # Add custom classification head
    print("  ➕ Adding custom classification layers...")
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Binary classification: cat vs dog
    ])
    
    # Compile model
    print("  ⚙️  Compiling model...")
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Model created successfully!\n")
    print(model.summary())
    
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    """Train the model with data augmentation"""
    print("\n" + "=" * 60)
    print("🚀 Training Model")
    print("=" * 60)
    
    # Data augmentation for better generalization
    print("  📊 Setting up data augmentation...")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Prepare generators
    train_generator = train_datagen.flow(
        X_train, y_train,
        batch_size=BATCH_SIZE
    )
    
    val_generator = val_datagen.flow(
        X_val, y_val,
        batch_size=BATCH_SIZE
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(MODELS_DIR, 'catdog_transfer_learning.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    print(f"\n  🎯 Training for up to {EPOCHS} epochs...")
    print(f"  📊 Training samples: {len(X_train)}")
    print(f"  📊 Validation samples: {len(X_val)}\n")
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def fine_tune_model(model, X_train, y_train, X_val, y_val, history):
    """Fine-tune by unfreezing some layers"""
    print("\n" + "=" * 60)
    print("🔓 Fine-tuning Model (Unfreezing layers)")
    print("=" * 60)
    
    # Unfreeze the last few layers of base model
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Freeze all layers except the last 20
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    print(f"  ✅ Unfroze last 20 layers for fine-tuning")
    
    # Re-compile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Data generators (same as before)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)
    
    # Fine-tune
    print(f"\n  🎯 Fine-tuning for {EPOCHS // 2} more epochs...\n")
    
    history_fine = model.fit(
        train_generator,
        epochs=EPOCHS // 2,
        validation_data=val_generator,
        verbose=1
    )
    
    # Save final model
    model.save(os.path.join(MODELS_DIR, 'catdog_transfer_learning.h5'))
    print(f"\n  💾 Model saved to: {MODELS_DIR}/catdog_transfer_learning.h5")
    
    return history_fine

def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set"""
    print("\n" + "=" * 60)
    print("📊 Evaluating Model")
    print("=" * 60)
    
    # Normalize test data
    X_test_normalized = X_test / 255.0
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test_normalized, y_test, verbose=0)
    
    print(f"\n  Test Loss: {loss:.4f}")
    print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Sample predictions
    predictions = model.predict(X_test_normalized[:5], verbose=0)
    print(f"\n  Sample Predictions:")
    for i, (pred, true_label) in enumerate(zip(predictions, y_test[:5])):
        pred_label = 'Dog' if pred[0] > 0.5 else 'Cat'
        true_label_str = 'Dog' if true_label == 1 else 'Cat'
        confidence = pred[0] if pred[0] > 0.5 else 1 - pred[0]
        print(f"    Image {i+1}: Predicted: {pred_label} ({confidence*100:.1f}%), Actual: {true_label_str}")
    
    return accuracy

def main():
    print("\n" + "=" * 60)
    print("🐱🐶 Transfer Learning: Cat vs Dog Classifier")
    print("    Using ImageNet Pre-trained MobileNetV2")
    print("=" * 60 + "\n")
    
    # Load dataset
    images, labels = load_dataset()
    
    if images is None:
        return
    
    # Split data: 70% train, 15% validation, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=0.15, random_state=42, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )  # 0.176 of 85% ≈ 15% of total
    
    print("📊 Dataset Split:")
    print(f"   Training: {len(X_train)} images ({len(X_train)/len(images)*100:.1f}%)")
    print(f"   Validation: {len(X_val)} images ({len(X_val)/len(images)*100:.1f}%)")
    print(f"   Testing: {len(X_test)} images ({len(X_test)/len(images)*100:.1f}%)\n")
    
    # Create model
    model = create_transfer_learning_model()
    
    # Train model
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Fine-tune model
    history_fine = fine_tune_model(model, X_train, y_train, X_val, y_val, history)
    
    # Evaluate
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Summary
    print("\n" + "=" * 60)
    print("✨ Training Complete!")
    print("=" * 60)
    print(f"\n📈 Final Results:")
    print(f"   Test Accuracy: {accuracy*100:.2f}%")
    print(f"\n💾 Model saved to: {MODELS_DIR}/catdog_transfer_learning.h5")
    print(f"\n🚀 To use this model:")
    print(f"   1. Restart the Flask server: python app.py")
    print(f"   2. Select 'Transfer Learning' in the dropdown")
    print(f"   3. Upload a cat or dog image")
    print("=" * 60 + "\n")

if __name__ == '__main__':
    main()