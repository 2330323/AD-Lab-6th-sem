import os
import numpy as np
import cv2
import joblib
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = 'data'
MODELS_DIR = 'models'
IMG_SIZE = (224, 224)

os.makedirs(MODELS_DIR, exist_ok=True)

def load_dataset():
    """Load images and labels"""
    print("=" * 60)
    print("📂 Loading Dataset for Hybrid Random Forest")
    print("=" * 60)
    images = []
    labels = []
    categories = {'cats': 0, 'dogs': 1}
    
    for category, label in categories.items():
        path = os.path.join(DATA_DIR, category)
        if not os.path.exists(path):
            print(f"⚠️  {path} not found!")
            continue
            
        files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.gif'))]
        print(f"  Loading {len(files)} {category} images...")
        
        for f in files:
            try:
                img = cv2.imread(os.path.join(path, f))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, IMG_SIZE)
                    images.append(img)
                    labels.append(label)
            except Exception as e:
                print(f"  ⚠️  Error loading {f}: {e}")
    
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

def extract_features(images):
    """
    Use MobileNetV2 to extract 'Deep Features' instead of raw pixels.
    This converts an image into a list of 1280 meaningful numbers describing shapes/textures.
    """
    print("🧠 Extracting Deep Features using MobileNetV2...")
    print("-" * 60)
    print("  📦 Loading MobileNetV2 feature extractor...")
    model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=(224, 224, 3)
    )
    print("  🔧 Preprocessing images...")
    processed_images = preprocess_input(images.astype(np.float32))
    print("  ⚙️  Extracting features (this may take a moment)...")
    features = model.predict(processed_images, batch_size=32, verbose=1)
    
    print(f"  ✅ Features extracted! Shape: {features.shape}")
    print(f"     Each image → {features.shape[1]} feature values\n")
    
    return features

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Cat', 'Dog'], 
                yticklabels=['Cat', 'Dog'])
    plt.title('Random Forest Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"  📊 Confusion matrix saved to: {save_path}")
    plt.close()

def main():
    print("\n" + "=" * 60)
    print("🌲 High-Accuracy Random Forest Classifier")
    print("   Deep Learning Features + Random Forest Logic")
    print("=" * 60 + "\n")
    
    # 1. Load Data
    X_images, y = load_dataset()
    if X_images is None:
        return

    X_features = extract_features(X_images)

    print("📊 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}\n")

    print("=" * 60)
    print("🌲 Training Random Forest on Deep Features")
    print("=" * 60)
    print(f"  🌳 Building forest with 200 trees...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    rf_model.fit(X_train, y_train)
    print("  ✅ Training complete!\n")
    
    print("=" * 60)
    print("📊 Evaluating Model Performance")
    print("=" * 60)
    
    train_preds = rf_model.predict(X_train)
    test_preds = rf_model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    
    print(f"\n  Training Accuracy: {train_acc*100:.2f}%")
    print(f"  Testing Accuracy:  {test_acc*100:.2f}%")
    
    print("\n" + "=" * 60)
    print(f"🚀 FINAL TEST ACCURACY: {test_acc*100:.2f}%")
    print("=" * 60)
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, test_preds, target_names=['Cat', 'Dog']))
    
    # Plot confusion matrix
    cm_path = os.path.join(MODELS_DIR, 'rf_confusion_matrix.png')
    plot_confusion_matrix(y_test, test_preds, cm_path)

    print("\n💾 Saving model...")
    save_path = os.path.join(MODELS_DIR, 'catdog_rf_enhanced.joblib')
    joblib.dump(rf_model, save_path)
    print(f"  ✅ Model saved to: {save_path}")
    
    # Also save the feature extractor reference (we'll load MobileNetV2 in backend)
    print("  ℹ️  Note: Feature extraction uses MobileNetV2 (loaded automatically)\n")
    
    print("=" * 60)
    print("✨ Training Complete!")
    print("=" * 60)
    print(f"\n📈 Summary:")
    print(f"   Model: Random Forest (200 trees)")
    print(f"   Features: MobileNetV2 deep features (1280 dims)")
    print(f"   Test Accuracy: {test_acc*100:.2f}%")
    print(f"\n🚀 To use this model:")
    print(f"   1. Restart Flask server: python app.py")
    print(f"   2. Select 'RF Enhanced' in dropdown")
    print(f"   3. Upload a cat or dog image")
    print("=" * 60 + "\n")

if __name__ == '__main__':
    main()