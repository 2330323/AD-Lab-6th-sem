import os
import numpy as np
import cv2
import joblib
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
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
    print("📂 Loading Dataset for Hybrid SVM")
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
    """
    print("🧠 Extracting Deep Features using MobileNetV2...")
    print("-" * 60)
    
    # Load MobileNetV2 without the top layer (classifier), using Average Pooling
    print("  📦 Loading MobileNetV2 feature extractor...")
    model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=(224, 224, 3)
    )
    
    # Preprocess images exactly as MobileNet expects
    print("  🔧 Preprocessing images...")
    processed_images = preprocess_input(images.astype(np.float32))
    
    # Extract features
    print("  ⚙️  Extracting features (this may take a moment)...")
    features = model.predict(processed_images, batch_size=32, verbose=1)
    
    print(f"  ✅ Features extracted! Shape: {features.shape}")
    print(f"     Each image → {features.shape[1]} feature values\n")
    
    return features

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plotting and saving the confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Cat', 'Dog'], 
                yticklabels=['Cat', 'Dog'])
    plt.title('SVM Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"  📊 Confusion matrix saved to: {save_path}")
    plt.close()

def main():
    print("\n" + "=" * 60)
    print("🎯 High-Accuracy SVM Classifier")
    print("   Deep Learning Features + Support Vector Machine")
    print("=" * 60 + "\n")
    
    # 1. Load Data
    X_images, y = load_dataset()
    if X_images is None:
        return

    # 2. Extract Features (For High Accuracy)
    X_features = extract_features(X_images)

    # 3. Split Data
    print("📊 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}\n")

    # 4. Train SVM
    print("=" * 60)
    print("🎯 Training Support Vector Machine")
    print("=" * 60)
    
    print("  ⚙️  Training SVM with RBF kernel...")
    svm_model = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        probability=True,  
        random_state=42,
        verbose=True
    )
    
    svm_model.fit(X_train, y_train)
    print("  ✅ Training complete!\n")
    
    # Option 2: Grid Search (Uncomment for hyperparameter tuning - takes longer)
    """
    print("  🔍 Performing Grid Search for optimal hyperparameters...")
    print("     (This may take several minutes)")
    
    param_grid = {
        'C': [1, 10, 100],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'linear']
    }
    
    grid_search = GridSearchCV(
        SVC(probability=True, random_state=42),
        param_grid,
        cv=3,
        scoring='accuracy',
        verbose=2,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    svm_model = grid_search.best_estimator_
    
    print(f"\n  ✅ Best parameters found: {grid_search.best_params_}")
    print(f"     Best CV score: {grid_search.best_score_*100:.2f}%\n")
    """
    
    print("=" * 60)
    print("📊 Evaluating Model Performance")
    print("=" * 60)
    
    train_preds = svm_model.predict(X_train)
    test_preds = svm_model.predict(X_test)
    
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
    cm_path = os.path.join(MODELS_DIR, 'svm_confusion_matrix.png')
    plot_confusion_matrix(y_test, test_preds, cm_path)
    
    # Show support vectors info
    print(f"\n🔍 SVM Details:")
    print(f"   Support Vectors: {svm_model.n_support_}")
    print(f"   Total: {sum(svm_model.n_support_)} support vectors used")

    # 6. Save Model
    print("\n💾 Saving model...")
    save_path = os.path.join(MODELS_DIR, 'catdog_svm_enhanced.joblib')
    joblib.dump(svm_model, save_path)
    print(f"  ✅ Model saved to: {save_path}")
    
    # Also save feature extractor reference
    print("  ℹ️  Note: Feature extraction uses MobileNetV2 (loaded automatically)\n")
    
    print("=" * 60)
    print("✨ Training Complete!")
    print("=" * 60)
    print(f"\n📈 Summary:")
    print(f"   Model: Support Vector Machine (RBF kernel)")
    print(f"   Features: MobileNetV2 deep features (1280 dims)")
    print(f"   Test Accuracy: {test_acc*100:.2f}%")
    print(f"   Support Vectors: {sum(svm_model.n_support_)}")
    print(f"\n🚀 To use this model:")
    print(f"   1. Restart Flask server: python app.py")
    print(f"   2. Select 'SVM Enhanced' in dropdown")
    print(f"   3. Upload a cat or dog image")
    print("=" * 60 + "\n")

if __name__ == '__main__':
    main()