import numpy as np
import yfinance as yf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import (
    MobileNetV2, 
    ResNet50, 
    VGG16,
    InceptionV3
)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess, decode_predictions
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import cv2
import pickle
import os
from io import BytesIO
from PIL import Image

app = Flask(__name__, static_folder='static')
CORS(app)

MODELS_FOLDER = 'models'
os.makedirs(MODELS_FOLDER, exist_ok=True)

imagenet_models = {}
cat_dog_models = {}

def load_imagenet_models():
    """Load ImageNet pre-trained models"""
    print("📦 Loading ImageNet pre-trained models...")
    
    try:
        print("  Loading MobileNetV2...")
        imagenet_models['mobilenet'] = {
            'model': MobileNetV2(weights='imagenet'),
            'preprocess': mobilenet_preprocess,
            'size': (224, 224)
        }
        print("  ✅ MobileNetV2 loaded")
        
        print("  Loading ResNet50...")
        imagenet_models['resnet50'] = {
            'model': ResNet50(weights='imagenet'),
            'preprocess': resnet_preprocess,
            'size': (224, 224)
        }
        print("  ✅ ResNet50 loaded")
        
        print("  Loading VGG16...")
        imagenet_models['vgg16'] = {
            'model': VGG16(weights='imagenet'),
            'preprocess': vgg_preprocess,
            'size': (224, 224)
        }
        print("  ✅ VGG16 loaded")
        
        print("  Loading InceptionV3...")
        imagenet_models['inception'] = {
            'model': InceptionV3(weights='imagenet'),
            'preprocess': inception_preprocess,
            'size': (299, 299)
        }
        print("  ✅ InceptionV3 loaded")
        
        print("✅ All ImageNet models loaded successfully!\n")
        
    except Exception as e:
        print(f"⚠️  Error loading models: {e}")
        # Fallback to just MobileNetV2
        if 'mobilenet' not in imagenet_models:
            imagenet_models['mobilenet'] = {
                'model': MobileNetV2(weights='imagenet'),
                'preprocess': mobilenet_preprocess,
                'size': (224, 224)
            }

def load_cat_dog_models():
    """Load pre-trained cat vs dog models if available"""
    print("🐱🐶 Checking for custom trained models...")
    
    try:
        # Check for transfer learning model (fine-tuned on cat/dog)
        if os.path.exists(os.path.join(MODELS_FOLDER, 'catdog_transfer_learning.h5')):
            cat_dog_models['transfer_learning'] = load_model(
                os.path.join(MODELS_FOLDER, 'catdog_transfer_learning.h5')
            )
            print("  ✅ Loaded transfer learning model")
        
        # Check for enhanced models (Deep Features + ML)
        if os.path.exists(os.path.join(MODELS_FOLDER, 'catdog_svm_enhanced.joblib')):
            import joblib
            cat_dog_models['svm_enhanced'] = joblib.load(
                os.path.join(MODELS_FOLDER, 'catdog_svm_enhanced.joblib')
            )
            print("  ✅ Loaded SVM Enhanced model (MobileNetV2 features)")
        
        if os.path.exists(os.path.join(MODELS_FOLDER, 'catdog_rf_enhanced.joblib')):
            import joblib
            cat_dog_models['rf_enhanced'] = joblib.load(
                os.path.join(MODELS_FOLDER, 'catdog_rf_enhanced.joblib')
            )
            print("  ✅ Loaded Random Forest Enhanced model (MobileNetV2 features)")
        
        # Check for traditional ML models (old versions)
        if os.path.exists(os.path.join(MODELS_FOLDER, 'cat_dog_svm.pkl')):
            with open(os.path.join(MODELS_FOLDER, 'cat_dog_svm.pkl'), 'rb') as f:
                cat_dog_models['svm'] = pickle.load(f)
            print("  ✅ Loaded SVM model (basic)")
        
        if os.path.exists(os.path.join(MODELS_FOLDER, 'cat_dog_rf.pkl')):
            with open(os.path.join(MODELS_FOLDER, 'cat_dog_rf.pkl'), 'rb') as f:
                cat_dog_models['random_forest'] = pickle.load(f)
            print("  ✅ Loaded Random Forest model (basic)")
                
        if os.path.exists(os.path.join(MODELS_FOLDER, 'scaler.pkl')):
            with open(os.path.join(MODELS_FOLDER, 'scaler.pkl'), 'rb') as f:
                cat_dog_models['scaler'] = pickle.load(f)
            print("  ✅ Loaded feature scaler")
        
        # Load MobileNetV2 feature extractor for enhanced models
        if 'svm_enhanced' in cat_dog_models or 'rf_enhanced' in cat_dog_models:
            cat_dog_models['feature_extractor'] = MobileNetV2(
                weights='imagenet',
                include_top=False,
                pooling='avg',
                input_shape=(224, 224, 3)
            )
            print("  ✅ Loaded MobileNetV2 feature extractor")
        
        if cat_dog_models:
            print(f"✅ Loaded custom models: {list(cat_dog_models.keys())}\n")
        else:
            print("ℹ️  No custom models found. Using ImageNet models only.\n")
                
    except Exception as e:
        print(f"⚠️  Error loading custom models: {e}\n")

# Load all models on startup
load_imagenet_models()
load_cat_dog_models()

def prepare_image_imagenet(img_bytes, model_key='mobilenet'):
    """Prepare image for ImageNet models"""
    model_info = imagenet_models.get(model_key, imagenet_models['mobilenet'])
    target_size = model_info['size']
    preprocess_func = model_info['preprocess']
    
    img = Image.open(BytesIO(img_bytes)).convert('RGB')
    img = img.resize(target_size)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_func(img_array)

def prepare_image_for_catdog(img_bytes, img_size=(128, 128)):
    img = Image.open(BytesIO(img_bytes))
    img = np.array(img)
    
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    
    img = cv2.resize(img, img_size)
    return img / 255.0

def extract_deep_features(img_bytes):
    """Extract MobileNetV2 deep features for enhanced models"""
    # Load and preprocess image
    img = Image.open(BytesIO(img_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_preprocessed = mobilenet_preprocess(img_array)
    
    # Extract features using MobileNetV2
    if 'feature_extractor' in cat_dog_models:
        features = cat_dog_models['feature_extractor'].predict(img_preprocessed, verbose=0)
        return features
    else:
        # Fallback: create feature extractor on the fly
        feature_extractor = MobileNetV2(
            weights='imagenet',
            include_top=False,
            pooling='avg',
            input_shape=(224, 224, 3)
        )
        features = feature_extractor.predict(img_preprocessed, verbose=0)
        return features

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/classify', methods=['POST'])
def classify_image():
    """Classify image using ImageNet pre-trained models or custom models"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file'}), 400
        
        file = request.files['file']
        img_bytes = file.read()
        model_type = request.form.get('model', 'mobilenet')
        
        print(f"🔍 Classifying with model: {model_type}")
        
        # Transfer Learning Model (Cat vs Dog fine-tuned)
        if model_type == 'transfer_learning' and 'transfer_learning' in cat_dog_models:
            img = prepare_image_imagenet(img_bytes, 'mobilenet')
            prediction = cat_dog_models['transfer_learning'].predict(img, verbose=0)[0][0]
            
            if prediction > 0.5:
                label = 'Dog'
                confidence = prediction * 100
            else:
                label = 'Cat'
                confidence = (1 - prediction) * 100
            
            return jsonify({
                'label': label,
                'confidence': f"{confidence:.2f}%",
                'model': 'Transfer Learning (ImageNet Fine-tuned)',
                'description': 'MobileNetV2 fine-tuned on cat/dog dataset'
            })
        
        # Enhanced Models (Deep Features + ML)
        if model_type in ['svm_enhanced', 'rf_enhanced'] and model_type in cat_dog_models:
            # Extract deep features using MobileNetV2
            features = extract_deep_features(img_bytes)
            
            # Predict using the trained model
            prediction = cat_dog_models[model_type].predict(features)[0]
            label = 'Dog' if prediction == 1 else 'Cat'
            
            # Get probability if available
            if hasattr(cat_dog_models[model_type], 'predict_proba'):
                proba = cat_dog_models[model_type].predict_proba(features)[0]
                confidence = float(max(proba) * 100)
            else:
                confidence = 90.0
            
            model_name = 'SVM Enhanced' if model_type == 'svm_enhanced' else 'Random Forest Enhanced'
            
            return jsonify({
                'label': label,
                'confidence': f"{confidence:.2f}%",
                'model': model_name,
                'description': f'{model_name} (MobileNetV2 features + ML)'
            })
        
        # Traditional ML models (Cat vs Dog)
        if model_type in ['svm', 'random_forest'] and model_type in cat_dog_models:
            img = prepare_image_for_catdog(img_bytes)
            img_flat = img.reshape(1, -1)
            
            if 'scaler' in cat_dog_models:
                img_scaled = cat_dog_models['scaler'].transform(img_flat)
            else:
                img_scaled = img_flat
            
            prediction = cat_dog_models[model_type].predict(img_scaled)[0]
            label = 'Dog' if prediction == 1 else 'Cat'
            
            if hasattr(cat_dog_models[model_type], 'predict_proba'):
                proba = cat_dog_models[model_type].predict_proba(img_scaled)[0]
                confidence = float(max(proba) * 100)
            else:
                confidence = 85.0
            
            model_name = 'SVM' if model_type == 'svm' else 'Random Forest'
            
            return jsonify({
                'label': label,
                'confidence': f"{confidence:.2f}%",
                'model': model_name,
                'description': f'{model_name} trained on cat/dog images'
            })
        
        # ImageNet Pre-trained Models
        if model_type in imagenet_models:
            img = prepare_image_imagenet(img_bytes, model_type)
            model_info = imagenet_models[model_type]
            
            preds = model_info['model'].predict(img, verbose=0)
            results = decode_predictions(preds, top=3)[0]
            
            # Get top prediction
            top_pred = results[0]
            label = top_pred[1].replace('_', ' ').title()
            confidence = top_pred[2] * 100
            
            # Get top 3 for additional info
            top_3 = [
                {
                    'label': r[1].replace('_', ' ').title(),
                    'confidence': float(r[2] * 100)
                }
                for r in results
            ]
            
            model_names = {
                'mobilenet': 'MobileNetV2',
                'resnet50': 'ResNet50',
                'vgg16': 'VGG16',
                'inception': 'InceptionV3'
            }
            
            return jsonify({
                'label': label,
                'confidence': f"{confidence:.2f}%",
                'model': model_names.get(model_type, 'MobileNetV2'),
                'description': f'ImageNet pre-trained (1000 classes)',
                'top_3': top_3
            })
        
        # Default to MobileNetV2
        img = prepare_image_imagenet(img_bytes, 'mobilenet')
        preds = imagenet_models['mobilenet']['model'].predict(img, verbose=0)
        results = decode_predictions(preds, top=1)[0]
        
        return jsonify({
            'label': results[0][1].replace('_', ' ').title(),
            'confidence': f"{results[0][2]*100:.2f}%",
            'model': 'MobileNetV2',
            'description': 'ImageNet pre-trained (1000 classes)'
        })
        
    except Exception as e:
        print(f"❌ Classification error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/available_models', methods=['GET'])
def get_available_models():
    """Return list of available models"""
    models = []
    
    # ImageNet models
    for key, name in [
        ('mobilenet', 'MobileNetV2 (Fast & Accurate)'),
        ('resnet50', 'ResNet50 (Deep Learning)'),
        ('vgg16', 'VGG16 (Classic CNN)'),
        ('inception', 'InceptionV3 (Advanced)')
    ]:
        if key in imagenet_models:
            models.append({
                'value': key,
                'label': name,
                'type': 'imagenet',
                'classes': 1000
            })
    
    # Custom models
    if 'transfer_learning' in cat_dog_models:
        models.append({
            'value': 'transfer_learning',
            'label': 'Transfer Learning (Cat vs Dog)',
            'type': 'custom',
            'classes': 2,
            'description': 'Fine-tuned MobileNetV2'
        })
    
    if 'svm_enhanced' in cat_dog_models:
        models.append({
            'value': 'svm_enhanced',
            'label': 'SVM Enhanced (Cat vs Dog)',
            'type': 'custom',
            'classes': 2,
            'description': 'MobileNetV2 features + SVM'
        })
    
    if 'rf_enhanced' in cat_dog_models:
        models.append({
            'value': 'rf_enhanced',
            'label': 'Random Forest Enhanced (Cat vs Dog)',
            'type': 'custom',
            'classes': 2,
            'description': 'MobileNetV2 features + RF'
        })
    
    if 'svm' in cat_dog_models:
        models.append({
            'value': 'svm',
            'label': 'SVM Basic (Cat vs Dog)',
            'type': 'custom',
            'classes': 2,
            'description': 'Traditional SVM'
        })
    
    if 'random_forest' in cat_dog_models:
        models.append({
            'value': 'random_forest',
            'label': 'Random Forest Basic (Cat vs Dog)',
            'type': 'custom',
            'classes': 2,
            'description': 'Traditional Random Forest'
        })
    
    return jsonify({'models': models})

@app.route('/predict_stock', methods=['POST'])
def predict_stock():
    try:
        data = request.json
        ticker = data.get('ticker', 'AAPL').upper()
        
        # Fetch Data
        stock = yf.download(ticker, period='2y', interval='1d', progress=False)
        if stock.empty:
            return jsonify({'error': f'Invalid ticker: {ticker}'}), 400

        values = stock['Close'].values.reshape(-1, 1)
        
        # LINEAR REGRESSION
        X_lin = np.arange(len(values)).reshape(-1, 1)
        y_lin = values
        X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(
            X_lin, y_lin, test_size=0.2, shuffle=False
        )
        
        lr = LinearRegression()
        lr.fit(X_train_lin, y_train_lin)
        lr_preds = lr.predict(X_test_lin).flatten()
        
        lr_mse = np.mean((y_test_lin.flatten() - lr_preds) ** 2)
        lr_r2 = lr.score(X_test_lin, y_test_lin)

        # LSTM
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(values)
        
        look_back = 60
        X_lstm, y_lstm = [], []
        
        for i in range(look_back, len(scaled_data)):
            X_lstm.append(scaled_data[i-look_back:i, 0])
            y_lstm.append(scaled_data[i, 0])
            
        X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
        X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))
        
        split_idx = int(len(X_lstm) * 0.8)
        X_train_lstm, X_test_lstm = X_lstm[:split_idx], X_lstm[split_idx:]
        y_train_lstm, y_test_lstm = y_lstm[:split_idx], y_lstm[split_idx:]

        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train_lstm, y_train_lstm, batch_size=32, epochs=5, verbose=0)
        
        lstm_preds_scaled = model.predict(X_test_lstm, verbose=0)
        lstm_preds = scaler.inverse_transform(lstm_preds_scaled).flatten()
        
        lstm_mse = np.mean((scaler.inverse_transform(y_test_lstm.reshape(-1, 1)).flatten() - lstm_preds) ** 2)

        # Prepare response
        all_dates = stock.index.strftime('%Y-%m-%d').tolist()
        lr_padding = [None] * (len(values) - len(lr_preds))
        lstm_padding = [None] * (len(values) - len(lstm_preds))
        
        # Next day prediction
        last_60_days = scaled_data[-look_back:]
        last_60_days = np.reshape(last_60_days, (1, look_back, 1))
        next_day_scaled = model.predict(last_60_days, verbose=0)
        next_day_price = scaler.inverse_transform(next_day_scaled)[0][0]

        return jsonify({
            'ticker': ticker,
            'dates': all_dates,
            'actual': values.flatten().tolist(),
            'linear_preds': lr_padding + lr_preds.tolist(),
            'lstm_preds': lstm_padding + lstm_preds.tolist(),
            'next_day_prediction': float(next_day_price),
            'current_price': float(values[-1][0]),
            'metrics': {
                'linear_regression': {'mse': float(lr_mse), 'r2': float(lr_r2)},
                'lstm': {'mse': float(lstm_mse)}
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 ML Experiments Server")
    print("=" * 60)
    print(f"📍 Server: http://localhost:5000")
    print(f"🤖 ImageNet Models: {list(imagenet_models.keys())}")
    print(f"🎯 Custom Models: {list(cat_dog_models.keys()) if cat_dog_models else 'None'}")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000, host='0.0.0.0')