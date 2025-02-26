import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model, Model
import pickle
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from io import BytesIO

app = Flask(__name__)

# Define Corrected Paths
# base_path = os.path.abspath(os.path.dirname(__file__))
# model_path = os.path.join(base_path, "trained_features", "keras", "rl_jewelry_model.keras")
# scaler_path = os.path.join(base_path, "trained_features", "scaler.pkl")
# pairwise_features_path = os.path.join(base_path, "trained_features", "pandas", "pairwise_features.npy")

model_path = "trained_features\keras\rl_jewelry_model.keras"
scaler_path = "trained_features\scaler.pkl"
pairwise_features_path = "trained_features\pandas\pairwise_features.npy"

print("🔍 Checking paths:")
print("Model Path:", model_path, "Exists:", os.path.exists(model_path))
print("Scaler Path:", scaler_path, "Exists:", os.path.exists(scaler_path))
print("Pairwise Features Path:", pairwise_features_path, "Exists:", os.path.exists(pairwise_features_path))

# Fix InputLayer issue
class CustomInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        super().__init__(**kwargs)

class JewelryRLPredictor:
    def __init__(self, model_path, scaler_path, pairwise_features_path):
        for path in [model_path, scaler_path, pairwise_features_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"❌ Missing required file: {path}")

        print("✅ Loading model...")
        self.model = load_model(model_path, custom_objects={'InputLayer': CustomInputLayer})
        self.img_size = (224, 224)
        self.feature_size = 1280
        self.device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"

        print("✅ Loading scaler...")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        print("✅ Setting up MobileNetV2 feature extractor...")
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        global_avg_layer = tf.keras.layers.GlobalAveragePooling2D()
        reduction_layer = tf.keras.layers.Dense(self.feature_size, activation="relu")
        self.feature_extractor = Model(
            inputs=base_model.input,
            outputs=reduction_layer(global_avg_layer(base_model.output))
        )

        print("✅ Loading pairwise features...")
        self.pairwise_features = np.load(pairwise_features_path, allow_pickle=True).item()
        self.pairwise_features = {
            k: self.scaler.transform(np.array(v).reshape(1, -1))
            for k, v in self.pairwise_features.items() if v is not None and v.size == 1280
        }
        self.jewelry_list = list(self.pairwise_features.values())
        self.jewelry_names = list(self.pairwise_features.keys())
        print("✅ Predictor initialized successfully!")

    def extract_features(self, img_file):
        try:
            img = image.load_img(BytesIO(img_file.read()), target_size=self.img_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            features = self.feature_extractor.predict(img_array, verbose=0)
            return self.scaler.transform(features)
        except Exception as e:
            print(f"⚠️ Error extracting features: {e}")
            return None

    def predict_compatibility(self, face_img_file, jewel_img_file):
        face_features = self.extract_features(face_img_file)
        jewel_features = self.extract_features(jewel_img_file)
        if face_features is None or jewel_features is None:
            return None, "Feature extraction failed"

        face_norm = face_features / np.linalg.norm(face_features, axis=1, keepdims=True)
        jewel_norm = jewel_features / np.linalg.norm(jewel_features, axis=1, keepdims=True)
        cosine_similarity = np.sum(face_norm * jewel_norm, axis=1)[0]  
        scaled_score = (cosine_similarity + 1) / 2.0
        category = "🌟 Very Good" if scaled_score >= 0.8 else "✅ Good" if scaled_score >= 0.6 else "😐 Neutral" if scaled_score >= 0.4 else "⚠️ Bad" if scaled_score >= 0.2 else "❌ Very Bad"
        
        with tf.device(self.device):
            q_values = self.model.predict(face_features, verbose=0)[0]
        top_indices = np.argsort(q_values)[::-1]
        top_recommendations = [(self.jewelry_names[idx], q_values[idx]) for idx in top_indices[:10]]
        recommendations = [name for name, _ in top_recommendations]
        
        return scaled_score, category, recommendations

try:
    predictor = JewelryRLPredictor(model_path, scaler_path, pairwise_features_path)
except Exception as e:
    print(f"❌ Failed to initialize JewelryRLPredictor: {e}")
    predictor = None

@app.route('/predict', methods=['POST'])
def predict():
    if predictor is None:
        return jsonify({'error': 'Model is not loaded properly', 'details': 'Initialization failure'}), 500

    if 'face' not in request.files or 'jewelry' not in request.files:
        return jsonify({'error': 'Both face and jewelry images must be provided', 'details': 'Missing files'}), 400

    face_file = request.files['face']
    jewel_file = request.files['jewelry']

    if face_file.filename == '' or jewel_file.filename == '':
        return jsonify({'error': 'No file selected for one or both images', 'details': 'Empty filename'}), 400

    score, category, recommendations = predictor.predict_compatibility(face_file, jewel_file)
    if score is None:
        return jsonify({'error': 'Prediction failed', 'details': category}), 500

    return jsonify({
        'score': float(score),
        'category': category,
        'recommendations': recommendations
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
