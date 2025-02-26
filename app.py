import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model, Model
import pickle
import requests
from flask import Flask, request, jsonify
from io import BytesIO

# Define Paths (Update paths if needed)
model_path = "trained_features/keras/rl_jewelry_model.keras"
scaler_path = "trained_features/scaler.pkl"
pairwise_features_path = "trained_features/pandas/pairwise_features.npy"

# Flask app
app = Flask(__name__)

# ---------------------- Jewelry RL Predictor ----------------------
class JewelryRLPredictor:
    def __init__(self, model_path, scaler_path, pairwise_features_path):
        for path in [model_path, scaler_path, pairwise_features_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing required file: {path}")

        print("🚀 Loading model...")
        self.model = load_model(model_path)
        self.img_size = (224, 224)
        self.feature_size = 1280
        self.device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"

        print("📏 Loading scaler...")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        print("🔄 Setting up MobileNetV2 feature extractor...")
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        global_avg_layer = tf.keras.layers.GlobalAveragePooling2D()
        reduction_layer = tf.keras.layers.Dense(self.feature_size, activation="relu")
        self.feature_extractor = Model(
            inputs=base_model.input,
            outputs=reduction_layer(global_avg_layer(base_model.output))
        )

        print("📂 Loading pairwise features...")
        self.pairwise_features = np.load(pairwise_features_path, allow_pickle=True).item()
        self.pairwise_features = {
            k: self.scaler.transform(np.array(v).reshape(1, -1))
            for k, v in self.pairwise_features.items() if v is not None and v.size == 1280
        }
        self.jewelry_list = list(self.pairwise_features.values())
        self.jewelry_names = list(self.pairwise_features.keys())
        print("✅ Predictor initialized successfully!")

    def extract_features(self, img_data):
        """Extract features from an image file or URL."""
        try:
            img = image.load_img(BytesIO(img_data), target_size=self.img_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            features = self.feature_extractor.predict(img_array, verbose=0)
            return self.scaler.transform(features)
        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            return None

    def predict_compatibility(self, face_data, jewel_data):
        """Predict compatibility between a face and jewelry."""
        face_features = self.extract_features(face_data)
        jewel_features = self.extract_features(jewel_data)
        if face_features is None or jewel_features is None:
            return None, "Feature extraction failed", []

        # Normalize and compute cosine similarity
        face_norm = face_features / np.linalg.norm(face_features, axis=1, keepdims=True)
        jewel_norm = jewel_features / np.linalg.norm(jewel_features, axis=1, keepdims=True)
        cosine_similarity = np.sum(face_norm * jewel_norm, axis=1)[0]
        scaled_score = (cosine_similarity + 1) / 2.0
        category = ("🌟 Very Good" if scaled_score >= 0.8 else
                    "✅ Good" if scaled_score >= 0.6 else
                    "😐 Neutral" if scaled_score >= 0.4 else
                    "⚠️ Bad" if scaled_score >= 0.2 else
                    "❌ Very Bad")

        with tf.device(self.device):
            q_values = self.model.predict(face_features, verbose=0)[0]
        top_indices = np.argsort(q_values)[::-1]
        top_recommendations = [(self.jewelry_names[idx], q_values[idx]) for idx in top_indices[:10]]
        recommendations = [name for name, _ in top_recommendations]

        return scaled_score, category, recommendations

# Initialize predictor
try:
    predictor = JewelryRLPredictor(model_path, scaler_path, pairwise_features_path)
except Exception as e:
    print(f"🚨 Failed to initialize JewelryRLPredictor: {e}")
    predictor = None

# ---------------------- Flask API Endpoint ----------------------
@app.route('/predict', methods=['POST'])
def predict():
    if predictor is None:
        return jsonify({'error': 'Model is not loaded properly'}), 500

    # Accept file uploads or URLs
    face_data = request.files.get('face')
    jewelry_data = request.files.get('jewelry')
    face_url = request.form.get('face_url')
    jewelry_url = request.form.get('jewelry_url')

    if face_data:
        face_data = face_data.read()
    elif face_url:
        face_data = requests.get(face_url).content
    else:
        return jsonify({'error': 'Face image is required'}), 400

    if jewelry_data:
        jewelry_data = jewelry_data.read()
    elif jewelry_url:
        jewelry_data = requests.get(jewelry_url).content
    else:
        return jsonify({'error': 'Jewelry image is required'}), 400

    score, category, recommendations = predictor.predict_compatibility(face_data, jewelry_data)
    if score is None:
        return jsonify({'error': 'Prediction failed'}), 500

    return jsonify({'score': float(score), 'category': category, 'recommendations': recommendations})

# ---------------------- Production Deployment ----------------------
if __name__ == "__main__":
    # Note: For production deployment on Render or any similar service,
    # it is recommended to use a production WSGI server like gunicorn.
    # Example command: gunicorn app:app --bind 0.0.0.0:5000
    app.run(host='0.0.0.0', port=5000, debug=False)
