import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model, Model
import pickle
import requests
import logging
from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image  # Ensure PIL is imported for proper image handling

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Paths
model_path = "rl_jewelry_model.keras"
scaler_path = "scaler.pkl"
pairwise_features_path = "pairwise_features.npy"

# Flask app
app = Flask(__name__)

# ---------------------- Jewelry RL Predictor ----------------------
class JewelryRLPredictor:
    def __init__(self, model_path, scaler_path, pairwise_features_path):
        # Check file existence and log missing files
        missing_files = [p for p in [model_path, scaler_path, pairwise_features_path] if not os.path.exists(p)]
        if missing_files:
            raise FileNotFoundError(f"Missing files: {', '.join(missing_files)}")

        logger.info("🚀 Loading model...")
        self.model = load_model(model_path)  # No custom objects needed
        self.img_size = (224, 224)
        self.feature_size = 1280
        self.device = "/CPU:0"  # Force CPU usage

        logger.info("📏 Loading scaler...")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        logger.info("🔄 Setting up MobileNetV2 feature extractor...")
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        global_avg_layer = tf.keras.layers.GlobalAveragePooling2D()
        self.feature_extractor = Model(inputs=base_model.input, outputs=global_avg_layer(base_model.output))

        logger.info("📂 Loading pairwise features...")
        try:
            self.pairwise_features = np.load(pairwise_features_path, allow_pickle=False).item()
        except ValueError:
            logger.warning("⚠️ Could not load pairwise features with `allow_pickle=False`, retrying with `allow_pickle=True`")
            self.pairwise_features = np.load(pairwise_features_path, allow_pickle=True).item()

        self.jewelry_names = list(self.pairwise_features.keys())

        logger.info("✅ Predictor initialized successfully!")

    def extract_features(self, img_data):
        """Extract features from an image file or URL."""
        try:
            img = Image.open(BytesIO(img_data))  # Open image from bytes
            img = img.resize(self.img_size)  # Resize properly
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            features = self.feature_extractor.predict(img_array, verbose=0)
            return self.scaler.transform(features)
        except Exception as e:
            logger.error(f"❌ Error extracting features: {e}")
            return None

    def predict_compatibility(self, face_data, jewel_data):
        """Predict compatibility between a face and jewelry."""
        face_features = self.extract_features(face_data)
        jewel_features = self.extract_features(jewel_data)
        if face_features is None or jewel_features is None:
            return None, "Feature extraction failed", []

        cosine_similarity = np.dot(face_features, jewel_features.T).flatten()[0]
        scaled_score = (cosine_similarity + 1) / 2.0  # Normalize to 0-1 range
        category = ("🌟 Very Good" if scaled_score >= 0.8 else
                    "✅ Good" if scaled_score >= 0.6 else
                    "😐 Neutral" if scaled_score >= 0.4 else
                    "⚠️ Bad" if scaled_score >= 0.2 else
                    "❌ Very Bad")

        return scaled_score, category, self.jewelry_names[:10]  # Return top jewelry names as dummy recommendations

# Initialize predictor
try:
    predictor = JewelryRLPredictor(model_path, scaler_path, pairwise_features_path)
except Exception as e:
    logger.error(f"🚨 Failed to initialize JewelryRLPredictor: {e}")
    predictor = None

# ---------------------- Home Route ----------------------
@app.route('/')
def home():
    return "Jewelry RL API is running!", 200

# ---------------------- Flask API Endpoint ----------------------
@app.route('/predict', methods=['POST'])
def predict():
    if predictor is None:
        return jsonify({'error': 'Model is not loaded properly'}), 500

    face_data = request.files.get('face')
    jewelry_data = request.files.get('jewelry')
    face_url = request.form.get('face_url')
    jewelry_url = request.form.get('jewelry_url')

    if face_data:
        face_data = face_data.read()
    elif face_url:
        try:
            face_data = requests.get(face_url).content
        except requests.RequestException as e:
            return jsonify({'error': f'Failed to fetch face image: {e}'}), 400
    else:
        return jsonify({'error': 'Face image is required'}), 400

    if jewelry_data:
        jewelry_data = jewelry_data.read()
    elif jewelry_url:
        try:
            jewelry_data = requests.get(jewelry_url).content
        except requests.RequestException as e:
            return jsonify({'error': f'Failed to fetch jewelry image: {e}'}), 400
    else:
        return jsonify({'error': 'Jewelry image is required'}), 400

    score, category, recommendations = predictor.predict_compatibility(face_data, jewelry_data)
    if score is None:
        return jsonify({'error': 'Prediction failed'}), 500

    return jsonify({'score': float(score), 'category': category, 'recommendations': recommendations})

# ---------------------- Production Deployment ----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's assigned port
    app.run(host='0.0.0.0', port=port, debug=False)
