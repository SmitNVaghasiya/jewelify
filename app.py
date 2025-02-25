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

# ✅ Corrected Base Path (Relative to Server Folder)
# base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "trained_features"))
model_path = "trained_features/keras/rl_jewelry_model.keras"
scaler_path = "trained_features/scaler.pkl"
pairwise_features_path = "trained_features/pandas/pairwise_features.npy"

# ✅ Debugging Paths
print("🔍 Checking paths:")
# print("Base Path:", base_path)
print("Model Path:", model_path, "Exists:", os.path.exists(model_path))
print("Scaler Path:", scaler_path, "Exists:", os.path.exists(scaler_path))
print("Pairwise Features Path:", pairwise_features_path, "Exists:", os.path.exists(pairwise_features_path))

# Custom InputLayer to handle 'batch_shape' keyword
class CustomInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        super().__init__(**kwargs)

class JewelryRLPredictor:
    def __init__(self, model_path, scaler_path, pairwise_features_path):
        # 🔍 Ensure all required files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model file not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"❌ Scaler file not found: {scaler_path}")
        if not os.path.exists(pairwise_features_path):
            raise FileNotFoundError(f"❌ Pairwise features file not found: {pairwise_features_path}")

        print("✅ Loading model...")
        self.model = load_model(model_path, custom_objects={'InputLayer': CustomInputLayer})
        self.img_size = (224, 224)
        self.device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
        self.feature_size = 1280

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

    def categorize_suitability(self, score):
        """Convert numerical compatibility score into categories."""
        if score < 0.2:
            return "❌ Very Bad"
        elif score < 0.4:
            return "⚠️ Bad"
        elif score < 0.6:
            return "😐 Neutral"
        elif score < 0.8:
            return "✅ Good"
        else:
            return "🌟 Very Good"

    def predict_compatibility(self, face_img_file, jewel_img_file):
        face_features = self.extract_features(face_img_file)
        jewel_features = self.extract_features(jewel_img_file)
        if face_features is None or jewel_features is None:
            return None, None, None

        face_norm = face_features / np.linalg.norm(face_features, axis=1, keepdims=True)
        jewel_norm = jewel_features / np.linalg.norm(jewel_features, axis=1, keepdims=True)
        cosine_similarity = np.sum(face_norm * jewel_norm, axis=1)[0]  
        scaled_score = (cosine_similarity + 1) / 2.0
        category = self.categorize_suitability(scaled_score)

        # Recommend jewelry using RL model
        with tf.device(self.device):
            q_values = self.model.predict(face_features, verbose=0)[0]
        top_indices = np.argsort(q_values)[::-1]
        total_options = len(self.jewelry_list)
        top_n = min(max(10, total_options), 15)
        top_recommendations = [(self.jewelry_names[idx], q_values[idx]) for idx in top_indices[:top_n]]
        top_recommendations.sort(key=lambda x: x[0])
        recommendations = [name for name, _ in top_recommendations]

        return scaled_score, category, recommendations

# ✅ Initialize Predictor (Handle Errors)
try:
    predictor = JewelryRLPredictor(model_path, scaler_path, pairwise_features_path)
except Exception as e:
    print(f"❌ Failed to initialize JewelryRLPredictor: {e}")
    predictor = None

@app.route('/predict', methods=['POST'])
def predict():
    if predictor is None:
        return jsonify({'error': 'Model is not loaded properly'}), 500

    # ✅ Ensure correct field names
    if 'face_image' not in request.files or 'jewel_image' not in request.files:
        return jsonify({'error': 'Both face_image and jewel_image must be provided'}), 400

    face_file = request.files['face_image']
    jewel_file = request.files['jewel_image']

    if face_file.filename == '' or jewel_file.filename == '':
        return jsonify({'error': 'No file selected for one or both images'}), 400

    score, category, recommendations = predictor.predict_compatibility(face_file, jewel_file)
    if score is None:
        return jsonify({'error': 'Prediction failed'}), 500

    response = {
        'score': float(score),
        'category': category,
        'recommendations': recommendations
    }
    return jsonify(response), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
