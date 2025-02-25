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

# Paths (relative to project root on Render)
base_path = "./trained_features"  # Assumes files are in trained_features/ folder
model_path = f"{base_path}/keras/rl_jewelry_model.keras"
scaler_path = f"{base_path}/scaler.pkl"
pairwise_features_path = f"{base_path}/pandas/pairwise_features.npy"

class JewelryRLPredictor:
    def __init__(self, model_path, scaler_path, pairwise_features_path):
        self.model = load_model(model_path)
        self.img_size = (224, 224)
        self.device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
        self.feature_size = 1280

        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        global_avg_layer = tf.keras.layers.GlobalAveragePooling2D()
        reduction_layer = tf.keras.layers.Dense(self.feature_size, activation="relu")
        self.feature_extractor = Model(
            inputs=base_model.input,
            outputs=reduction_layer(global_avg_layer(base_model.output))
        )

        self.pairwise_features = np.load(pairwise_features_path, allow_pickle=True).item()
        self.pairwise_features = {k: self.scaler.transform(np.array(v).reshape(1, -1)) for k, v in self.pairwise_features.items() if v is not None and v.size == 1280}
        self.jewelry_list = list(self.pairwise_features.values())
        self.jewelry_names = list(self.pairwise_features.keys())

    def extract_features(self, img_file):
        try:
            # Process image directly from memory
            img = image.load_img(BytesIO(img_file.read()), target_size=self.img_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            features = self.feature_extractor.predict(img_array, verbose=0)
            return self.scaler.transform(features)
        except Exception as e:
            print(f"⚠️ Error extracting features: {e}")
            return None

    def predict_compatibility(self, face_img_file):
        face_features = self.extract_features(face_img_file)
        if face_features is None:
            return None, None

        with tf.device(self.device):
            q_values = self.model.predict(face_features, verbose=0)[0]
            action = np.argmax(q_values)
            score = q_values[action]

        recommendations = self.recommend_jewelry(face_features)
        return score, recommendations

    def recommend_jewelry(self, face_features, min_top_n=10, max_top_n=15):
        with tf.device(self.device):
            q_values = self.model.predict(face_features, verbose=0)[0]
        
        top_indices = np.argsort(q_values)[::-1]
        total_options = len(self.jewelry_list)
        top_n = min(max(min_top_n, total_options), max_top_n)
        top_recommendations = [(self.jewelry_names[idx], q_values[idx]) for idx in top_indices[:top_n]]
        top_recommendations.sort(key=lambda x: x[0])
        return [name for name, _ in top_recommendations]

# Initialize predictor
predictor = JewelryRLPredictor(model_path, scaler_path, pairwise_features_path)

@app.route('/predict', methods=['POST'])
def predict():
    if 'face_image' not in request.files:
        return jsonify({'error': 'No face image provided'}), 400
    
    file = request.files['face_image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Process the file directly from memory
    score, recommendations = predictor.predict_compatibility(file)
    
    if score is None:
        return jsonify({'error': 'Prediction failed'}), 500
    
    response = {
        'score': float(score),
        'recommendations': recommendations
    }
    return jsonify(response), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)