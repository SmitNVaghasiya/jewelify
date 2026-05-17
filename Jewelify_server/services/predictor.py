import cv2
import numpy as np
import tensorflow as tf
import xgboost as xgb
import logging
import os
from typing import List, Tuple
import asyncio
from dotenv import load_dotenv
import time
from services.database import get_db_client
from bson import ObjectId
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import urllib.parse

load_dotenv()

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
logging_level = logging.DEBUG if ENVIRONMENT == "development" else logging.WARNING
logging.basicConfig(level=logging_level, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logging.getLogger("pymongo").setLevel(logging.INFO)


class JewelryPredictor:
    def __init__(self, device: str = "CPU"):
        self.device = device

        cascade_path = os.getenv("HAAR_CASCADE_PATH", "haarcascade_frontalface_default.xml")
        if not os.path.exists(cascade_path):
            logger.warning(f"Haar Cascade file not found at {cascade_path}. Face validation will be skipped.")
            self.face_cascade = None
        else:
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                logger.warning(f"Failed to load Haar Cascade at {cascade_path}. Face validation will be skipped.")
                self.face_cascade = None
            else:
                logger.info("Haar Cascade loaded successfully")

        try:
            logger.info("Loading MobileNetV2 model...")
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights="imagenet",
                pooling=None,
            )
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(640, activation='relu')(x)
            self.mobile_net = Model(inputs=base_model.input, outputs=x)
            logger.info("MobileNetV2 model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading MobileNetV2 model: {str(e)}")
            raise RuntimeError(f"Failed to load MobileNetV2 model: {str(e)}")

        try:
            logger.info("Loading XGBoost model...")
            t = time.time()
            xgboost_model_path = os.getenv("XGBOOST_MODEL_PATH", "xgboost_jewelry_v1.model")
            if not os.path.exists(xgboost_model_path):
                raise FileNotFoundError(f"XGBoost model not found at {xgboost_model_path}")
            self.xgboost_model = xgb.Booster()
            self.xgboost_model.load_model(xgboost_model_path)
            logger.info(f"XGBoost model loaded in {time.time() - t:.2f}s")
        except Exception as e:
            logger.error(f"Error loading XGBoost model: {str(e)}")
            raise RuntimeError(f"Failed to load XGBoost model: {str(e)}")

        try:
            logger.info("Loading MLP model...")
            t = time.time()
            mlp_model_path = os.getenv("MLP_MODEL_PATH", "mlp_jewelry_v1.keras")
            if not os.path.exists(mlp_model_path):
                raise FileNotFoundError(f"MLP model not found at {mlp_model_path}")
            self.mlp_model = tf.keras.models.load_model(mlp_model_path)
            logger.info(f"MLP model loaded in {time.time() - t:.2f}s")
        except Exception as e:
            logger.error(f"Error loading MLP model: {str(e)}")
            raise RuntimeError(f"Failed to load MLP model: {str(e)}")

        # M-007: corrected S3 path ("earrings" not "earings")
        self.s3_base_url = os.getenv(
            "S3_BASE_URL",
            "https://jewelify-images.s3.eu-north-1.amazonaws.com/Necklace with earrings_sorted_jpg/"
        )

        self.jewelry_categories = [
            "Necklace with earrings",
            "Earrings only",
            "Necklace only",
            "Bracelet",
            "Ring",
            "Not Assigned",
        ]

    # --- sync helpers (run in thread pool via asyncio.to_thread) ---

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        image = cv2.resize(image, (224, 224))
        image = image.astype("float32")
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return np.expand_dims(image, axis=0)

    def _validate_image_sync(self, image: np.ndarray) -> Tuple[bool, str]:
        if image is None or image.size == 0:
            return False, "Image is empty"
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if np.std(gray) < 10:
            return False, "Image is blank"
        return True, "Valid"

    def _validate_face_sync(self, image: np.ndarray) -> Tuple[bool, str]:
        if self.face_cascade is None:
            return True, "Validation skipped (Haar Cascade not loaded)"
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return False, "No faces detected in the image"
        return True, "Valid"

    def _extract_features_sync(self, image: np.ndarray) -> np.ndarray:
        preprocessed = self._preprocess_image(image)
        with tf.device(self.device):
            features = self.mobile_net.predict(preprocessed, verbose=0)
        return features.flatten()

    def _predict_xgboost_sync(self, features: np.ndarray) -> Tuple[float, str]:
        try:
            dmatrix = xgb.DMatrix(features.reshape(1, -1))
            prediction = self.xgboost_model.predict(dmatrix)[0]
            predicted_class = int(round(prediction)) if 0 <= prediction < len(self.jewelry_categories) else 0
            return 100.0, self.jewelry_categories[predicted_class]
        except Exception as e:
            logger.error(f"XGBoost prediction error: {e}")
            return 0.0, "Not Assigned"

    def _predict_mlp_sync(self, features: np.ndarray) -> Tuple[float, str]:
        try:
            proba = self.mlp_model.predict(features.reshape(1, -1), verbose=0)[0]
            predicted_class = int(np.argmax(proba))
            return float(proba[predicted_class] * 100), self.jewelry_categories[predicted_class]
        except Exception as e:
            logger.error(f"MLP prediction error: {e}")
            return 0.0, "Not Assigned"

    # --- async public methods (C-002: all blocking calls in to_thread) ---

    async def validate_images(self, face_image: np.ndarray, jewelry_image: np.ndarray, prediction_id: str) -> bool:
        client = get_db_client()
        if not client:
            logger.error("Database connection not available")
            return False
        db = client["jewelify"]
        col = db["predictions"]

        logger.info(f"Validating images for prediction {prediction_id}...")
        try:
            is_valid_face, face_msg = await asyncio.to_thread(self._validate_face_sync, face_image)
            if not is_valid_face:
                logger.warning(f"Face validation failed: {face_msg}")
                col.update_one({"_id": ObjectId(prediction_id)}, {"$set": {"validation_status": "failed"}})
                return False

            is_valid_jewelry, jewelry_msg = await asyncio.to_thread(self._validate_image_sync, jewelry_image)
            if not is_valid_jewelry:
                logger.warning(f"Jewelry validation failed: {jewelry_msg}")
                col.update_one({"_id": ObjectId(prediction_id)}, {"$set": {"validation_status": "failed"}})
                return False

            col.update_one({"_id": ObjectId(prediction_id)}, {"$set": {"validation_status": "completed"}})
            logger.info(f"Validation completed for prediction {prediction_id}")
            return True
        except Exception as e:
            logger.error(f"Validation error for {prediction_id}: {e}")
            col.update_one({"_id": ObjectId(prediction_id)}, {"$set": {"validation_status": "failed"}})
            raise

    def _build_recommendations(self, category: str, model_tag: str, top_k: int = 10) -> List[dict]:
        recs = []
        for i in range(top_k):
            file_name = f"Necklace with earrings_{i}.jpg"
            display_url = self.s3_base_url + urllib.parse.quote(file_name)
            recs.append({
                "name": f"{category}_{model_tag}_{i}",
                "category": category,
                "score": float(100 - i * 5),
                "display_url": display_url,
            })
        return recs

    async def predict_both(
        self,
        face_image: np.ndarray,
        jewelry_image: np.ndarray,
        face_image_path: str,
        jewelry_image_path: str,
        prediction_id: str,
    ) -> dict:
        client = get_db_client()
        if not client:
            logger.error("Database connection not available")
            return {}
        db = client["jewelify"]
        col = db["predictions"]

        logger.info(f"Starting prediction for {prediction_id}...")
        t = time.time()
        try:
            # C-002: feature extraction in thread pool
            face_features, jewelry_features = await asyncio.gather(
                asyncio.to_thread(self._extract_features_sync, face_image),
                asyncio.to_thread(self._extract_features_sync, jewelry_image),
            )

            if face_features is None or jewelry_features is None:
                logger.error("Feature extraction failed")
                col.update_one({"_id": ObjectId(prediction_id)}, {"$set": {"prediction_status": "failed"}})
                return {}

            combined = np.concatenate([face_features, jewelry_features])
            logger.info(f"Combined features shape: {combined.shape}")

            # C-002: model inference in thread pool
            xgb_result, mlp_result = await asyncio.gather(
                asyncio.to_thread(self._predict_xgboost_sync, combined),
                asyncio.to_thread(self._predict_mlp_sync, combined),
            )

            xgb_confidence, xgb_category = xgb_result
            mlp_confidence, mlp_category = mlp_result

            xgb_recs = self._build_recommendations(xgb_category, "xgboost")
            mlp_recs = self._build_recommendations(mlp_category, "mlp")

            prediction_result = {
                "prediction1": {
                    "score": float(xgb_confidence),
                    "category": xgb_category,
                    "recommendations": xgb_recs,
                    "feedback_required": True,
                    "overall_feedback": 0.5,
                },
                "prediction2": {
                    "score": float(mlp_confidence),
                    "category": mlp_category,
                    "recommendations": mlp_recs,
                    "feedback_required": True,
                    "overall_feedback": 0.5,
                },
                "face_image_path": face_image_path,
                "jewelry_image_path": jewelry_image_path,
            }

            col.update_one(
                {"_id": ObjectId(prediction_id)},
                {"$set": {
                    "prediction1": prediction_result["prediction1"],
                    "prediction2": prediction_result["prediction2"],
                    "face_image_path": face_image_path,
                    "jewelry_image_path": jewelry_image_path,
                    "prediction_status": "completed",
                }},
            )
            logger.info(f"Prediction {prediction_id} completed in {time.time() - t:.2f}s")
            return prediction_result
        except Exception as e:
            logger.error(f"Prediction error for {prediction_id}: {e}")
            col.update_one({"_id": ObjectId(prediction_id)}, {"$set": {"prediction_status": "failed"}})
            raise


if __name__ == "__main__":
    predictor = JewelryPredictor()
    image = cv2.imread("sample_face.jpg")
    jewelry_image = cv2.imread("sample_jewelry.jpg")
    t = time.time()
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(
        predictor.predict_both(image, jewelry_image, "sample_face.jpg", "sample_jewelry.jpg", "test_id")
    )
    print(result)
    print(f"Total time: {time.time() - t:.2f}s")
