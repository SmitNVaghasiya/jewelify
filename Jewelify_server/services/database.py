from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING
import os
from datetime import datetime
from dotenv import load_dotenv
import logging
from bson import ObjectId

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
logging_level = logging.DEBUG if ENVIRONMENT == "development" else logging.WARNING
logging.basicConfig(level=logging_level)
logger = logging.getLogger(__name__)

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
client = None


def get_db_client():
    global client
    if client is None:
        if not MONGO_URI:
            logger.error("MONGO_URI not set")
            return None
        client = AsyncIOMotorClient(
            MONGO_URI,
            maxPoolSize=50,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
        )
        logger.info("Motor client created")
    return client


async def ensure_indexes():
    client = get_db_client()
    if not client:
        logger.warning("ensure_indexes: No MongoDB client, skipping")
        return
    try:
        db = client["jewelify"]
        await db["users"].create_index("username", unique=True)
        await db["users"].create_index("email", unique=True)
        await db["otps"].create_index("email")
        await db["otps"].create_index("expires_at", expireAfterSeconds=0)
        await db["predictions"].create_index([("user_id", ASCENDING), ("timestamp", DESCENDING)])
        await db["reviews"].create_index([("prediction_id", ASCENDING), ("user_id", ASCENDING)])
        logger.info("MongoDB indexes ensured")
    except Exception as e:
        logger.error(f"Error ensuring indexes: {e}")


async def save_prediction(prediction_data: dict, user_id: str):
    client = get_db_client()
    if not client:
        logger.error("No MongoDB client, cannot save prediction")
        return None
    try:
        db = client["jewelify"]
        prediction = {
            "user_id": ObjectId(user_id),
            "prediction1": prediction_data["prediction1"],
            "prediction2": prediction_data["prediction2"],
            "face_image_path": prediction_data["face_image_path"],
            "jewelry_image_path": prediction_data["jewelry_image_path"],
            "timestamp": datetime.utcnow().isoformat(),
            "validation_status": "pending",
            "prediction_status": "pending",
        }
        result = await db["predictions"].insert_one(prediction)
        logger.info(f"Saved prediction: {result.inserted_id}")
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"Error saving prediction: {e}")
        return None


async def get_prediction_by_id(prediction_id, user_id):
    client = get_db_client()
    if not client:
        return {"error": "Database connection error"}
    try:
        db = client["jewelify"]
        predictions_collection = db["predictions"]
        reviews_collection = db["reviews"]

        prediction = await predictions_collection.find_one({
            "_id": ObjectId(prediction_id),
            "user_id": ObjectId(user_id),
        })
        if not prediction:
            return {"error": "Prediction not found"}

        prediction["prediction1"] = prediction.get("prediction1", {})
        prediction["prediction2"] = prediction.get("prediction2", {})

        recommendation_reviews = await reviews_collection.find({
            "prediction_id": ObjectId(prediction_id),
            "user_id": ObjectId(user_id),
            "feedback_type": "recommendation",
        }).to_list(length=None)
        individual_feedback = {"prediction1": {}, "prediction2": {}}
        for review in recommendation_reviews:
            mt = review["model_type"]
            if mt in individual_feedback:
                individual_feedback[mt][review["recommendation_name"]] = review["score"]

        prediction_reviews = await reviews_collection.find({
            "prediction_id": ObjectId(prediction_id),
            "user_id": ObjectId(user_id),
            "feedback_type": "prediction",
        }).to_list(length=None)
        overall_feedback = {"prediction1": None, "prediction2": None}
        for review in prediction_reviews:
            mt = review["model_type"]
            if mt in overall_feedback:
                overall_feedback[mt] = review["score"]

        overall_feedback["prediction1"] = float(overall_feedback["prediction1"]) if overall_feedback["prediction1"] is not None else 0.5
        overall_feedback["prediction2"] = float(overall_feedback["prediction2"]) if overall_feedback["prediction2"] is not None else 0.5

        liked = {"prediction1": [], "prediction2": []}
        for model_type in ["prediction1", "prediction2"]:
            for rec in prediction[model_type].get("recommendations", []):
                name = rec["name"]
                if name in individual_feedback[model_type]:
                    if individual_feedback[model_type][name] >= 0.75:
                        liked[model_type].append(name)
                elif overall_feedback[model_type] >= 0.75:
                    liked[model_type].append(name)

        for model_type in ["prediction1", "prediction2"]:
            for rec in prediction[model_type].get("recommendations", []):
                rec["liked"] = rec["name"] in liked[model_type]

        return {
            "id": str(prediction["_id"]),
            "user_id": str(prediction["user_id"]),
            "prediction1": {
                "score": prediction["prediction1"].get("score", 0.0),
                "category": prediction["prediction1"].get("category", "Not Assigned"),
                "recommendations": prediction["prediction1"].get("recommendations", []),
                "overall_feedback": overall_feedback["prediction1"],
                "feedback_required": prediction["prediction1"].get("feedback_required", True),
            },
            "prediction2": {
                "score": prediction["prediction2"].get("score", 0.0),
                "category": prediction["prediction2"].get("category", "Not Assigned"),
                "recommendations": prediction["prediction2"].get("recommendations", []),
                "overall_feedback": overall_feedback["prediction2"],
                "feedback_required": prediction["prediction2"].get("feedback_required", True),
            },
            "face_image_path": prediction.get("face_image_path"),
            "jewelry_image_path": prediction.get("jewelry_image_path"),
            "timestamp": prediction["timestamp"],
            "validation_status": prediction.get("validation_status", "pending"),
            "prediction_status": prediction.get("prediction_status", "pending"),
        }
    except Exception as e:
        logger.error(f"Error retrieving prediction: {e}")
        return {"error": f"Database error: {str(e)}"}


async def save_review(user_id, prediction_id, model_type, recommendation_name, score, feedback_type):
    client = get_db_client()
    if not client:
        logger.error("No MongoDB client, cannot save review")
        return False
    try:
        db = client["jewelify"]
        review_doc = {
            "user_id": ObjectId(user_id),
            "prediction_id": ObjectId(prediction_id),
            "model_type": model_type,
            "recommendation_name": recommendation_name,
            "score": float(score),
            "feedback_type": feedback_type,
            "timestamp": datetime.utcnow().isoformat(),
        }
        result = await db["reviews"].insert_one(review_doc)
        logger.info(f"Saved review: {result.inserted_id}")
        return True
    except Exception as e:
        logger.error(f"Error saving review: {e}")
        return False
