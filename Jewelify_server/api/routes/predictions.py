from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request
import cv2
import numpy as np
from typing import Optional
from datetime import datetime
import logging
import os
from dotenv import load_dotenv
from api.dependencies import get_current_user, validate_object_id
from services.database import get_db_client
import asyncio
from bson import ObjectId

load_dotenv()

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
logging_level = logging.DEBUG if ENVIRONMENT == "development" else logging.WARNING
logging.basicConfig(level=logging_level, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("pymongo").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predictions", tags=["predictions"])

POLLING_TIMEOUT_SECONDS = int(os.getenv("POLLING_TIMEOUT_SECONDS", 60))
POLLING_INTERVAL_SECONDS = int(os.getenv("POLLING_INTERVAL_SECONDS", 5))


async def save_prediction_to_db(prediction_data: dict, user_id: str) -> str:
    client = get_db_client()
    if not client:
        raise HTTPException(status_code=500, detail="Database connection unavailable")
    try:
        prediction_data["user_id"] = ObjectId(user_id)
        prediction_data["timestamp"] = datetime.utcnow().isoformat()
        prediction_data["validation_status"] = "pending"
        prediction_data["prediction_status"] = "pending"
        result = await client["jewelify"]["predictions"].insert_one(prediction_data)
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"Failed to save prediction: {e}")
        raise HTTPException(status_code=500, detail="Database error saving prediction")


@router.post("/predict")
async def predict(
    request: Request,
    face: UploadFile = File(...),
    jewelry: UploadFile = File(...),
    face_image_path: str = Form(...),
    jewelry_image_path: str = Form(...),
    user: dict = Depends(get_current_user),
):
    logger.info(f"POST /predictions/predict user={user.get('_id')}")

    # C-001: use cached predictor from app.state
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not available")

    try:
        if "_id" not in user:
            raise HTTPException(status_code=401, detail="Invalid token: User ID not found")

        # Read and decode images (async)
        face_contents, jewelry_contents = await asyncio.gather(face.read(), jewelry.read())

        face_image = await asyncio.to_thread(
            cv2.imdecode, np.frombuffer(face_contents, np.uint8), cv2.IMREAD_COLOR
        )
        if face_image is None:
            raise HTTPException(status_code=400, detail="Invalid face image format")

        jewelry_image = await asyncio.to_thread(
            cv2.imdecode, np.frombuffer(jewelry_contents, np.uint8), cv2.IMREAD_COLOR
        )
        if jewelry_image is None:
            raise HTTPException(status_code=400, detail="Invalid jewelry image format")

        prediction_data = {
            "prediction1": {"score": 0.0, "category": "Not Assigned", "recommendations": [], "feedback_required": True, "overall_feedback": 0.5},
            "prediction2": {"score": 0.0, "category": "Not Assigned", "recommendations": [], "feedback_required": True, "overall_feedback": 0.5},
            "face_image_path": face_image_path,
            "jewelry_image_path": jewelry_image_path,
        }

        prediction_id = await save_prediction_to_db(prediction_data, str(user["_id"]))

        validation_task = asyncio.create_task(
            predictor.validate_images(face_image, jewelry_image, prediction_id)
        )
        prediction_task = asyncio.create_task(
            predictor.predict_both(face_image, jewelry_image, face_image_path, jewelry_image_path, prediction_id)
        )
        results = await asyncio.gather(validation_task, prediction_task, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} failed: {result}")
                raise HTTPException(status_code=500, detail="Processing failed")

        if not results[0]:
            raise HTTPException(status_code=400, detail="Validation failed: Ensure a face is visible and jewelry is clear.")

        return {"status": "success", "prediction_id": prediction_id, "message": "Prediction completed successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(status_code=500, detail="An error occurred processing your request")


@router.get("/get_prediction/{prediction_id}")
async def get_prediction(prediction_id: str, user: dict = Depends(get_current_user)):
    client = get_db_client()
    if not client:
        raise HTTPException(status_code=500, detail="Database connection unavailable")

    user_oid = validate_object_id(str(user["_id"]))
    pred_oid = validate_object_id(prediction_id)

    try:
        db = client["jewelify"]
        col = db["predictions"]
        prediction = await col.find_one({"_id": pred_oid, "user_id": user_oid})
        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction not found")

        start_wait = datetime.utcnow()
        while True:
            v_status = prediction.get("validation_status", "pending")
            p_status = prediction.get("prediction_status", "pending")
            if v_status != "pending" and p_status != "pending":
                break
            if (datetime.utcnow() - start_wait).total_seconds() >= POLLING_TIMEOUT_SECONDS:
                raise HTTPException(status_code=408, detail="Request timed out waiting for prediction to complete")
            await asyncio.sleep(POLLING_INTERVAL_SECONDS)
            prediction = await col.find_one({"_id": pred_oid, "user_id": user_oid})

        if prediction.get("validation_status") == "failed":
            raise HTTPException(status_code=400, detail="Validation failed")
        if prediction.get("prediction_status") == "failed":
            raise HTTPException(status_code=500, detail="Prediction failed")

        p1 = prediction.get("prediction1", {})
        p2 = prediction.get("prediction2", {})

        return {
            "prediction_id": str(prediction["_id"]),
            "user_id": str(prediction["user_id"]),
            "prediction1": {
                "score": p1.get("score", 0.0),
                "category": p1.get("category", "Not Assigned"),
                "recommendations": p1.get("recommendations", []),
                "overall_feedback": float(p1.get("overall_feedback") or 0.5),
                "feedback_required": p1.get("feedback_required", True),
            },
            "prediction2": {
                "score": p2.get("score", 0.0),
                "category": p2.get("category", "Not Assigned"),
                "recommendations": p2.get("recommendations", []),
                "overall_feedback": float(p2.get("overall_feedback") or 0.5),
                "feedback_required": p2.get("feedback_required", True),
            },
            "face_image_path": prediction.get("face_image_path"),
            "jewelry_image_path": prediction.get("jewelry_image_path"),
            "timestamp": prediction.get("timestamp"),
            "validation_status": prediction.get("validation_status"),
            "prediction_status": prediction.get("prediction_status"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching prediction {prediction_id}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred fetching your prediction")


@router.post("/feedback/{feedback_type}")
async def submit_feedback(
    feedback_type: str,
    prediction_id: str = Form(...),
    model_type: str = Form(...),
    recommendation_name: Optional[str] = Form(None),
    score: str = Form(...),
    user: dict = Depends(get_current_user),
):
    client = get_db_client()
    if not client:
        raise HTTPException(status_code=500, detail="Database connection unavailable")

    if feedback_type not in ["prediction", "recommendation"]:
        raise HTTPException(status_code=400, detail="Invalid feedback type")
    if model_type not in ["prediction1", "prediction2"]:
        raise HTTPException(status_code=400, detail="Invalid model type")

    try:
        score_float = float(score)
        if not 0 <= score_float <= 1:
            raise ValueError()
    except ValueError:
        raise HTTPException(status_code=400, detail="Score must be a float between 0 and 1")

    user_oid = validate_object_id(str(user["_id"]))
    pred_oid = validate_object_id(prediction_id)

    try:
        db = client["jewelify"]
        prediction = await db["predictions"].find_one({"_id": pred_oid, "user_id": user_oid})
        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction not found")

        from services.database import save_review
        success = await save_review(
            user_id=str(user["_id"]),
            prediction_id=prediction_id,
            model_type=model_type,
            recommendation_name=recommendation_name,
            score=score_float,
            feedback_type=feedback_type,
        )
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save feedback")

        if feedback_type == "prediction":
            await db["predictions"].update_one(
                {"_id": pred_oid},
                {"$set": {f"{model_type}.feedback_required": False}},
            )

        return {"message": "Feedback submitted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail="An error occurred submitting feedback")
