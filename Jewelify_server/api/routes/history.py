from fastapi import APIRouter, Depends, HTTPException, Query
from api.dependencies import get_current_user, validate_object_id
from services.database import get_db_client
from bson import ObjectId
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/history", tags=["history"])


@router.get("/")
async def get_user_history(
    current_user: dict = Depends(get_current_user),
    limit: int = Query(10, ge=1, le=100),
    skip: int = Query(0, ge=0),
):
    client = get_db_client()
    if not client:
        raise HTTPException(status_code=500, detail="Database connection unavailable")

    user_oid = validate_object_id(str(current_user["_id"]))

    try:
        db = client["jewelify"]
        predictions_col = db["predictions"]
        reviews_col = db["reviews"]
        predictions = await predictions_col.find({"user_id": user_oid}).sort("timestamp", -1).skip(skip).limit(limit).to_list(length=limit)
    except Exception as e:
        logger.error(f"Database error fetching history: {e}")
        raise HTTPException(status_code=500, detail="Database error fetching history")

    if not predictions:
        return {"message": "No predictions found", "history": []}

    results = []
    for pred in predictions:
        prediction_id = str(pred["_id"])
        user_id_str = str(current_user["_id"])
        pred_oid = ObjectId(prediction_id)
        user_oid2 = ObjectId(user_id_str)

        try:
            rec_reviews = await reviews_col.find({
                "prediction_id": pred_oid,
                "user_id": user_oid2,
                "feedback_type": "recommendation",
            }).to_list(length=None)
            individual_feedback = {"prediction1": {}, "prediction2": {}}
            for review in rec_reviews:
                mt = review["model_type"]
                if mt in individual_feedback:
                    individual_feedback[mt][review["recommendation_name"]] = review["score"]

            pred_reviews = await reviews_col.find({
                "prediction_id": pred_oid,
                "user_id": user_oid2,
                "feedback_type": "prediction",
            }).to_list(length=None)
            overall_feedback = {"prediction1": None, "prediction2": None}
            for review in pred_reviews:
                mt = review["model_type"]
                if mt in overall_feedback:
                    overall_feedback[mt] = review["score"]

            overall_feedback["prediction1"] = float(overall_feedback["prediction1"]) if overall_feedback["prediction1"] is not None else 0.5
            overall_feedback["prediction2"] = float(overall_feedback["prediction2"]) if overall_feedback["prediction2"] is not None else 0.5

            def apply_feedback(recs, model_key):
                for rec in recs:
                    name = rec["name"]
                    if name in individual_feedback[model_key]:
                        rec["user_score"] = individual_feedback[model_key][name]
                        rec["liked"] = rec["user_score"] >= 0.75
                    elif overall_feedback[model_key] is not None:
                        rec["user_score"] = overall_feedback[model_key]
                        rec["liked"] = rec["user_score"] >= 0.75
                    else:
                        rec["user_score"] = None
                        rec["liked"] = rec.get("score", 0.0) >= 75.0
                return sorted(recs, key=lambda x: (x["liked"], x.get("score", 0.0)), reverse=True)[:10]

            xgb_recs = apply_feedback(pred.get("prediction1", {}).get("recommendations", []), "prediction1")
            mlp_recs = apply_feedback(pred.get("prediction2", {}).get("recommendations", []), "prediction2")

            results.append({
                "id": prediction_id,
                "user_id": user_id_str,
                "prediction1": {
                    "score": pred.get("prediction1", {}).get("score", 0.0),
                    "category": pred.get("prediction1", {}).get("category", "Not Assigned"),
                    "recommendations": xgb_recs,
                    "overall_feedback": overall_feedback["prediction1"],
                    "feedback_required": overall_feedback["prediction1"] == 0.5,
                },
                "prediction2": {
                    "score": pred.get("prediction2", {}).get("score", 0.0),
                    "category": pred.get("prediction2", {}).get("category", "Not Assigned"),
                    "recommendations": mlp_recs,
                    "overall_feedback": overall_feedback["prediction2"],
                    "feedback_required": overall_feedback["prediction2"] == 0.5,
                },
                "face_image_path": pred.get("face_image_path"),
                "jewelry_image_path": pred.get("jewelry_image_path"),
                "timestamp": pred["timestamp"],
            })
        except Exception as e:
            logger.error(f"Error processing prediction {prediction_id}: {e}")
            continue

    return results
