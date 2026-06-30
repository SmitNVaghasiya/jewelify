from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from services.database import get_db_client

limiter = Limiter(key_func=get_remote_address)
from services.auth import (
    hash_password,
    create_access_token,
    verify_password,
    generate_otp,
    send_otp_via_email,
    store_otp,
    verify_otp,
)
from models.user import (
    UserRegister,
    UserLogin,
    UserOut,
    OtpRequest,
    OtpVerify,
    ForgotPasswordRequest,
    ResetPasswordRequest,
)
from datetime import datetime
import os
from dotenv import load_dotenv
import logging

load_dotenv()

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
logging_level = logging.DEBUG if ENVIRONMENT == "development" else logging.WARNING
logging.basicConfig(level=logging_level, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/send-otp")
@limiter.limit("3/minute")
async def send_otp(request: Request, body: OtpRequest):
    email = body.email
    logger.info(f"Request to send OTP to {email}")
    try:
        otp = generate_otp()
        if not await store_otp(email, otp):
            raise HTTPException(status_code=500, detail="Failed to store OTP")
        if not send_otp_via_email(email, otp):
            raise HTTPException(status_code=500, detail="Failed to send OTP email")
        return {"message": f"OTP sent to {email}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending OTP to {email}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process OTP request: {str(e)}")


@router.post("/verify-otp")
@limiter.limit("5/minute")
async def verify_otp_endpoint(request: Request, body: OtpVerify):
    email = body.email
    logger.info(f"Verifying OTP for {email}")
    try:
        if not await verify_otp(email, body.otp):
            raise HTTPException(status_code=400, detail="Invalid or expired OTP")
        return {"message": "OTP verified", "verified": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying OTP for {email}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to verify OTP: {str(e)}")


@router.post("/register", response_model=UserOut)
async def register(user: UserRegister):
    logger.info(f"Registering user '{user.username}' / '{user.email}'")
    try:
        client = get_db_client()
        if not client:
            raise HTTPException(status_code=500, detail="Database connection unavailable")
        db = client["jewelify"]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

    if await db["users"].find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already exists")
    if await db["users"].find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already exists")

    hashed_password = hash_password(user.password)
    user_data = {
        "name": user.name,
        "username": user.username,
        "email": user.email,
        "hashed_password": hashed_password,
        "created_at": datetime.utcnow().isoformat(),
    }

    try:
        result = await db["users"].insert_one(user_data)
        user_id = str(result.inserted_id)
        access_token = create_access_token(data={"sub": user_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create user: {str(e)}")

    logger.info(f"User {user_id} registered")
    return UserOut(
        id=user_id,
        name=user.name,
        username=user.username,
        email=user.email,
        created_at=user_data["created_at"],
        access_token=access_token,
    )


@router.post("/login", response_model=UserOut)
@limiter.limit("5/minute")
async def login(request: Request, credentials: UserLogin):
    logger.info(f"Login attempt for {credentials.email}")
    try:
        client = get_db_client()
        if not client:
            raise HTTPException(status_code=500, detail="Database connection unavailable")
        db = client["jewelify"]
        user = await db["users"].find_one({"email": credentials.email})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    if not user or not verify_password(credentials.password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Incorrect email or password")

    access_token = create_access_token(data={"sub": str(user["_id"])})
    logger.info(f"User {user['_id']} logged in")
    return UserOut(
        id=str(user["_id"]),
        name=user.get("name", ""),
        username=user["username"],
        email=user["email"],
        created_at=user["created_at"],
        access_token=access_token,
    )


@router.post("/forgot-password")
async def forgot_password(request: ForgotPasswordRequest):
    email = request.email
    logger.info(f"Forgot-password request for {email}")
    try:
        client = get_db_client()
        if not client:
            raise HTTPException(status_code=500, detail="Database connection unavailable")
        db = client["jewelify"]
        user = await db["users"].find_one({"email": email})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    if not user:
        return {"message": "Password reset OTP sent"}

    try:
        otp = generate_otp()
        if not await store_otp(email, otp):
            raise HTTPException(status_code=500, detail="Failed to store OTP")
        if not send_otp_via_email(email, otp):
            raise HTTPException(status_code=500, detail="Failed to send OTP email")
        return {"message": "Password reset OTP sent"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process request: {str(e)}")


@router.post("/reset-password")
async def reset_password(request: ResetPasswordRequest):
    email = request.email
    logger.info(f"Password reset attempt for {email}")
    try:
        if not await verify_otp(email, request.otp):
            raise HTTPException(status_code=400, detail="Invalid or expired OTP")

        client = get_db_client()
        if not client:
            raise HTTPException(status_code=500, detail="Database connection unavailable")
        db = client["jewelify"]

        hashed_password = hash_password(request.new_password)
        result = await db["users"].update_one(
            {"email": email},
            {"$set": {"hashed_password": hashed_password}},
        )
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="User not found")

        logger.info(f"Password reset for {email}")
        return {"message": "Password reset successful"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset password: {str(e)}")


@router.get("/check-user/{email}")
async def check_user(email: str):
    logger.info(f"Checking user {email}")
    try:
        client = get_db_client()
        if not client:
            raise HTTPException(status_code=500, detail="Database connection unavailable")
        db = client["jewelify"]
        user = await db["users"].find_one({"email": email.lower()})
        return {"exists": bool(user)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
