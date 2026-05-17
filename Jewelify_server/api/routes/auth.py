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

# Load environment variables
load_dotenv()

# Configure logging based on environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
logging_level = logging.DEBUG if ENVIRONMENT == "development" else logging.WARNING
logging.basicConfig(level=logging_level, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/send-otp")
@limiter.limit("3/minute")
async def send_otp(request: Request, body: OtpRequest):
    """Generate and send OTP to the user's email address."""
    email = body.email
    logger.info(f"Request to send OTP to {email}")
    try:
        otp = generate_otp()
        if not store_otp(email, otp):
            logger.error(f"Failed to store OTP for {email}")
            raise HTTPException(status_code=500, detail="Failed to store OTP")
        if not send_otp_via_email(email, otp):
            logger.error(f"Failed to send OTP email to {email}")
            raise HTTPException(status_code=500, detail="Failed to send OTP email")
        logger.info(f"OTP sent to {email}")
        return {"message": f"OTP sent to {email}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending OTP to {email}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process OTP request: {str(e)}")


@router.post("/verify-otp")
@limiter.limit("5/minute")
async def verify_otp_endpoint(request: Request, body: OtpVerify):
    """Verify the OTP provided by the user."""
    email = body.email
    otp = body.otp
    logger.info(f"Verifying OTP for {email}")
    try:
        if not verify_otp(email, otp):
            logger.warning(f"Invalid or expired OTP for {email}")
            raise HTTPException(status_code=400, detail="Invalid or expired OTP")
        logger.info(f"OTP verified successfully for {email}")
        return {"message": "OTP verified", "verified": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying OTP for {email}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to verify OTP: {str(e)}")


@router.post("/register", response_model=UserOut)
async def register(user: UserRegister):
    """Register a new user with name, username, email, and password."""
    logger.info(f"Registering new user with username '{user.username}' and email '{user.email}'")
    try:
        client = get_db_client()
        if not client:
            logger.error("Database connection not available")
            raise HTTPException(status_code=500, detail="Database connection unavailable")
        db = client["jewelify"]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

    if db["users"].find_one({"username": user.username}):
        logger.warning(f"Username '{user.username}' already exists")
        raise HTTPException(status_code=400, detail="Username already exists")
    if db["users"].find_one({"email": user.email}):
        logger.warning(f"Email '{user.email}' already exists")
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
        result = db["users"].insert_one(user_data)
        user_id = str(result.inserted_id)
        access_token = create_access_token(data={"sub": user_id})
    except Exception as e:
        logger.error(f"Error registering user '{user.username}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create user: {str(e)}")

    logger.info(f"User {user_id} registered successfully")
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
    """Login a user using email and password."""
    logger.info(f"Login attempt for {credentials.email}")
    try:
        client = get_db_client()
        if not client:
            logger.error("Database connection not available")
            raise HTTPException(status_code=500, detail="Database connection unavailable")
        db = client["jewelify"]
        user = db["users"].find_one({"email": credentials.email})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Database error during login for {credentials.email}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    if not user or not verify_password(credentials.password, user["hashed_password"]):
        logger.warning(f"Failed login attempt for {credentials.email}: incorrect email or password")
        raise HTTPException(status_code=400, detail="Incorrect email or password")

    access_token = create_access_token(data={"sub": str(user["_id"])})
    logger.info(f"User {user['_id']} logged in successfully")
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
    """Send a password-reset OTP to the user's email if the account exists."""
    email = request.email
    logger.info(f"Forgot-password request for {email}")
    try:
        client = get_db_client()
        if not client:
            logger.error("Database connection not available")
            raise HTTPException(status_code=500, detail="Database connection unavailable")
        db = client["jewelify"]
        user = db["users"].find_one({"email": email})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Database error during forgot-password for {email}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    if not user:
        # Return a generic message to avoid user enumeration
        logger.warning(f"Forgot-password request for non-existent email: {email}")
        return {"message": "Password reset OTP sent"}

    try:
        otp = generate_otp()
        if not store_otp(email, otp):
            logger.error(f"Failed to store reset OTP for {email}")
            raise HTTPException(status_code=500, detail="Failed to store OTP")
        if not send_otp_via_email(email, otp):
            logger.error(f"Failed to send reset OTP email to {email}")
            raise HTTPException(status_code=500, detail="Failed to send OTP email")
        logger.info(f"Password reset OTP sent to {email}")
        return {"message": "Password reset OTP sent"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing forgot-password for {email}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process request: {str(e)}")


@router.post("/reset-password")
async def reset_password(request: ResetPasswordRequest):
    """Verify OTP and update the user's password."""
    email = request.email
    logger.info(f"Password reset attempt for {email}")
    try:
        if not verify_otp(email, request.otp):
            logger.warning(f"Invalid or expired reset OTP for {email}")
            raise HTTPException(status_code=400, detail="Invalid or expired OTP")

        client = get_db_client()
        if not client:
            logger.error("Database connection not available")
            raise HTTPException(status_code=500, detail="Database connection unavailable")
        db = client["jewelify"]

        hashed_password = hash_password(request.new_password)
        result = db["users"].update_one(
            {"email": email},
            {"$set": {"hashed_password": hashed_password}}
        )
        if result.matched_count == 0:
            logger.warning(f"No user found for email {email} during password reset")
            raise HTTPException(status_code=404, detail="User not found")

        logger.info(f"Password reset successful for {email}")
        return {"message": "Password reset successful"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting password for {email}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reset password: {str(e)}")


@router.get("/check-user/{email}")
async def check_user(email: str):
    """Check if a user exists by email address."""
    logger.info(f"Checking if user with email {email} exists")
    try:
        client = get_db_client()
        if not client:
            logger.error("Database connection not available")
            raise HTTPException(status_code=500, detail="Database connection unavailable")
        db = client["jewelify"]
        user = db["users"].find_one({"email": email.lower()})
        exists = bool(user)
        logger.debug(f"User with email {email} exists: {exists}")
        return {"exists": exists}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking user with email {email}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
