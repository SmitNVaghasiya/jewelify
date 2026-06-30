from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from bson import ObjectId
from bson.errors import InvalidId
from services.database import get_db_client
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

# Load JWT settings from environment
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    logger.error("JWT_SECRET_KEY environment variable not set")
    raise ValueError("JWT_SECRET_KEY environment variable not set")

ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
if ALGORITHM not in ["HS256", "HS384", "HS512"]:
    logger.error(f"Invalid JWT_ALGORITHM: {ALGORITHM}. Must be one of HS256, HS384, or HS512")
    raise ValueError(f"Invalid JWT_ALGORITHM: {ALGORITHM}. Must be one of HS256, HS384, or HS512")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


def validate_object_id(id_str: str) -> ObjectId:
    """Validate and convert a string to a MongoDB ObjectId.

    Raises HTTP 400 if the string is not a valid ObjectId.
    """
    try:
        return ObjectId(id_str)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid ID format")


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    logger.debug(f"Received token for validation: {token[:10]}... (truncated for security)")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            logger.warning("JWT token missing 'sub' field")
            raise credentials_exception
        logger.debug(f"Successfully decoded JWT token. User ID: {user_id}")
    except JWTError as e:
        logger.error(f"Failed to decode JWT token: {str(e)}")
        raise credentials_exception from e

    try:
        client = get_db_client()
        if not client:
            logger.error("Failed to get MongoDB client connection")
            raise HTTPException(status_code=500, detail="Failed to connect to the database")
        db = client["jewelify"]
        user = await db["users"].find_one({"_id": validate_object_id(user_id)})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Database error while fetching user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error while fetching user: {str(e)}")

    if user is None:
        logger.warning(f"User with ID {user_id} not found in the database")
        raise credentials_exception

    logger.info(f"User {user_id} authenticated successfully")
    return user
