import os
import secrets
from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta
from dotenv import load_dotenv
from services.database import get_db_client
import logging

# Load environment variables
load_dotenv()

# Configure logging based on environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
logging_level = logging.DEBUG if ENVIRONMENT == "development" else logging.WARNING
logging.basicConfig(level=logging_level, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# JWT settings
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not JWT_SECRET_KEY:
    logger.error("JWT_SECRET_KEY environment variable not set")
    raise ValueError("JWT_SECRET_KEY environment variable not set")

JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
if JWT_ALGORITHM not in ["HS256", "HS384", "HS512"]:
    logger.error(f"Invalid JWT_ALGORITHM: {JWT_ALGORITHM}. Must be one of HS256, HS384, or HS512")
    raise ValueError(f"Invalid JWT_ALGORITHM: {JWT_ALGORITHM}. Must be one of HS256, HS384, or HS512")

ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 43200))

# OTP settings
OTP_LENGTH = 6
OTP_EXPIRY_MINUTES = 5

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    logger.debug("Hashing password for user")
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a hashed password."""
    logger.debug("Verifying password")
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    logger.debug(f"Created JWT token with expiration: {expire}")
    return encoded_jwt


def generate_otp(length: int = OTP_LENGTH) -> str:
    """Generate a cryptographically secure random OTP of specified length."""
    otp = str(secrets.randbelow(10 ** length)).zfill(length)
    logger.debug("Generated OTP")
    return otp


def send_otp_via_email(email: str, otp: str) -> bool:
    """Send OTP to the given email address."""
    from services.email import send_otp_email
    return send_otp_email(email, otp, otp_expiry_minutes=OTP_EXPIRY_MINUTES)


def store_otp(email: str, otp: str) -> bool:
    """Store the OTP in MongoDB with an expiration time."""
    client = get_db_client()
    if not client:
        logger.error("Cannot store OTP: No MongoDB client available")
        return False

    try:
        db = client["jewelify"]
        otps_collection = db["otps"]
        expiry = datetime.utcnow() + timedelta(minutes=OTP_EXPIRY_MINUTES)
        otp_doc = {
            "email": email,
            "otp": otp,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": expiry
        }
        otps_collection.insert_one(otp_doc)
        logger.info(f"OTP stored for {email}")
        return True
    except Exception as e:
        logger.error(f"Error storing OTP for {email}: {str(e)}")
        return False


def verify_otp(email: str, otp: str) -> bool:
    """Verify the OTP for the given email address."""
    client = get_db_client()
    if not client:
        logger.error("Cannot verify OTP: No MongoDB client available")
        return False

    try:
        db = client["jewelify"]
        otps_collection = db["otps"]
        otp_doc = otps_collection.find_one(
            {"email": email, "otp": otp},
            sort=[("created_at", -1)]  # Get the most recent OTP
        )
        if not otp_doc:
            logger.warning(f"No OTP found for {email} or OTP does not match")
            return False

        expiry = otp_doc["expires_at"]
        # Handle both datetime objects and ISO strings for backwards compatibility
        if isinstance(expiry, str):
            expiry = datetime.fromisoformat(expiry)

        if datetime.utcnow() > expiry:
            logger.warning(f"OTP for {email} has expired")
            return False

        # OTP is valid, delete it from the database
        otps_collection.delete_one({"_id": otp_doc["_id"]})
        logger.info(f"OTP verified for {email}")
        return True
    except Exception as e:
        logger.error(f"Error verifying OTP for {email}: {str(e)}")
        return False
