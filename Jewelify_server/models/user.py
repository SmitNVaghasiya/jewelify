# models/user.py
import re
from bson import ObjectId
from pydantic import BaseModel, Field, field_validator
from typing import Optional


class UserRegister(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(...)
    password: str = Field(...)

    @field_validator("email")
    def validate_email(cls, v):
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", v):
            raise ValueError("Invalid email address format")
        return v.lower()

    @field_validator("password")
    def validate_password_strength(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one digit")
        if not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]", v):
            raise ValueError("Password must contain at least one special character")
        return v


class UserLogin(BaseModel):
    email: str = Field(...)
    password: str = Field(...)

    @field_validator("email")
    def validate_email(cls, v):
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", v):
            raise ValueError("Invalid email address format")
        return v.lower()


class UserOut(BaseModel):
    id: str
    name: str
    username: str
    email: str
    created_at: str
    access_token: Optional[str] = None

    class Config:
        from_attributes = True
        populate_by_name = True
        json_encoders = {ObjectId: str}


class OtpRequest(BaseModel):
    email: str = Field(...)

    @field_validator("email")
    def validate_email(cls, v):
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", v):
            raise ValueError("Invalid email address format")
        return v.lower()


class OtpVerify(BaseModel):
    email: str = Field(...)
    otp: str = Field(..., min_length=6, max_length=6)

    @field_validator("email")
    def validate_email(cls, v):
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", v):
            raise ValueError("Invalid email address format")
        return v.lower()


class ForgotPasswordRequest(BaseModel):
    email: str = Field(...)

    @field_validator("email")
    def validate_email(cls, v):
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", v):
            raise ValueError("Invalid email address format")
        return v.lower()


class ResetPasswordRequest(BaseModel):
    email: str = Field(...)
    otp: str = Field(..., min_length=6, max_length=6)
    new_password: str = Field(...)

    @field_validator("email")
    def validate_email(cls, v):
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", v):
            raise ValueError("Invalid email address format")
        return v.lower()

    @field_validator("new_password")
    def validate_password_strength(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one digit")
        if not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]", v):
            raise ValueError("Password must contain at least one special character")
        return v
