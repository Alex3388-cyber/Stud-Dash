from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel, EmailStr
from database.models import UserRole


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshRequest(BaseModel):
    refresh_token: str


class UserCreate(BaseModel):
    email: str
    password: str
    full_name: str | None = None
    role: UserRole = UserRole.viewer


class UserOut(BaseModel):
    id: int
    email: str
    full_name: str | None
    role: UserRole
    is_active: bool
    created_at: datetime
    last_login: datetime | None

    model_config = {"from_attributes": True}
