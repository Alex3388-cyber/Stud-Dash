"""Auth router — login, refresh, logout, register, user management."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from api.deps import get_db, get_current_user, require_admin
from api.schemas.auth import LoginRequest, TokenResponse, UserCreate, UserOut, RefreshRequest
from config import get_settings
from database.models import User, UserRole
from database.pg_operations import (
    create_session,
    create_user,
    get_user_by_email,
    get_session_by_token,
    list_users,
    revoke_session,
    revoke_all_user_sessions,
    update_last_login,
    write_audit_log,
)

try:
    from jose import jwt
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
except ImportError:
    jwt = None
    pwd_context = None

router = APIRouter(prefix="/auth", tags=["auth"])


def _hash_password(password: str) -> str:
    if pwd_context is None:
        raise HTTPException(status_code=500, detail="Auth library not installed")
    return pwd_context.hash(password)


def _verify_password(plain: str, hashed: str) -> bool:
    if pwd_context is None:
        return False
    return pwd_context.verify(plain, hashed)


def _create_access_token(user_id: int, expires_delta: timedelta) -> str:
    settings = get_settings()
    payload = {"sub": str(user_id), "exp": datetime.now(timezone.utc) + expires_delta}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


@router.post("/register", response_model=UserOut, status_code=201)
def register(body: UserCreate, db: Session = Depends(get_db)):
    if get_user_by_email(db, body.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    user = create_user(db, body.email, _hash_password(body.password), body.full_name, body.role)
    write_audit_log(db, action="REGISTER", user_email=body.email, resource_type="user")
    return user


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, request: Request, db: Session = Depends(get_db)):
    user = get_user_by_email(db, body.email)
    if not user or not _verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")

    settings = get_settings()
    access_expires = timedelta(minutes=settings.access_token_expire_minutes)
    refresh_expires = timedelta(days=settings.refresh_token_expire_days)

    access_token = _create_access_token(user.id, access_expires)
    refresh_token = _create_access_token(user.id, refresh_expires)
    expires_at = datetime.now(timezone.utc) + access_expires

    create_session(db, user.id, access_token, refresh_token, expires_at)
    update_last_login(db, user.id)
    write_audit_log(
        db, action="LOGIN", user_id=user.id, user_email=user.email,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        response_status=200,
    )
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=int(access_expires.total_seconds()),
    )


@router.post("/refresh", response_model=TokenResponse)
def refresh(body: RefreshRequest, db: Session = Depends(get_db)):
    settings = get_settings()
    try:
        payload = jwt.decode(body.refresh_token, settings.secret_key, algorithms=[settings.algorithm])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    user_id = int(payload.get("sub", 0))
    access_expires = timedelta(minutes=settings.access_token_expire_minutes)
    new_access = _create_access_token(user_id, access_expires)
    refresh_expires = timedelta(days=settings.refresh_token_expire_days)
    new_refresh = _create_access_token(user_id, refresh_expires)
    expires_at = datetime.now(timezone.utc) + access_expires

    create_session(db, user_id, new_access, new_refresh, expires_at)
    return TokenResponse(access_token=new_access, refresh_token=new_refresh, expires_in=int(access_expires.total_seconds()))


@router.post("/logout", status_code=204)
def logout(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    write_audit_log(db, action="LOGOUT", user_id=current_user.id, user_email=current_user.email)


@router.get("/me", response_model=UserOut)
def me(current_user: User = Depends(get_current_user)):
    return current_user


@router.get("/users", response_model=list[UserOut])
def get_users(db: Session = Depends(get_db), _admin: User = Depends(require_admin)):
    return list_users(db)
