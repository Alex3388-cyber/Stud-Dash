"""Database engine, session factory, and lifecycle helpers.

Works with SQLite (dev) and PostgreSQL (prod) transparently via DATABASE_URL.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from config import get_settings
from database.models import Base


def _build_engine():
    settings = get_settings()
    url = settings.database_url
    kwargs: dict = {"echo": False}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["pool_size"] = 10
        kwargs["max_overflow"] = 20
        kwargs["pool_pre_ping"] = True
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def enable_foreign_keys(conn, _):
            conn.execute("PRAGMA foreign_keys=ON")
    return engine


_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = _build_engine()
    return _engine


def get_session_factory() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal


def init_db() -> None:
    """Create all tables when they do not already exist."""
    Base.metadata.create_all(bind=get_engine())


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Yield a database session and always close it after use."""
    factory = get_session_factory()
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db_session() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session."""
    with get_db() as session:
        yield session
