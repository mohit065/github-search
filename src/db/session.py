import os
import time
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from src.config import settings

def get_engine():
    db_url = settings.POSTGRES_URL if settings.USE_POSTGRES else settings.DATABASE_URL
    if db_url.startswith("sqlite"):
        engine = create_engine(
            db_url, 
            connect_args={"check_same_thread": False},
            echo=False
        )
    else:
        # Retry connection for PostgreSQL container startup
        retries = 10
        delay = 2
        engine = None
        for i in range(retries):
            try:
                engine = create_engine(
                    db_url, 
                    pool_size=10, 
                    max_overflow=20, 
                    pool_pre_ping=True,
                    echo=False
                )
                with engine.connect() as conn:
                    pass
                break
            except Exception as e:
                print(f"[DB] Waiting for database to become available ({i+1}/{retries})...")
                time.sleep(delay)
                if i == retries - 1:
                    print(f"[DB] Could not connect to PostgreSQL ({e}). Falling back to engine initialization.")
        if engine is None:
            engine = create_engine(db_url, pool_pre_ping=True)
    return engine

engine = get_engine()
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
