import os
from sqlalchemy import text
from src.config import settings
from src.db.session import engine
from src.db.models import Base

def init_database():
    """Create tables and initialize PostgreSQL extensions / indexes if applicable."""
    print(f"[DB Init] Initializing database with engine: {engine.url}")
    
    if settings.USE_POSTGRES:
        try:
            with engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))
                conn.commit()
                print("[DB Init] PostgreSQL vector and pg_trgm extensions verified.")
        except Exception as e:
            print(f"[DB Init] Warning enabling Postgres extensions: {e}")
            
    Base.metadata.create_all(bind=engine)
    print("[DB Init] Tables initialized successfully.")

if __name__ == "__main__":
    init_database()
