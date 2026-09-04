import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    PROJECT_NAME: str = "GitHub Semantic Search Engine"
    API_V1_PREFIX: str = "/api"
    
    # Storage & DB Config
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        f"sqlite:///{BASE_DIR}/data/github_search.db"
    )
    POSTGRES_URL: str = os.getenv(
        "POSTGRES_URL", 
        "postgresql://postgres:postgres@localhost:5432/github_search"
    )
    USE_POSTGRES: bool = os.getenv("USE_POSTGRES", "false").lower() in ("true", "1", "yes")
    
    # Auto ingestion on first startup
    AUTO_INGEST_ON_STARTUP: bool = os.getenv("AUTO_INGEST_ON_STARTUP", "true").lower() in ("true", "1", "yes")
    AUTO_INGEST_SAMPLE_SIZE: int = int(os.getenv("AUTO_INGEST_SAMPLE_SIZE", "15000"))
    
    # Raw Data & Index Paths
    DATASET_PATH: str = os.getenv(
        "DATASET_PATH", 
        str(BASE_DIR / "repo_metadata.json")
    )
    INDEX_STORE_PATH: str = os.getenv(
        "INDEX_STORE_PATH",
        str(BASE_DIR / "data" / "search_index.pkl")
    )
    
    # ML & Embedding Config
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME", 
        "all-MiniLM-L6-v2"
    )
    EMBEDDING_DIM: int = 384
    BATCH_SIZE: int = 128
    
    # Search & Ranking Weights
    SEMANTIC_WEIGHT: float = 0.55
    LEXICAL_WEIGHT: float = 0.30
    METADATA_WEIGHT: float = 0.15
    
    # Recency decay parameter
    RECENCY_LAMBDA: float = 0.0019

settings = Settings()
