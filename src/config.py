import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    PROJECT_NAME: str = "GitHub Semantic Search Engine"
    API_V1_PREFIX: str = "/api"

    # PostgreSQL ? used for metadata, stats, and language queries
    POSTGRES_URL: str = os.getenv(
        "POSTGRES_URL",
        "postgresql://postgres:postgres@localhost:5432/github_search"
    )
    USE_POSTGRES: bool = os.getenv("USE_POSTGRES", "false").lower() in ("true", "1", "yes")

    # SQLite fallback (local dev without Docker)
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{BASE_DIR}/data/github_search.db"
    )

    # Qdrant ? used for all vector / hybrid search
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "repositories")
    QDRANT_ON_DISK: bool = os.getenv("QDRANT_ON_DISK", "true").lower() in ("true", "1", "yes")

    # Auto ingestion on first startup
    AUTO_INGEST_ON_STARTUP: bool = os.getenv("AUTO_INGEST_ON_STARTUP", "true").lower() in ("true", "1", "yes")
    AUTO_INGEST_SAMPLE_SIZE: int = int(os.getenv("AUTO_INGEST_SAMPLE_SIZE", "0"))

    # Dataset path
    DATASET_PATH: str = os.getenv(
        "DATASET_PATH",
        str(BASE_DIR / "repo_metadata.json")
    )

    # Embedding model ? BAAI/bge-small-en-v1.5 is ~10% better than all-MiniLM-L6-v2 on MTEB
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    EMBEDDING_DIM: int = 384
    BATCH_SIZE: int = 256  # larger batch fits well for bge-small

    # Sparse BM25 model (via fastembed)
    SPARSE_MODEL_NAME: str = "Qdrant/bm25"

    # Metadata ranking weights (applied as post-Qdrant boost)
    METADATA_WEIGHT: float = 0.15
    QDRANT_WEIGHT: float = 0.85

    # Recency decay parameter for activity score
    RECENCY_LAMBDA: float = 0.0019

settings = Settings()
