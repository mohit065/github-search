import os
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config import settings
from src.db.init_db import init_database
from src.api.routes import router as api_router


def _background_startup():
    """
    Run ETL + Qdrant index build in a daemon thread so the API is reachable
    immediately. Search returns empty results until the index is ready.
    """
    from src.db.session import SessionLocal
    from src.db.models import Repository

    # 1. Ingest into Postgres if empty
    db = SessionLocal()
    try:
        repo_count = db.query(Repository).count()
        print(f"[FastAPI] Postgres repo count: {repo_count:,}")
        if repo_count == 0 and settings.AUTO_INGEST_ON_STARTUP and os.path.exists(settings.DATASET_PATH):
            n = settings.AUTO_INGEST_SAMPLE_SIZE or None
            label = f"{n:,}" if n else "ALL"
            print(f"[FastAPI] Ingesting {label} repos from dataset (background)...")
            from src.etl.pipeline import stream_and_ingest_dataset
            stream_and_ingest_dataset(dataset_path=settings.DATASET_PATH, sample_size=n)
    except Exception as e:
        print(f"[FastAPI] Ingestion error: {e}")
    finally:
        db.close()

    # 2. Build / validate Qdrant index
    try:
        print("[FastAPI] Building Qdrant search index (background)...")
        from src.search.engine import HybridSearchEngine
        engine = HybridSearchEngine()
        if engine._total_indexed == 0:
            engine.reload_from_db()
        print(f"[FastAPI] Search engine ready: {engine._total_indexed:,} repositories indexed.")
    except Exception as e:
        print(f"[FastAPI] Search engine error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Init DB schema synchronously (fast, < 1s)
    print("[FastAPI] Initializing Postgres schema...")
    init_database()

    # Kick off ETL + Qdrant indexing in the background ? API starts immediately
    t = threading.Thread(target=_background_startup, name="startup-indexer", daemon=True)
    t.start()
    print("[FastAPI] Startup tasks running in background. API is ready.")

    yield
    print("[FastAPI] Shutting down.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Semantic Search Engine for GitHub Repositories ? Qdrant hybrid dense+sparse search with BAAI/bge-small-en-v1.5 embeddings.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.get("/")
def root():
    from src.search.engine import HybridSearchEngine
    engine = HybridSearchEngine()
    indexed = engine._total_indexed
    return {
        "name": settings.PROJECT_NAME,
        "version": "2.0.0",
        "backend": "Qdrant hybrid search (dense + sparse BM25)",
        "model": settings.EMBEDDING_MODEL_NAME,
        "indexed_repositories": indexed,
        "status": "ready" if indexed > 0 else "warming_up",
        "docs": "/docs",
        "api": settings.API_V1_PREFIX,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)
