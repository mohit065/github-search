import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config import settings
from src.db.init_db import init_database
from src.db.session import SessionLocal
from src.db.models import Repository
from src.api.routes import router as api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[FastAPI] Initializing database schema...")
    init_database()

    # Auto-ingest if DB is empty and dataset file exists
    db = SessionLocal()
    try:
        repo_count = db.query(Repository).count()
        print(f"[FastAPI] Repository count in DB: {repo_count}")
        if repo_count == 0 and settings.AUTO_INGEST_ON_STARTUP and os.path.exists(settings.DATASET_PATH):
            print(f"[FastAPI] Auto-ingesting {settings.AUTO_INGEST_SAMPLE_SIZE:,} repos from dataset...")
            from src.etl.pipeline import stream_and_ingest_dataset
            stream_and_ingest_dataset(
                dataset_path=settings.DATASET_PATH,
                sample_size=settings.AUTO_INGEST_SAMPLE_SIZE
            )
    except Exception as e:
        print(f"[FastAPI] Error during startup ingestion: {e}")
    finally:
        db.close()

    # Warm up the search engine AFTER ingestion
    print("[FastAPI] Warming up search engine...")
    from src.search.engine import HybridSearchEngine
    engine = HybridSearchEngine()
    if not engine.repo_ids:
        engine.reload_from_db()
    print(f"[FastAPI] Search engine ready: {len(engine.repo_ids):,} repositories indexed.")

    yield
    print("[FastAPI] Shutting down.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Semantic Search Engine for GitHub Repositories using Hybrid Dense Vector, BM25, and Activity Ranking.",
    version="1.0.0",
    lifespan=lifespan
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
    return {"name": settings.PROJECT_NAME, "docs": "/docs",
            "api": settings.API_V1_PREFIX, "status": "online"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)
