from typing import Optional, List
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from src.db.session import get_db
from src.db.models import Repository
from src.api.schemas import (
    SearchResponse,
    RepositoryItem,
    DatasetStatsResponse,
    HealthResponse,
    LanguageStat
)

router = APIRouter()

# Lazy getter ? avoids importing the singleton at module load time
def get_engine():
    from src.search.engine import HybridSearchEngine
    return HybridSearchEngine()

@router.get("/health", response_model=HealthResponse)
def health_check():
    engine = get_engine()
    return HealthResponse(
        status="healthy",
        project="GitHub Semantic Search Engine",
        total_indexed=len(engine.repo_ids)
    )

@router.get("/search", response_model=SearchResponse)
def search_repositories(
    q: str = Query(..., description="Natural language search query"),
    mode: str = Query("hybrid", pattern="^(hybrid|semantic|lexical)$"),
    language: Optional[str] = Query(None),
    min_stars: int = Query(0, ge=0),
    sort_by: str = Query("relevance", pattern="^(relevance|stars|activity|recency)$"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    engine = get_engine()
    return engine.search(
        query=q, mode=mode, language=language,
        min_stars=min_stars, sort_by=sort_by,
        limit=limit, offset=offset
    )

@router.get("/repos/{owner}/{name}", response_model=RepositoryItem)
def get_repository(owner: str, name: str, db: Session = Depends(get_db)):
    repo = db.query(Repository).filter(
        Repository.name_with_owner == f"{owner}/{name}"
    ).first()
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")
    return repo.to_dict()

@router.get("/languages", response_model=List[str])
def get_available_languages(db: Session = Depends(get_db)):
    results = (
        db.query(Repository.primary_language)
        .filter(Repository.primary_language != "Unknown",
                Repository.primary_language.isnot(None))
        .distinct()
        .order_by(Repository.primary_language.asc())
        .all()
    )
    return [r[0] for r in results if r[0]]

@router.get("/stats", response_model=DatasetStatsResponse)
def get_dataset_stats(db: Session = Depends(get_db)):
    engine = get_engine()
    total = db.query(func.count(Repository.id)).scalar() or 0
    avg_stars = db.query(func.avg(Repository.stars)).scalar() or 0.0
    avg_activity = db.query(func.avg(Repository.activity_score)).scalar() or 0.0

    top_langs = (
        db.query(Repository.primary_language, func.count(Repository.id).label("c"))
        .filter(Repository.primary_language != "Unknown",
                Repository.primary_language.isnot(None))
        .group_by(Repository.primary_language)
        .order_by(func.count(Repository.id).desc())
        .limit(10)
        .all()
    )

    return DatasetStatsResponse(
        total_repositories=total,
        indexed_in_search=len(engine.repo_ids),
        top_languages=[LanguageStat(language=l, count=c) for l, c in top_langs],
        avg_stars=round(float(avg_stars), 2),
        avg_activity_score=round(float(avg_activity), 4)
    )
