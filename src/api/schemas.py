from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class RepositoryItem(BaseModel):
    id: int
    name_with_owner: str
    owner: str
    name: str
    description: Optional[str] = ""
    primary_language: Optional[str] = "Unknown"
    languages: List[str] = []
    topics: List[str] = []
    stars: int = 0
    forks: int = 0
    watchers: int = 0
    pull_requests: int = 0
    issues: int = 0
    commits: int = 0
    disk_usage_kb: int = 0
    is_fork: bool = False
    is_archived: bool = False
    created_at: Optional[str] = None
    pushed_at: Optional[str] = None
    license: Optional[str] = None
    activity_score: float = 0.0
    popularity_score: float = 0.0

class SearchResultItem(BaseModel):
    repo: RepositoryItem
    score: float
    dense_score: float
    lexical_score: float
    metadata_score: float

class SearchResponse(BaseModel):
    total: int
    count: int
    offset: int
    limit: int
    query: str
    mode: str
    results: List[SearchResultItem]

class LanguageStat(BaseModel):
    language: str
    count: int

class DatasetStatsResponse(BaseModel):
    total_repositories: int
    indexed_in_search: int
    top_languages: List[LanguageStat]
    avg_stars: float
    avg_activity_score: float

class HealthResponse(BaseModel):
    status: str
    project: str
    total_indexed: int
