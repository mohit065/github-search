from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class LanguageItem(BaseModel):
    name: str
    size: int = 0

class TopicItem(BaseModel):
    name: str
    stars: int = 0

class RawRepoRecord(BaseModel):
    owner: Optional[str] = None
    name: Optional[str] = None
    nameWithOwner: Optional[str] = None
    description: Optional[str] = None
    primaryLanguage: Optional[str] = None
    languages: Optional[List[Dict[str, Any]]] = []
    languageCount: Optional[int] = 0
    topics: Optional[List[Dict[str, Any]]] = []
    topicCount: Optional[int] = 0
    stars: Optional[int] = 0
    forks: Optional[int] = 0
    watchers: Optional[int] = 0
    pullRequests: Optional[int] = 0
    issues: Optional[int] = 0
    diskUsageKb: Optional[int] = 0
    isFork: Optional[bool] = False
    isArchived: Optional[bool] = False
    createdAt: Optional[str] = None
    pushedAt: Optional[str] = None
    defaultBranchCommitCount: Optional[int] = 0
    license: Optional[str] = None
    assignableUserCount: Optional[int] = 0
    codeOfConduct: Optional[str] = None
    forkingAllowed: Optional[bool] = True
    parent: Optional[Any] = None

class ProcessedRepository(BaseModel):
    id: Optional[int] = None
    name_with_owner: str
    owner: str
    name: str
    description: str = ""
    primary_language: str = "Unknown"
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
    synthesized_text: str = ""
