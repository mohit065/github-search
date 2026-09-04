import re
import math
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from src.etl.schema import ProcessedRepository
from src.config import settings

HTML_TAG_RE = re.compile(r"<[^>]+>")
SPECIAL_CHARS_RE = re.compile(r"[\r\n\t]+")
MULTI_SPACE_RE = re.compile(r"\s+")

def clean_text(text: Optional[str]) -> str:
    """Sanitize and normalize string content."""
    if not text:
        return ""
    # Strip HTML tags
    cleaned = HTML_TAG_RE.sub(" ", text)
    # Normalize whitespace & newlines
    cleaned = SPECIAL_CHARS_RE.sub(" ", cleaned)
    cleaned = MULTI_SPACE_RE.sub(" ", cleaned)
    return cleaned.strip()

def parse_iso_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    """Parse ISO datetime string to timezone-aware UTC datetime."""
    if not dt_str:
        return None
    try:
        if dt_str.endswith("Z"):
            dt_str = dt_str[:-1] + "+00:00"
        return datetime.fromisoformat(dt_str)
    except Exception:
        return None

def compute_activity_score(
    commits: int,
    pull_requests: int,
    issues: int,
    pushed_at_str: Optional[str],
    reference_time: Optional[datetime] = None
) -> float:
    """
    Compute composite repository activity score normalized between 0.0 and 1.0.
    Factors:
    - Commit volume (log scaled)
    - Interaction volume: PRs + Issues (log scaled)
    - Recency decay factor e^(-lambda * delta_days)
    """
    if reference_time is None:
        reference_time = datetime(2026, 9, 1, tzinfo=timezone.utc)
    
    # Recency factor
    pushed_at = parse_iso_datetime(pushed_at_str)
    if pushed_at:
        days_since_push = max(0.0, (reference_time - pushed_at).total_seconds() / 86400.0)
    else:
        days_since_push = 1500.0  # Default to older inactivity if missing
    
    recency_factor = math.exp(-settings.RECENCY_LAMBDA * days_since_push)
    
    # Log scaled activity metrics
    norm_commits = min(1.0, math.log10(commits + 1) / 4.0)
    norm_interactions = min(1.0, math.log10(pull_requests + issues + 1) / 4.0)
    
    score = (0.35 * norm_commits) + (0.25 * norm_interactions) + (0.40 * recency_factor)
    return round(float(score), 4)

def compute_popularity_score(stars: int, forks: int) -> float:
    """Compute normalized popularity score from stars and forks."""
    norm_stars = min(1.0, math.log10(stars + 1) / 5.5)
    norm_forks = min(1.0, math.log10(forks + 1) / 4.5)
    score = (0.75 * norm_stars) + (0.25 * norm_forks)
    return round(float(score), 4)

def synthesize_search_document(
    name_with_owner: str,
    description: str,
    primary_language: str,
    languages: List[str],
    topics: List[str],
    stars: int,
    activity_score: float
) -> str:
    """
    Synthesize an enriched text representation of the repository optimized
    for dense vector embedding and natural language matching.
    """
    parts = []
    
    # Name and Owner
    parts.append(f"Repository: {name_with_owner}")
    
    # Primary Language
    if primary_language and primary_language != "Unknown":
        parts.append(f"Primary Language: {primary_language}")
        
    # Other languages
    other_langs = [l for l in languages if l != primary_language]
    if other_langs:
        parts.append(f"Languages: {', '.join(other_langs[:5])}")
        
    # Topics / Tags
    if topics:
        parts.append(f"Topics and tags: {', '.join(topics[:12])}")
        
    # Cleaned Description
    if description:
        parts.append(f"Description: {description}")
    
    # Activity Level Hint
    if activity_score > 0.7:
        parts.append("Status: Highly active and maintained")
    elif activity_score < 0.3:
        parts.append("Status: Inactive or legacy")
        
    return " | ".join(parts)

def transform_raw_record(raw: Dict[str, Any]) -> Optional[ProcessedRepository]:
    """Transform raw dictionary record into clean ProcessedRepository."""
    owner = clean_text(raw.get("owner"))
    name = clean_text(raw.get("name"))
    name_with_owner = raw.get("nameWithOwner") or f"{owner}/{name}"
    
    if not name_with_owner or name_with_owner == "/":
        return None
        
    description = clean_text(raw.get("description"))
    primary_language = clean_text(raw.get("primaryLanguage")) or "Unknown"
    
    # Extract language list
    raw_languages = raw.get("languages") or []
    languages = []
    for item in raw_languages:
        if isinstance(item, dict) and "name" in item and item["name"]:
            languages.append(clean_text(item["name"]))
        elif isinstance(item, str) and item:
            languages.append(clean_text(item))
            
    # Extract topics list
    raw_topics = raw.get("topics") or []
    topics = []
    for item in raw_topics:
        if isinstance(item, dict) and "name" in item and item["name"]:
            topics.append(clean_text(item["name"]).lower())
        elif isinstance(item, str) and item:
            topics.append(clean_text(item).lower())
            
    stars = int(raw.get("stars") or 0)
    forks = int(raw.get("forks") or 0)
    watchers = int(raw.get("watchers") or 0)
    pull_requests = int(raw.get("pullRequests") or 0)
    issues = int(raw.get("issues") or 0)
    commits = int(raw.get("defaultBranchCommitCount") or 0)
    disk_usage_kb = int(raw.get("diskUsageKb") or 0)
    is_fork = bool(raw.get("isFork") or False)
    is_archived = bool(raw.get("isArchived") or False)
    created_at = raw.get("createdAt")
    pushed_at = raw.get("pushedAt")
    license_str = clean_text(raw.get("license")) or None
    
    activity_score = compute_activity_score(
        commits=commits,
        pull_requests=pull_requests,
        issues=issues,
        pushed_at_str=pushed_at
    )
    
    popularity_score = compute_popularity_score(stars=stars, forks=forks)
    
    synthesized_text = synthesize_search_document(
        name_with_owner=name_with_owner,
        description=description,
        primary_language=primary_language,
        languages=languages,
        topics=topics,
        stars=stars,
        activity_score=activity_score
    )
    
    return ProcessedRepository(
        name_with_owner=name_with_owner,
        owner=owner,
        name=name,
        description=description,
        primary_language=primary_language,
        languages=languages,
        topics=topics,
        stars=stars,
        forks=forks,
        watchers=watchers,
        pull_requests=pull_requests,
        issues=issues,
        commits=commits,
        disk_usage_kb=disk_usage_kb,
        is_fork=is_fork,
        is_archived=is_archived,
        created_at=created_at,
        pushed_at=pushed_at,
        license=license_str,
        activity_score=activity_score,
        popularity_score=popularity_score,
        synthesized_text=synthesized_text
    )
