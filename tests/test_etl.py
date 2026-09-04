import pytest
from datetime import datetime, timezone
from src.etl.transformers import (
    clean_text,
    compute_activity_score,
    compute_popularity_score,
    synthesize_search_document,
    transform_raw_record
)

def test_clean_text():
    raw_html = "<p>Fast &amp; modern <b>web framework</b></p>\n\t"
    cleaned = clean_text(raw_html)
    assert "<p>" not in cleaned
    assert "<b>" not in cleaned
    assert "Fast" in cleaned and "web framework" in cleaned

def test_compute_activity_score():
    # Active repo pushed recently with many commits and PRs
    active_score = compute_activity_score(
        commits=1500,
        pull_requests=300,
        issues=200,
        pushed_at_str="2026-08-25T12:00:00Z",
        reference_time=datetime(2026, 9, 1, tzinfo=timezone.utc)
    )
    assert 0.0 <= active_score <= 1.0
    assert active_score > 0.65

    # Dormant repo pushed 5 years ago
    dormant_score = compute_activity_score(
        commits=10,
        pull_requests=0,
        issues=0,
        pushed_at_str="2019-01-01T00:00:00Z",
        reference_time=datetime(2026, 9, 1, tzinfo=timezone.utc)
    )
    assert dormant_score < active_score
    assert dormant_score < 0.25

def test_compute_popularity_score():
    pop_high = compute_popularity_score(stars=50000, forks=10000)
    pop_low = compute_popularity_score(stars=5, forks=1)
    assert pop_high > pop_low
    assert 0.0 <= pop_high <= 1.0

def test_synthesize_search_document():
    doc = synthesize_search_document(
        name_with_owner="fastapi/fastapi",
        description="FastAPI framework, high performance, easy to learn, fast to code, ready for production",
        primary_language="Python",
        languages=["Python", "Shell"],
        topics=["api", "asyncio", "python", "pydantic"],
        stars=75000,
        activity_score=0.88
    )
    assert "Repository: fastapi/fastapi" in doc
    assert "Primary Language: Python" in doc
    assert "Topics and tags: api, asyncio, python, pydantic" in doc
    assert "Description: FastAPI framework" in doc

def test_transform_raw_record():
    raw = {
        "owner": "psf",
        "name": "requests",
        "nameWithOwner": "psf/requests",
        "description": "A simple, yet elegant, HTTP library for Python.",
        "primaryLanguage": "Python",
        "languages": [{"name": "Python", "size": 120000}],
        "topics": [{"name": "http", "stars": 100}, {"name": "python", "stars": 500}],
        "stars": 51000,
        "forks": 9200,
        "watchers": 1400,
        "pullRequests": 1200,
        "issues": 800,
        "defaultBranchCommitCount": 6000,
        "createdAt": "2011-02-13T18:00:00Z",
        "pushedAt": "2026-08-10T10:00:00Z",
        "license": "Apache License 2.0"
    }
    processed = transform_raw_record(raw)
    assert processed is not None
    assert processed.name_with_owner == "psf/requests"
    assert processed.primary_language == "Python"
    assert "http" in processed.topics
    assert processed.activity_score > 0.6
