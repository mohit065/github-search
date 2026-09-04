from sqlalchemy import Column, Integer, String, Float, Boolean, Text, JSON, DateTime, Index
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Repository(Base):
    __tablename__ = "repositories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name_with_owner = Column(String(255), unique=True, nullable=False, index=True)
    owner = Column(String(128), nullable=False, index=True)
    name = Column(String(128), nullable=False)
    description = Column(Text, nullable=True)
    primary_language = Column(String(64), nullable=True, index=True)
    languages = Column(JSON, nullable=True)
    topics = Column(JSON, nullable=True)
    
    stars = Column(Integer, default=0, index=True)
    forks = Column(Integer, default=0)
    watchers = Column(Integer, default=0)
    pull_requests = Column(Integer, default=0)
    issues = Column(Integer, default=0)
    commits = Column(Integer, default=0)
    disk_usage_kb = Column(Integer, default=0)
    
    is_fork = Column(Boolean, default=False)
    is_archived = Column(Boolean, default=False)
    created_at = Column(String(64), nullable=True)
    pushed_at = Column(String(64), nullable=True)
    license = Column(String(128), nullable=True)
    
    activity_score = Column(Float, default=0.0, index=True)
    popularity_score = Column(Float, default=0.0, index=True)
    synthesized_text = Column(Text, nullable=False)

    __table_args__ = (
        Index("idx_repo_lang_stars", "primary_language", "stars"),
        Index("idx_repo_activity_stars", "activity_score", "stars"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "name_with_owner": self.name_with_owner,
            "owner": self.owner,
            "name": self.name,
            "description": self.description,
            "primary_language": self.primary_language,
            "languages": self.languages or [],
            "topics": self.topics or [],
            "stars": self.stars,
            "forks": self.forks,
            "watchers": self.watchers,
            "pull_requests": self.pull_requests,
            "issues": self.issues,
            "commits": self.commits,
            "disk_usage_kb": self.disk_usage_kb,
            "is_fork": self.is_fork,
            "is_archived": self.is_archived,
            "created_at": self.created_at,
            "pushed_at": self.pushed_at,
            "license": self.license,
            "activity_score": self.activity_score,
            "popularity_score": self.popularity_score,
            "synthesized_text": self.synthesized_text
        }
