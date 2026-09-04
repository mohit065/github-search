import pytest
import os
import tempfile
from fastapi.testclient import TestClient

# Use a separate temp DB so tests never pollute the real index
@pytest.fixture(scope="module", autouse=True)
def isolated_db(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("testdb")
    db_path = str(tmp / "test_github.db")
    idx_path = str(tmp / "test_index.pkl")
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    os.environ["INDEX_STORE_PATH"] = idx_path
    os.environ["USE_POSTGRES"] = "false"
    # Reset settings singleton
    import src.config as cfg
    cfg.settings.DATABASE_URL = f"sqlite:///{db_path}"
    cfg.settings.INDEX_STORE_PATH = idx_path
    cfg.settings.USE_POSTGRES = False
    # Reset DB engine & session
    import src.db.session as sess
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker, scoped_session
    sess.engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    sess.SessionLocal = scoped_session(sessionmaker(bind=sess.engine))
    # Reset search engine singleton
    import src.search.engine as eng
    eng.HybridSearchEngine._instance = None
    yield

@pytest.fixture(scope="module")
def setup_test_data():
    from src.db.init_db import init_database
    from src.db.session import SessionLocal
    from src.db.models import Repository
    from src.search.engine import HybridSearchEngine

    init_database()
    db = SessionLocal()
    db.query(Repository).delete()

    r1 = Repository(
        name_with_owner="tiangolo/fastapi",
        owner="tiangolo", name="fastapi",
        description="FastAPI framework, high performance, easy to learn, fast to code, ready for production",
        primary_language="Python",
        languages=["Python"], topics=["api", "asyncio", "python"],
        stars=70000, forks=6000, commits=3500,
        activity_score=0.92, popularity_score=0.95,
        synthesized_text="Repository: tiangolo/fastapi | Primary Language: Python | Description: FastAPI framework high performance easy to learn | Topics: api, asyncio, python"
    )
    r2 = Repository(
        name_with_owner="facebook/react",
        owner="facebook", name="react",
        description="The library for web and native user interfaces",
        primary_language="JavaScript",
        languages=["JavaScript"], topics=["react", "ui", "frontend"],
        stars=220000, forks=45000, commits=16000,
        activity_score=0.98, popularity_score=0.99,
        synthesized_text="Repository: facebook/react | Primary Language: JavaScript | Description: The library for web and native user interfaces | Topics: react, ui, frontend"
    )
    db.add_all([r1, r2])
    db.commit()
    db.close()

    engine = HybridSearchEngine()
    engine.reload_from_db()
    yield

def test_health_endpoint(setup_test_data):
    from src.api.main import app
    client = TestClient(app)
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"
    assert res.json()["total_indexed"] >= 2

def test_languages_endpoint(setup_test_data):
    from src.api.main import app
    client = TestClient(app)
    res = client.get("/api/languages")
    assert res.status_code == 200
    langs = res.json()
    assert "Python" in langs
    assert "JavaScript" in langs

def test_search_endpoint(setup_test_data):
    from src.api.main import app
    client = TestClient(app)
    res = client.get("/api/search", params={"q": "async web framework python", "mode": "hybrid"})
    assert res.status_code == 200
    data = res.json()
    assert data["total"] > 0
    names = [r["repo"]["name_with_owner"] for r in data["results"]]
    assert "tiangolo/fastapi" in names
