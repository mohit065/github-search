import os
import pickle
from typing import List, Dict, Any, Optional
import numpy as np
from src.config import settings
from src.search.embedder import Embedder
from src.search.bm25 import LexicalSearchIndex
from src.search.ranker import compute_metadata_score
from src.db.session import SessionLocal
from src.db.models import Repository
from src.db.init_db import init_database

class HybridSearchEngine:
    """Singleton hybrid search engine combining dense vector and BM25 lexical retrieval."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HybridSearchEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self.embedder = Embedder()
        self.bm25_index = LexicalSearchIndex()
        self.repo_ids: List[int] = []
        self.embeddings: np.ndarray = np.empty((0, settings.EMBEDDING_DIM), dtype=np.float32)
        self.repo_lookup: Dict[int, Dict[str, Any]] = {}
        self._initialized = True
        # Defer index loading until explicitly called
        self._load_or_build_index()

    def _db_count(self) -> int:
        """Return number of repos currently in the database."""
        try:
            db = SessionLocal()
            count = db.query(Repository).count()
            db.close()
            return count
        except Exception:
            return 0

    def _load_or_build_index(self):
        """Load cached index if it matches DB count, otherwise rebuild from DB."""
        db_count = self._db_count()

        if os.path.exists(settings.INDEX_STORE_PATH) and db_count > 0:
            try:
                with open(settings.INDEX_STORE_PATH, "rb") as f:
                    data = pickle.load(f)
                cached_ids = data.get("repo_ids", [])
                # Only use cache if it matches the DB (within 1% tolerance)
                if abs(len(cached_ids) - db_count) <= max(10, db_count * 0.01):
                    self.repo_ids = cached_ids
                    self.embeddings = data["embeddings"]
                    self.repo_lookup = data["repo_lookup"]
                    self.bm25_index.build_index(
                        self.repo_ids,
                        [self.repo_lookup[rid]["synthesized_text"] for rid in self.repo_ids]
                    )
                    print(f"[Search Engine] Loaded cached index: {len(self.repo_ids):,} repositories.")
                    return
                else:
                    print(f"[Search Engine] Cache mismatch (cache={len(cached_ids)}, DB={db_count}). Rebuilding.")
            except Exception as e:
                print(f"[Search Engine] Cache load error: {e}. Rebuilding.")

        if db_count > 0:
            self.reload_from_db()
        else:
            print("[Search Engine] No repositories in database yet. Index will be built after ingestion.")

    def reload_from_db(self):
        """Load all repositories from database, generate embeddings, and build indexes."""
        init_database()
        db = SessionLocal()
        try:
            repos = db.query(Repository).all()
            if not repos:
                print("[Search Engine] No repositories found in database.")
                return

            print(f"[Search Engine] Building index for {len(repos):,} repositories...")
            self.repo_ids = [r.id for r in repos]
            self.repo_lookup = {r.id: r.to_dict() for r in repos}
            texts = [r.synthesized_text for r in repos]

            print("[Search Engine] Computing dense vector embeddings...")
            self.embeddings = self.embedder.embed_texts(texts, batch_size=settings.BATCH_SIZE)

            print("[Search Engine] Building BM25 index...")
            self.bm25_index.build_index(self.repo_ids, texts)

            # Persist index
            os.makedirs(os.path.dirname(settings.INDEX_STORE_PATH), exist_ok=True)
            with open(settings.INDEX_STORE_PATH, "wb") as f:
                pickle.dump({
                    "repo_ids": self.repo_ids,
                    "embeddings": self.embeddings,
                    "repo_lookup": self.repo_lookup,
                }, f)
            print(f"[Search Engine] Index built and cached: {len(self.repo_ids):,} repositories.")
        except Exception as e:
            print(f"[Search Engine] Error building index: {e}")
            raise
        finally:
            db.close()

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        language: Optional[str] = None,
        min_stars: int = 0,
        sort_by: str = "relevance",
        limit: int = 20,
        offset: int = 0
    ) -> Dict[str, Any]:
        if not self.repo_ids:
            return {"total": 0, "count": 0, "offset": offset, "limit": limit,
                    "results": [], "query": query, "mode": mode}

        # 1. Dense vector similarities
        dense_scores: Dict[int, float] = {}
        if mode in ("hybrid", "semantic"):
            query_vec = self.embedder.embed_query(query)
            sims = np.dot(self.embeddings, query_vec)
            sims = np.clip((sims + 1.0) / 2.0, 0.0, 1.0)
            for idx, rid in enumerate(self.repo_ids):
                dense_scores[rid] = float(sims[idx])

        # 2. BM25 lexical scores
        lexical_scores: Dict[int, float] = {}
        if mode in ("hybrid", "lexical"):
            for rid, score in self.bm25_index.search(query, top_k=min(len(self.repo_ids), 500)):
                lexical_scores[rid] = score

        # 3. Build candidates with filtering and scoring
        candidates = []
        for rid in self.repo_ids:
            repo = self.repo_lookup.get(rid)
            if not repo:
                continue

            if min_stars > 0 and repo["stars"] < min_stars:
                continue
            if language and language.lower() not in ("all", ""):
                repo_lang = (repo["primary_language"] or "").lower()
                other_langs = [l.lower() for l in (repo.get("languages") or [])]
                if language.lower() != repo_lang and language.lower() not in other_langs:
                    continue

            d = dense_scores.get(rid, 0.0)
            l = lexical_scores.get(rid, 0.0)
            m = compute_metadata_score(
                stars=repo["stars"],
                activity_score=repo["activity_score"],
                primary_language=repo["primary_language"],
                target_language=language
            )

            if mode == "semantic":
                score = 0.85 * d + 0.15 * m
            elif mode == "lexical":
                score = 0.85 * l + 0.15 * m
            else:
                score = (settings.SEMANTIC_WEIGHT * d +
                         settings.LEXICAL_WEIGHT * l +
                         settings.METADATA_WEIGHT * m)

            candidates.append({
                "repo": repo,
                "score": round(score, 4),
                "dense_score": round(d, 4),
                "lexical_score": round(l, 4),
                "metadata_score": round(m, 4),
            })

        # 4. Sort
        if sort_by == "stars":
            candidates.sort(key=lambda x: (x["repo"]["stars"], x["score"]), reverse=True)
        elif sort_by == "activity":
            candidates.sort(key=lambda x: (x["repo"]["activity_score"], x["score"]), reverse=True)
        elif sort_by == "recency":
            candidates.sort(key=lambda x: (x["repo"]["pushed_at"] or "", x["score"]), reverse=True)
        else:
            candidates.sort(key=lambda x: x["score"], reverse=True)

        total = len(candidates)
        page = candidates[offset: offset + limit]
        return {
            "total": total,
            "count": len(page),
            "offset": offset,
            "limit": limit,
            "query": query,
            "mode": mode,
            "results": page,
        }
