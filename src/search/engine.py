"""
Hybrid Search Engine backed by Qdrant.

Dense vectors:  BAAI/bge-small-en-v1.5 (384d, cosine)
Sparse vectors: Qdrant/bm25 via fastembed  (replaces the old RAM BM25 index)
Fusion:         Qdrant native RRF (Reciprocal Rank Fusion)
Metadata boost: log-scaled stars + activity score, applied post-retrieval

At 4.2M repos the old approach (numpy matrix + BM25Okapi in RAM) would need
~10 GB RAM. Qdrant stores vectors on disk (mmap) and serves sub-5ms queries
from its HNSW index.
"""

from typing import Optional, Dict, Any, List
from src.config import settings
from src.search.embedder import Embedder
from src.search.ranker import compute_metadata_score


class HybridSearchEngine:
    """Singleton Qdrant-backed hybrid search engine."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.embedder = Embedder()
        self.collection = settings.QDRANT_COLLECTION
        self._total_indexed: int = 0

        # Lazy-import heavy deps so the module is importable without them
        from qdrant_client import QdrantClient
        self.client = QdrantClient(url=settings.QDRANT_URL, timeout=60)

        # fastembed sparse model for BM25
        from fastembed import SparseTextEmbedding
        print(f"[Engine] Loading sparse BM25 model: {settings.SPARSE_MODEL_NAME}")
        self.sparse_model = SparseTextEmbedding(model_name=settings.SPARSE_MODEL_NAME)
        print("[Engine] Sparse model ready.")

        self._load_or_build_index()

    # ?? Internal helpers ??????????????????????????????????????????????????????

    def _db_count(self) -> int:
        try:
            from src.db.session import SessionLocal
            from src.db.models import Repository
            db = SessionLocal()
            n = db.query(Repository).count()
            db.close()
            return n
        except Exception:
            return 0

    def _qdrant_count(self) -> int:
        try:
            info = self.client.get_collection(self.collection)
            return info.points_count or 0
        except Exception:
            return 0

    def _ensure_collection(self, recreate: bool = False):
        """Create the Qdrant collection (or recreate if asked)."""
        from qdrant_client.models import (
            VectorParams, SparseVectorParams, Distance,
            HnswConfigDiff, ScalarQuantization, ScalarQuantizationConfig, ScalarType,
        )

        exists = False
        try:
            self.client.get_collection(self.collection)
            exists = True
        except Exception:
            pass

        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False

        if not exists:
            print(f"[Engine] Creating Qdrant collection '{self.collection}'...")
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    "dense": VectorParams(
                        size=self.embedder.dim,
                        distance=Distance.COSINE,
                        on_disk=settings.QDRANT_ON_DISK,
                        hnsw_config=HnswConfigDiff(
                            m=16,
                            ef_construct=100,
                            on_disk=settings.QDRANT_ON_DISK,
                        ),
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams()
                },
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        always_ram=True,
                    )
                ),
            )
            # Payload indexes for fast server-side filtering
            from qdrant_client.models import PayloadSchemaType
            self.client.create_payload_index(self.collection, "primary_language", PayloadSchemaType.KEYWORD)
            self.client.create_payload_index(self.collection, "stars", PayloadSchemaType.INTEGER)
            self.client.create_payload_index(self.collection, "is_archived", PayloadSchemaType.BOOL)
            self.client.create_payload_index(self.collection, "activity_score", PayloadSchemaType.FLOAT)
            print(f"[Engine] Collection '{self.collection}' created with HNSW + INT8 scalar quantization.")

    def _load_or_build_index(self):
        db_count = self._db_count()
        qdrant_count = self._qdrant_count()
        tolerance = max(10, int(db_count * 0.01))

        if qdrant_count > 0 and abs(qdrant_count - db_count) <= tolerance:
            self._total_indexed = qdrant_count
            print(f"[Engine] Qdrant index ready: {qdrant_count:,} repositories.")
        elif db_count > 0:
            print(f"[Engine] Index stale (Qdrant={qdrant_count:,}, DB={db_count:,}). Rebuilding...")
            self.reload_from_db()
        else:
            print("[Engine] No repositories in DB yet. Index will be built after ingestion.")

    # ?? Public API ????????????????????????????????????????????????????????????

    def reload_from_db(self):
        """Read all repos from Postgres, generate dense+sparse embeddings, upsert to Qdrant."""
        from src.db.session import SessionLocal
        from src.db.models import Repository
        from src.db.init_db import init_database
        from qdrant_client.models import PointStruct, SparseVector

        init_database()
        self._ensure_collection()

        db = SessionLocal()
        try:
            total = db.query(Repository).count()
            if total == 0:
                print("[Engine] No repositories found in database.")
                return

            print(f"[Engine] Indexing {total:,} repositories into Qdrant...")
            batch_size = 512
            processed = 0
            offset = 0

            while True:
                repos = db.query(Repository).order_by(Repository.id).offset(offset).limit(batch_size).all()
                if not repos:
                    break

                texts = [r.synthesized_text for r in repos]

                # Dense embeddings (bge-small-en-v1.5, no query prefix for documents)
                dense_vecs = self.embedder.embed_texts(texts)

                # Sparse BM25 embeddings via fastembed
                sparse_embs = list(self.sparse_model.embed(texts))

                points = []
                for i, repo in enumerate(repos):
                    sp = sparse_embs[i]
                    payload = repo.to_dict()
                    # Exclude large text field from payload to save Qdrant storage
                    payload.pop("synthesized_text", None)
                    points.append(PointStruct(
                        id=repo.id,
                        vector={
                            "dense": dense_vecs[i].tolist(),
                            "sparse": SparseVector(
                                indices=sp.indices.tolist(),
                                values=sp.values.tolist(),
                            ),
                        },
                        payload=payload,
                    ))

                self.client.upsert(collection_name=self.collection, points=points, wait=False)
                processed += len(repos)
                offset += batch_size

                if processed % 5000 == 0 or processed == total:
                    pct = processed / total * 100
                    print(f"[Engine] Indexed {processed:,}/{total:,} ({pct:.1f}%)...")

            # Wait for indexing to complete
            self.client.update_collection(
                collection_name=self.collection,
                optimizer_config={"indexing_threshold": 20000},
            )
            self._total_indexed = processed
            print(f"[Engine] Qdrant index built: {processed:,} repositories.")

        except Exception as e:
            print(f"[Engine] Error building index: {e}")
            raise
        finally:
            db.close()

    @property
    def repo_ids(self) -> List[int]:
        """Compatibility shim for routes that check len(engine.repo_ids)."""
        return list(range(self._total_indexed))

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        language: Optional[str] = None,
        min_stars: int = 0,
        sort_by: str = "relevance",
        limit: int = 20,
        offset: int = 0,
    ) -> Dict[str, Any]:
        from qdrant_client.models import (
            Filter, FieldCondition, Range, MatchValue,
            Prefetch, FusionQuery, Fusion, SparseVector,
        )

        if self._total_indexed == 0:
            return {"total": 0, "count": 0, "offset": offset, "limit": limit,
                    "results": [], "query": query, "mode": mode}

        # ?? Build server-side filter ??????????????????????????????????????????
        must = []
        if min_stars > 0:
            must.append(FieldCondition(key="stars", range=Range(gte=min_stars)))
        if language and language.lower() not in ("all", ""):
            must.append(FieldCondition(key="primary_language", match=MatchValue(value=language)))
        qdrant_filter = Filter(must=must) if must else None

        # ?? Embed query ???????????????????????????????????????????????????????
        dense_vec = self.embedder.embed_query(query).tolist()
        sp = list(self.sparse_model.embed([query]))[0]
        sparse_vec = SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist())

        # Retrieve enough candidates for pagination + re-ranking
        n_candidates = min(offset + limit + 200, 2000)

        # ?? Qdrant search ?????????????????????????????????????????????????????
        try:
            if mode == "hybrid":
                raw = self.client.query_points(
                    collection_name=self.collection,
                    prefetch=[
                        Prefetch(query=dense_vec, using="dense",
                                 limit=n_candidates, filter=qdrant_filter),
                        Prefetch(query=sparse_vec, using="sparse",
                                 limit=n_candidates, filter=qdrant_filter),
                    ],
                    query=FusionQuery(fusion=Fusion.RRF),
                    filter=qdrant_filter,
                    limit=n_candidates,
                    with_payload=True,
                ).points
            elif mode == "semantic":
                raw = self.client.query_points(
                    collection_name=self.collection,
                    query=dense_vec,
                    using="dense",
                    filter=qdrant_filter,
                    limit=n_candidates,
                    with_payload=True,
                ).points
            else:  # lexical
                raw = self.client.query_points(
                    collection_name=self.collection,
                    query=sparse_vec,
                    using="sparse",
                    filter=qdrant_filter,
                    limit=n_candidates,
                    with_payload=True,
                ).points
        except Exception as e:
            print(f"[Engine] Qdrant search error: {e}")
            return {"total": 0, "count": 0, "offset": offset, "limit": limit,
                    "results": [], "query": query, "mode": mode}

        if not raw:
            return {"total": 0, "count": 0, "offset": offset, "limit": limit,
                    "results": [], "query": query, "mode": mode}

        # ?? Normalize Qdrant scores to [0, 1] ????????????????????????????????
        qdrant_scores = [p.score for p in raw]
        max_q = max(qdrant_scores) if qdrant_scores else 1.0
        min_q = min(qdrant_scores) if qdrant_scores else 0.0
        rng_q = max_q - min_q if max_q != min_q else 1.0

        # ?? Metadata boost + final scoring ????????????????????????????????????
        scored = []
        for point in raw:
            payload = dict(point.payload)
            payload["id"] = point.id  # ensure ID present
            norm_q = (point.score - min_q) / rng_q  # normalize to [0, 1]

            meta = compute_metadata_score(
                stars=payload.get("stars", 0),
                activity_score=payload.get("activity_score", 0.0),
                primary_language=payload.get("primary_language"),
                target_language=language,
            )
            final = round(settings.QDRANT_WEIGHT * norm_q + settings.METADATA_WEIGHT * meta, 4)

            scored.append({
                "repo": payload,
                "score": final,
                "dense_score": round(norm_q, 4),
                "lexical_score": round(norm_q, 4) if mode == "lexical" else 0.0,
                "metadata_score": round(meta, 4),
            })

        # ?? Sort ??????????????????????????????????????????????????????????????
        if sort_by == "stars":
            scored.sort(key=lambda x: x["repo"].get("stars", 0), reverse=True)
        elif sort_by == "activity":
            scored.sort(key=lambda x: x["repo"].get("activity_score", 0.0), reverse=True)
        elif sort_by == "recency":
            scored.sort(key=lambda x: x["repo"].get("pushed_at") or "", reverse=True)
        else:
            scored.sort(key=lambda x: x["score"], reverse=True)

        total = len(scored)
        page = scored[offset: offset + limit]
        return {
            "total": total,
            "count": len(page),
            "offset": offset,
            "limit": limit,
            "query": query,
            "mode": mode,
            "results": page,
        }
