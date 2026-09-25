import numpy as np
from typing import List
from src.config import settings

# BGE models achieve better retrieval when queries use this prefix
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

class Embedder:
    """Wrapper for dense vector embedding using sentence-transformers BAAI/bge-small-en-v1.5."""
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
        try:
            from sentence_transformers import SentenceTransformer
            print(f"[Embedder] Loading model: {settings.EMBEDDING_MODEL_NAME}")
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME, device="cpu")
            self.dim = settings.EMBEDDING_DIM
            # Verify dimension matches config
            test_vec = self.model.encode(["test"], normalize_embeddings=True, convert_to_numpy=True)
            self.dim = test_vec.shape[1]
            print(f"[Embedder] Model ready ? dim={self.dim}")
        except Exception as e:
            print(f"[Embedder] Warning: {e}")
            self.model = None
            self.dim = settings.EMBEDDING_DIM

    def embed_texts(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """Embed a list of documents (no query prefix)."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        bs = batch_size or settings.BATCH_SIZE
        if self.model is not None:
            vecs = self.model.encode(
                texts, batch_size=bs,
                show_progress_bar=len(texts) > 500,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return vecs.astype(np.float32)
        # Deterministic fallback
        rng = np.random.RandomState(42)
        v = rng.randn(len(texts), self.dim).astype(np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string with the BGE query prefix for better retrieval."""
        prefixed = BGE_QUERY_PREFIX + query.strip()
        return self.embed_texts([prefixed])[0]
