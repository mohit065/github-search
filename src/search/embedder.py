import numpy as np
from typing import List, Union
from src.config import settings

class Embedder:
    """Wrapper for dense vector embedding generation using Sentence Transformers."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Embedder, cls).__new__(cls)
            cls._instance._init_model()
        return cls._instance
        
    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            print(f"[Embedder] Loading sentence transformer model: {settings.EMBEDDING_MODEL_NAME}")
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME, device="cpu")
            self.dim = settings.EMBEDDING_DIM
        except Exception as e:
            print(f"[Embedder] Warning loading SentenceTransformer: {e}")
            self.model = None
            self.dim = settings.EMBEDDING_DIM

    def embed_texts(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        """Generate dense embeddings for a list of text strings."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
            
        if self.model is not None:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            return embeddings.astype(np.float32)
        else:
            # Fallback random deterministic unit vectors if model is absent
            rng = np.random.RandomState(42)
            vecs = rng.randn(len(texts), self.dim).astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / np.maximum(norms, 1e-9)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate normalized embedding vector for a single query."""
        res = self.embed_texts([query])
        return res[0]
