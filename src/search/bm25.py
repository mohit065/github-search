import re
from typing import List, Dict, Any, Tuple
import numpy as np
from rank_bm25 import BM25Okapi

WORD_RE = re.compile(r"\b[a-zA-Z0-9_\-\.]+\b")

def tokenize(text: str) -> List[str]:
    """Tokenize and normalize text for BM25 indexing."""
    if not text:
        return []
    return [t.lower() for t in WORD_RE.findall(text)]

class LexicalSearchIndex:
    """In-memory BM25 lexical index over repository corpus."""
    def __init__(self):
        self.corpus_ids: List[int] = []
        self.bm25: BM25Okapi = None

    def build_index(self, repo_ids: List[int], texts: List[str]):
        """Construct BM25 index from repository documents."""
        self.corpus_ids = repo_ids
        tokenized_corpus = [tokenize(doc) for doc in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        """Score query against corpus using BM25 and return (repo_id, score) pairs."""
        if not self.bm25 or not self.corpus_ids:
            return []
            
        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []
            
        scores = self.bm25.get_scores(tokenized_query)
        # Normalize BM25 scores to [0.0, 1.0]
        max_score = float(np.max(scores)) if len(scores) > 0 else 0.0
        if max_score > 0:
            norm_scores = scores / max_score
        else:
            norm_scores = scores

        top_indices = np.argsort(norm_scores)[::-1][:top_k]
        return [(self.corpus_ids[i], float(norm_scores[i])) for i in top_indices if norm_scores[i] > 0.0]
