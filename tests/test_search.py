import pytest
import numpy as np
from src.search.embedder import Embedder
from src.search.bm25 import LexicalSearchIndex
from src.search.ranker import compute_metadata_score, reciprocal_rank_fusion

def test_embedder_shapes_and_normalization():
    embedder = Embedder()
    texts = [
        "Lightweight async web server in Python",
        "Deep learning computer vision object detection"
    ]
    vecs = embedder.embed_texts(texts)
    assert vecs.shape == (2, 384)
    # Norm of vectors should be approximately 1.0 (normalized)
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-3)

def test_bm25_lexical_search():
    bm25 = LexicalSearchIndex()
    docs = [
        "PostgreSQL vector database search with pgvector extension",
        "React frontend responsive user interface dashboard",
        "PyTorch convolutional neural networks for image classification"
    ]
    bm25.build_index(repo_ids=[1, 2, 3], texts=docs)
    
    results = bm25.search("postgres vector pgvector")
    assert len(results) > 0
    assert results[0][0] == 1  # Repo ID 1 is top match
    
    react_res = bm25.search("react frontend")
    assert len(react_res) > 0
    assert react_res[0][0] == 2

def test_metadata_score_and_language_boost():
    score_py = compute_metadata_score(stars=10000, activity_score=0.8, primary_language="Python", target_language="Python")
    score_js = compute_metadata_score(stars=10000, activity_score=0.8, primary_language="JavaScript", target_language="Python")
    assert score_py > score_js

def test_reciprocal_rank_fusion():
    dense_ranks = {101: 1, 102: 2, 103: 3}
    lexical_ranks = {102: 1, 101: 2, 104: 3}
    rrf = reciprocal_rank_fusion(dense_ranks, lexical_ranks, k=60)
    
    # 101 and 102 are in both rankings, so their RRF score should exceed single-list candidates
    assert rrf[101] > rrf[103]
    assert rrf[102] > rrf[104]
