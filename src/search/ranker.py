from typing import List, Dict, Any, Optional
import numpy as np

def compute_metadata_score(
    stars: int,
    activity_score: float,
    primary_language: str,
    target_language: Optional[str] = None
) -> float:
    """
    Compute metadata quality and relevance boost score between 0.0 and 1.0.
    """
    # Star popularity component (log scale)
    star_comp = min(1.0, np.log10(stars + 1.0) / 5.0)
    
    # Activity component
    act_comp = max(0.0, min(1.0, activity_score))
    
    # Language match bonus
    lang_bonus = 0.0
    if target_language and primary_language:
        if target_language.lower() == primary_language.lower():
            lang_bonus = 0.25
            
    base_score = (0.50 * star_comp) + (0.50 * act_comp) + lang_bonus
    return float(min(1.0, base_score))

def reciprocal_rank_fusion(
    dense_ranks: Dict[int, int],
    lexical_ranks: Dict[int, int],
    k: int = 60
) -> Dict[int, float]:
    """
    Combine rankings using Reciprocal Rank Fusion (RRF).
    RRF(d) = sum(1 / (k + rank(d)))
    """
    all_ids = set(dense_ranks.keys()) | set(lexical_ranks.keys())
    rrf_scores = {}
    
    for rid in all_ids:
        score = 0.0
        if rid in dense_ranks:
            score += 1.0 / (k + dense_ranks[rid])
        if rid in lexical_ranks:
            score += 1.0 / (k + lexical_ranks[rid])
        rrf_scores[rid] = score
        
    return rrf_scores
