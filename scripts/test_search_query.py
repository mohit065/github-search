from src.search.engine import HybridSearchEngine

engine = HybridSearchEngine()
res = engine.search("web automation testing framework for games and apps", limit=3)
print(f"Total indexed: {len(engine.repo_ids)}")
print(f"Total matched: {res['total']}")
for r in res["results"]:
    repo = r["repo"]
    print(f"\n-> {repo['name_with_owner']} (? {repo['stars']:,} stars, Lang: {repo['primary_language']})")
    print(f"   Match Score: {r['score']:.4f} [Semantic: {r['dense_score']:.4f}, BM25: {r['lexical_score']:.4f}, Activity/Meta: {r['metadata_score']:.4f}]")
    print(f"   Description: {repo['description']}")
