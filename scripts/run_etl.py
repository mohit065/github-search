import sys
import argparse
from src.config import settings
from src.etl.pipeline import stream_and_ingest_dataset
from src.search.engine import HybridSearchEngine

def main():
    parser = argparse.ArgumentParser(description="Run GitHub Repository ETL and Search Indexing")
    parser.add_argument("--dataset", type=str, default=settings.DATASET_PATH, help="Path to repo_metadata.json")
    parser.add_argument("--sample-size", type=int, default=15000, help="Number of records to sample and ingest (0 or negative for all)")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for database ingestion")
    parser.add_argument("--skip-index", action="store_true", help="Skip re-building the vector/BM25 search index")
    args = parser.parse_args()

    sample_size = None if args.sample_size <= 0 else args.sample_size
    print(f"=== Starting GitHub Repository ETL Pipeline ===")
    stream_and_ingest_dataset(
        dataset_path=args.dataset,
        sample_size=sample_size,
        batch_size=args.batch_size
    )

    if not args.skip_index:
        print("\n=== Building Search Embeddings & Index ===")
        engine = HybridSearchEngine()
        engine.reload_from_db()
        print("=== Search Engine Ready! ===")

if __name__ == "__main__":
    main()
