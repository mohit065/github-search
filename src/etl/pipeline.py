import os
import time
from typing import Optional, List
import ijson
from sqlalchemy.orm import Session
from src.config import settings
from src.etl.transformers import transform_raw_record
from src.db.session import SessionLocal
from src.db.models import Repository
from src.db.init_db import init_database

def stream_and_ingest_dataset(
    dataset_path: str,
    sample_size: Optional[int] = 15000,
    batch_size: int = 500
):
    """
    Stream massive JSON array dataset using ijson, transform records,
    and bulk-load into the database in optimized batches.
    """
    init_database()
    db: Session = SessionLocal()
    
    start_time = time.time()
    print(f"[ETL Pipeline] Ingesting from: {dataset_path}")
    if sample_size:
        print(f"[ETL Pipeline] Target sample size: {sample_size:,} records")
    else:
        print("[ETL Pipeline] Ingesting full dataset...")

    saved_count = 0
    batch_records = []
    seen_names = set()

    # Pre-populate seen names from existing DB
    try:
        existing_repos = db.query(Repository.name_with_owner).all()
        seen_names.update([r[0] for r in existing_repos])
        print(f"[ETL Pipeline] Found {len(seen_names)} existing repositories in DB.")
    except Exception as e:
        print(f"[ETL Pipeline] Note querying existing names: {e}")

    try:
        with open(dataset_path, "rb") as f:
            parser = ijson.items(f, "item")
            for raw_data in parser:
                if not isinstance(raw_data, dict):
                    continue

                processed = transform_raw_record(raw_data)
                if not processed or processed.name_with_owner in seen_names:
                    continue

                seen_names.add(processed.name_with_owner)
                
                repo_entity = Repository(
                    name_with_owner=processed.name_with_owner,
                    owner=processed.owner,
                    name=processed.name,
                    description=processed.description,
                    primary_language=processed.primary_language,
                    languages=processed.languages,
                    topics=processed.topics,
                    stars=processed.stars,
                    forks=processed.forks,
                    watchers=processed.watchers,
                    pull_requests=processed.pull_requests,
                    issues=processed.issues,
                    commits=processed.commits,
                    disk_usage_kb=processed.disk_usage_kb,
                    is_fork=processed.is_fork,
                    is_archived=processed.is_archived,
                    created_at=processed.created_at,
                    pushed_at=processed.pushed_at,
                    license=processed.license,
                    activity_score=processed.activity_score,
                    popularity_score=processed.popularity_score,
                    synthesized_text=processed.synthesized_text
                )
                batch_records.append(repo_entity)

                if len(batch_records) >= batch_size:
                    db.bulk_save_objects(batch_records)
                    db.commit()
                    saved_count += len(batch_records)
                    batch_records = []
                    elapsed = time.time() - start_time
                    rate = saved_count / max(0.001, elapsed)
                    print(f"[ETL Pipeline] Ingested & Saved {saved_count:,} repos ({rate:.1f} repos/sec)...")

                if sample_size and (saved_count + len(batch_records)) >= sample_size:
                    break

            if batch_records:
                db.bulk_save_objects(batch_records)
                db.commit()
                saved_count += len(batch_records)

        total_time = time.time() - start_time
        print(f"[ETL Pipeline] Ingestion completed: {saved_count:,} repositories loaded in {total_time:.2f}s.")
        
    except Exception as e:
        db.rollback()
        print(f"[ETL Pipeline] Error during ingestion: {e}")
        raise e
    finally:
        db.close()
