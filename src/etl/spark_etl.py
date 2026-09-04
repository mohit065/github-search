"""
Apache Spark / PySpark ETL Pipeline for GitHub Repository Metadata.
Provides distributed extraction, transformation, feature engineering, and loading.
"""

from typing import Optional

def build_spark_session(app_name: str = "GitHubRepoETL"):
    """Initialize SparkSession with optimal configurations for JSON processing."""
    try:
        from pyspark.sql import SparkSession
        return (
            SparkSession.builder
            .appName(app_name)
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .getOrCreate()
        )
    except ImportError:
        raise ImportError("PySpark is required for spark_etl. Please install pyspark to run Spark jobs.")

def run_spark_pipeline(
    input_path: str,
    output_path: str,
    sample_fraction: Optional[float] = None,
    save_format: str = "parquet"
):
    """
    Execute distributed PySpark ETL pipeline:
    1. Read JSON dataset at scale
    2. Extract nested arrays (languages, topics)
    3. Filter out invalid or corrupted entries
    4. Compute engineered features (activity_score, popularity_score)
    5. Generate synthesized embedding document column
    6. Write partitioned structured output (Parquet / PostgreSQL)
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        StructType, StructField, StringType, IntegerType, 
        BooleanType, ArrayType, MapType
    )
    
    spark = build_spark_session()
    print(f"[Spark ETL] Reading dataset from: {input_path}")
    
    # Define explicit schema for fast parsing
    schema = StructType([
        StructField("owner", StringType(), True),
        StructField("name", StringType(), True),
        StructField("nameWithOwner", StringType(), True),
        StructField("description", StringType(), True),
        StructField("primaryLanguage", StringType(), True),
        StructField("languages", ArrayType(MapType(StringType(), StringType())), True),
        StructField("topics", ArrayType(MapType(StringType(), StringType())), True),
        StructField("stars", IntegerType(), True),
        StructField("forks", IntegerType(), True),
        StructField("watchers", IntegerType(), True),
        StructField("pullRequests", IntegerType(), True),
        StructField("issues", IntegerType(), True),
        StructField("diskUsageKb", IntegerType(), True),
        StructField("isFork", BooleanType(), True),
        StructField("isArchived", BooleanType(), True),
        StructField("createdAt", StringType(), True),
        StructField("pushedAt", StringType(), True),
        StructField("defaultBranchCommitCount", IntegerType(), True),
        StructField("license", StringType(), True),
    ])
    
    df = spark.read.schema(schema).json(input_path)
    
    if sample_fraction is not None and 0.0 < sample_fraction < 1.0:
        df = df.sample(fraction=sample_fraction, seed=42)
        
    # Transformations
    # 1. Clean strings & Fill Defaults
    df_clean = (
        df.filter(F.col("nameWithOwner").isNotNull())
          .withColumn("description", F.coalesce(F.trim(F.col("description")), F.lit("")))
          .withColumn("primaryLanguage", F.coalesce(F.col("primaryLanguage"), F.lit("Unknown")))
          .withColumn("stars", F.coalesce(F.col("stars"), F.lit(0)))
          .withColumn("forks", F.coalesce(F.col("forks"), F.lit(0)))
          .withColumn("pullRequests", F.coalesce(F.col("pullRequests"), F.lit(0)))
          .withColumn("issues", F.coalesce(F.col("issues"), F.lit(0)))
          .withColumn("commits", F.coalesce(F.col("defaultBranchCommitCount"), F.lit(0)))
    )
    
    # 2. Extract Topic names
    df_transformed = df_clean.withColumn(
        "topic_names",
        F.expr("transform(topics, x -> lower(x['name']))")
    )
    
    # 3. Compute Feature Engineering Metrics
    # Activity score calculation
    df_featured = df_transformed.withColumn(
        "norm_commits",
        F.least(F.lit(1.0), F.log10(F.col("commits") + 1.0) / 4.0)
    ).withColumn(
        "norm_interactions",
        F.least(F.lit(1.0), F.log10(F.col("pullRequests") + F.col("issues") + 1.0) / 4.0)
    ).withColumn(
        "days_since_push",
        F.greatest(F.lit(0.0), F.datediff(F.to_date(F.lit("2026-09-01")), F.to_date(F.col("pushedAt"))))
    ).withColumn(
        "recency_factor",
        F.exp(F.lit(-0.0019) * F.col("days_since_push"))
    ).withColumn(
        "activity_score",
        F.round(
            (F.lit(0.35) * F.col("norm_commits")) + 
            (F.lit(0.25) * F.col("norm_interactions")) + 
            (F.lit(0.40) * F.coalesce(F.col("recency_factor"), F.lit(0.0))),
            4
        )
    ).withColumn(
        "popularity_score",
        F.round(
            (F.lit(0.75) * F.least(F.lit(1.0), F.log10(F.col("stars") + 1.0) / 5.5)) +
            (F.lit(0.25) * F.least(F.lit(1.0), F.log10(F.col("forks") + 1.0) / 4.5)),
            4
        )
    )
    
    # 4. Construct Synthesized Document Column
    df_final = df_featured.withColumn(
        "synthesized_text",
        F.concat_ws(
            " | ",
            F.concat(F.lit("Repository: "), F.col("nameWithOwner")),
            F.concat(F.lit("Primary Language: "), F.col("primaryLanguage")),
            F.concat(F.lit("Topics: "), F.concat_ws(", ", F.col("topic_names"))),
            F.concat(F.lit("Description: "), F.col("description")),
            F.concat(F.lit("Stars: "), F.col("stars")),
            F.concat(F.lit("Activity: "), F.col("activity_score"))
        )
    )
    
    print(f"[Spark ETL] Writing transformed output to: {output_path} (Format: {save_format})")
    df_final.write.mode("overwrite").format(save_format).save(output_path)
    print("[Spark ETL] Pipeline finished successfully.")
    spark.stop()

if __name__ == "__main__":
    import sys
    in_p = sys.argv[1] if len(sys.argv) > 1 else "repo_metadata.json"
    out_p = sys.argv[2] if len(sys.argv) > 2 else "data/spark_transformed_parquet"
    run_spark_pipeline(in_p, out_p)
