# 🔎 GitHub Semantic & Hybrid Search Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![Sentence-Transformers](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-orange.svg)](https://sbert.net/)
[![PySpark](https://img.shields.io/badge/Apache%20Spark-3.5%2B-E25A1C.svg)](https://spark.apache.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade **AI-powered Natural Language Search Engine for GitHub Repositories**. This platform moves beyond rigid keyword matching to understand developer search intent through **dense transformer embeddings**, **BM25 lexical retrieval**, and **multi-factor activity & popularity ranking**.

---

## 📑 Table of Contents
- [1. Architecture Overview](#1-architecture-overview)
- [2. Deep Dive Architectural Q&A](#2-deep-dive-architectural-qa)
  - [Q1: What does this project do?](#q1-what-does-this-project-do)
  - [Q2: What is the dataset & what does it contain?](#q2-what-is-the-dataset--what-does-it-contain)
  - [Q3: How do you extract the dataset?](#q3-how-do-you-extract-the-dataset)
  - [Q4: What preprocessing, transformations, filtering & sanitization are done?](#q4-what-preprocessing-transformations-filtering--sanitization-are-done)
  - [Q5: How and where is processed data stored?](#q5-how-and-where-is-processed-data-stored)
  - [Q6: What is Apache Spark and how is it used here?](#q6-what-is-apache-spark-and-how-is-it-used-here)
  - [Q7: How do we convert data into vector embeddings?](#q7-how-do-we-convert-data-into-vector-embeddings)
  - [Q8: How do we convert natural language queries into vector embeddings?](#q8-how-do-we-convert-natural-language-queries-into-vector-embeddings)
  - [Q9: How do we match the two embeddings?](#q9-how-do-we-match-the-two-embeddings)
  - [Q10: What is BM25, Dense Vector Similarity & Metadata-Based Ranking?](#q10-what-is-bm25-dense-vector-similarity--metadata-based-ranking)
- [3. End-to-End Query Lifecycle](#3-end-to-end-query-lifecycle)
- [4. Ranking Formula & Fusion](#4-ranking-formula--fusion)
- [5. Interactive Streamlit UI & Pagination](#5-interactive-streamlit-ui--pagination)
- [6. FastAPI REST Endpoints](#6-fastapi-rest-endpoints)
- [7. Local Setup & Execution](#7-local-setup--execution)
- [8. Project Structure](#8-project-structure)
- [9. Benchmark Queries](#9-benchmark-queries)

---

## 1. Architecture Overview

```
+-------------------------------------------------------------------------------+
|                             GitHub Search Engine                              |
+--------------------------------------+----------------------------------------+
                                       |
    +----------------------------------+-----------------------------------+
    |                                                                      |
+-------------------+                                            +--------------------+
|  Data Ingestion   |                                            | Natural Language   |
|   & ETL Engine    |                                            |   Search Engine    |
+---------+---------+                                            +---------+----------+
          |                                                                |
   [repo_metadata.json]                                            [User Query]
          |                                                                |
   +------+----------------+                                               |
   |                       |                                               v
   v                       v                                       +------------------+
[ijson Batch Stream]  [Apache Spark ETL]                           | Sentence-Transf. |
   |                       |                                       | (all-MiniLM-L6)  |
   +------+----------------+                                       +--------+---------+
          |                                                                 |
          v                                                                 v
+------------------------+                                         +------------------+
| Preprocessing & Clean  |                                         | 384-d Unit Vector|
| - HTML / Whitespace    |                                         +--------+---------+
| - Topic Extraction     |                                                  |
| - Activity/Popularity  |                                                  v
| - Document Synthesis   |                                         +------------------+
+---------+--------------+                                         | Multi-Factor     |
          |                                                        | Retrieval Engine |
          v                                                        +--------+---------+
+------------------------+                                                  |
| Storage Layer          |                                                  |
| - SQLite / PostgreSQL  |<-------------------------------------------------+
| - Pickled Matrix Cache | (Vector dot product + BM25 score + Metadata rank)
+------------------------+
                                       |
                                       v
                     +-----------------------------------+
                     |  FastAPI Backend (/api/search)    |
                     +-----------------+-----------------+
                                       |
                                       v
                     +-----------------------------------+
                     | Streamlit Web UI with Pagination  |
                     +-----------------------------------+
```

---

## 2. Deep Dive Architectural Q&A

### Q1: What does this project do?
This project is an **AI-powered Semantic & Hybrid Search Engine for GitHub Repositories**. 

Standard keyword search often fails when users describe what a tool *does* rather than what it is *named* (e.g. searching *"fast async web framework in python"* rather than searching for the exact token `"fastapi"`). This engine bridges that semantic gap by projecting developer intent and repository metadata into a shared embedding space, combining:
1. **Semantic Dense Vector Search** for conceptual similarity and synonym matching.
2. **Lexical BM25 Okapi Search** for exact library, tool, and keyword precision.
3. **Repository Quality & Maintenance Signals** to surface reliable, actively maintained libraries over abandoned forks.

---

### Q2: What is the dataset & what does it contain?
The raw dataset is `repo_metadata.json` (a **~3.28 GB JSON array** containing hundreds of thousands of GitHub repositories).

Each record contains rich operational and descriptive attributes:
* **Identification:** `name`, `owner`, `nameWithOwner`
* **Textual Context:** `description`, `topics` (tag list), `license`
* **Technology Stack:** `primaryLanguage`, `languages` (language breakdown)
* **Popularity & Social Proof:** `stars`, `forks`, `watchers`
* **Maintenance & Activity:** `defaultBranchCommitCount` (commits), `pullRequests`, `issues`, `createdAt`, `pushedAt`, `diskUsageKb`, `isFork`, `isArchived`.

---

### Q3: How do you extract the dataset?
Because the JSON dataset is ~3.28 GB, standard `json.load()` risks exhausting system RAM. The project provides two extraction pathways:

1. **Memory-Efficient Streaming Extraction (`src/etl/pipeline.py`):**
   * Uses the iterative JSON parser `ijson.items(f, 'item')` to stream through the top-level array without loading the full file into memory.
   * Batches records in memory (e.g. 500 items/batch) and performs bulk database inserts using SQLAlchemy `bulk_save_objects()`.
2. **Distributed Apache Spark Extraction (`src/etl/spark_etl.py`):**
   * Uses PySpark with an explicit `StructType` schema (`spark.read.schema(schema).json(input_path)`), distributing parsing across worker cores and partitions.

---

### Q4: What preprocessing, transformations, filtering & sanitization are done?
Located in `src/etl/transformers.py` and `src/etl/spark_etl.py`:

* **Sanitization & Text Cleaning:**
  * Strips HTML markup (`<[^>]+>`) from repository descriptions.
  * Normalizes whitespace, carriage returns, and tabs (`\r\n\t`).
  * Replaces missing/null fields with safe defaults (`"Unknown"` language, `0` stars, `0` commits).
* **Topic & Tag Extraction:**
  * Flattens and lowercases nested topic objects (`[t.get('name').lower() for t in topics]`).
* **Timestamp Normalization:**
  * Converts ISO 8601 strings (`pushedAt`, `createdAt`) into timezone-aware UTC `datetime` objects.
* **Feature Engineering:**
  * **Activity Score ($[0.0, 1.0]$):**
    $$\text{Activity Score} = 0.35 \cdot \text{norm\_commits} + 0.25 \cdot \text{norm\_interactions} + 0.40 \cdot e^{-\lambda \cdot \Delta\text{days}}$$
    Where $\text{norm\_commits} = \min(1.0, \log_{10}(\text{commits}+1)/4.0)$ and $\Delta\text{days}$ is days since last push ($\lambda = 0.0019$).
  * **Popularity Score ($[0.0, 1.0]$):** Weighted combination of $\log_{10}(\text{stars})$ and $\log_{10}(\text{forks})$.
* **Document Synthesis:**
  * Synthesizes an enriched document combining `Repository`, `Primary Language`, `Languages`, `Topics and tags`, `Description`, and `Activity Status` for dense vector embedding and BM25 indexing.

---

### Q5: How and where is processed data stored?
* **Relational Database (SQLAlchemy / SQLite / PostgreSQL):**
  * Stored in the `Repository` table (`src/db/models.py`). Defaults locally to `data/github_search.db` (SQLite) or PostgreSQL via `DATABASE_URL`.
* **Search Index & Embeddings Cache:**
  * Persisted in `data/search_index.pkl` containing:
    * `embeddings`: Precomputed 384-dimensional dense vectors stored as a 2D NumPy float32 matrix `(N, 384)`.
    * `repo_ids`: Integer primary key list mapping vector rows to repositories.
    * `repo_lookup`: In-memory metadata dictionary for instant retrieval.
* **Partitioned Parquet (Spark Mode):**
  * Exported partitioned by `primaryLanguage` (`df.write.partitionBy("primaryLanguage").parquet(...)`).

---

### Q6: What is Apache Spark and how is it used here?
* **What is Apache Spark?**
  * Apache Spark is an open-source, distributed general-purpose cluster computing engine optimized for in-memory processing and massive-scale ETL workflows.
* **How is it used here (`src/etl/spark_etl.py`)?**
  * Provides a scalable distributed ETL alternative for processing millions of repositories across multiple cores or a compute cluster.
  * Uses Spark SQL Catalyst optimizer functions (`F.col`, `F.log10`, `F.datediff`, `F.exp`, `F.expr`) to perform string cleaning, nested array extraction, and activity feature engineering across distributed partitions without Python serialization bottlenecks.

---

### Q7: How do we convert data into vector embeddings?
Implemented in `src/search/embedder.py`:
1. Each repository's synthesized document (name, language, topics, description, activity hint) is fed into the **`sentence-transformers/all-MiniLM-L6-v2`** model.
2. The transformer encodes text into **384-dimensional dense vectors**:
   ```python
   embeddings = model.encode(texts, batch_size=128, normalize_embeddings=True, convert_to_numpy=True)
   ```
3. Vectors are normalized to unit length ($||\mathbf{v}||_2 = 1.0$) and stored in an $(N, 384)$ float32 NumPy matrix.

---

### Q8: How do we convert natural language queries into vector embeddings?
Implemented in `Embedder.embed_query()` (`src/search/embedder.py`):
1. When a developer submits a query (e.g. `"distributed kubernetes cluster monitoring"`), the text is tokenized and passed into the exact same `all-MiniLM-L6-v2` transformer.
2. The model outputs a single **384-dimensional unit vector** $\mathbf{q} \in \mathbb{R}^{384}$ ($||\mathbf{q}||_2 = 1.0$) capturing the semantic intent of the query.

---

### Q9: How do we match the two embeddings?
Implemented in `HybridSearchEngine.search()` (`src/search/engine.py`):

* **Cosine Similarity via Dot Product:**
  Because both repository vectors $\mathbf{d}_i$ and query vector $\mathbf{q}$ are $L_2$-normalized unit vectors, cosine similarity equals the vector dot product:
  $$\text{Cosine Similarity}(\mathbf{q}, \mathbf{d}_i) = \frac{\mathbf{q} \cdot \mathbf{d}_i}{\|\mathbf{q}\| \|\mathbf{d}_i\|} = \mathbf{q} \cdot \mathbf{d}_i$$
* **Vectorized NumPy Computation:**
  ```python
  sims = np.dot(self.embeddings, query_vec) # Fast (N, 384) x (384,) matrix-vector multiplication
  sims = np.clip((sims + 1.0) / 2.0, 0.0, 1.0) # Scaled to [0.0, 1.0]
  ```

---

### Q10: What is BM25, Dense Vector Similarity & Metadata-Based Ranking?

| Mechanism | Description | Role in GitHub Search Engine |
|---|---|---|
| **BM25 (Best Matching 25)** | Probabilistic **lexical / keyword ranking** algorithm that scores documents based on exact token matches, IDF rarity, and document length normalization. | Catches exact library names (e.g. `fastapi`, `react`, `playwright`) and specific technical terms (`src/search/bm25.py`). |
| **Dense Vector Similarity** | **Semantic / concept-level search** that maps queries and documents into latent space. | Connects user concepts with related tags and synonyms (e.g. *"computer vision"* $\rightarrow$ *"yolo"*, *"darknet"*, *"opencv"*). |
| **Metadata-Based Ranking** | **Quality and maintenance scoring** evaluating repository popularity and activity. | Boosts popular, actively maintained repos ($\log_{10}\text{stars}$, commit recency, PR activity) and penalizes abandoned projects. |

---

## 3. End-to-End Query Lifecycle

```
User enters natural language query: "fast async web framework in python"
    │
    ▼
1. Embedder: Transforms query into 384-d normalized vector (all-MiniLM-L6-v2)
    │
    ├──▶ 2. Dense Branch: np.dot(embeddings, query_vec) -> Semantic Similarity [0, 1]
    │
    ├──▶ 3. Lexical Branch: BM25Okapi token scoring -> Keyword Match Score [0, 1]
    │
    └──▶ 4. Metadata Branch: Log-scaled Stars + Activity Score + Language Match Bonus
    │
    ▼
5. Multi-Factor Fusion: Score = 0.55 * Dense + 0.30 * Lexical + 0.15 * Metadata
    │
    ▼
6. Filtering & Pagination: Apply Language, Min Stars, Sort criteria, and [offset : offset + limit]
    │
    ▼
7. FastAPI JSON Payload -> Streamlit UI Rendering with Score Breakdowns & Page Navigation
```

---

## 4. Ranking Formula & Fusion

### Default Multi-Factor Score:
$$\text{Final Score} = w_{\text{dense}} \cdot S_{\text{dense}} + w_{\text{lexical}} \cdot S_{\text{lexical}} + w_{\text{meta}} \cdot S_{\text{metadata}}$$

* $w_{\text{dense}} = 0.55$: Dense semantic similarity
* $w_{\text{lexical}} = 0.30$: BM25 lexical match
* $w_{\text{meta}} = 0.15$: Popularity, activity health, and language bonus

---

## 5. Interactive Streamlit UI & Pagination

The frontend (`src/frontend/app.py`) provides an interactive interface:

* **Natural Language Search Bar & Example Chips**: Click-to-search pre-configured complex queries.
* **Pagination & Result Controls**:
  - **Custom Page Size**: Select `10`, `15`, `25`, `50`, `100`, or `200` results per page.
  - **Direct Page Navigation**: Previous/Next buttons and direct page number selector.
  - **Result Range Indicator**: Shows `Showing 1–25 of 142 repositories (Page 1 of 6)`.
* **Sidebar Filters**:
  - Search Algorithm Switcher (Hybrid, Semantic Only, Lexical Only).
  - Primary Language filter dropdown (auto-populated from database).
  - Minimum Stars number input.
  - Sorting selector (Relevance, Stars, Activity, Recency).
* **Rich Repository Cards**:
  - Direct links to GitHub repositories.
  - Primary language badges, stars, forks, license, and activity scores.
  - Clickable topic pills (`#fastapi`, `#asyncio`, `#rest-api`).
  - **Score Breakdown Modal**: Shows detailed Semantic %, Lexical %, and Activity % contributions.

---

## 6. FastAPI REST Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/search` | Natural language repository search with filters (`q`, `mode`, `language`, `min_stars`, `sort_by`, `limit`, `offset`) |
| `GET` | `/api/repos/{owner}/{name}` | Detailed metadata inspection for a single repository |
| `GET` | `/api/languages` | List of all distinct programming languages in index |
| `GET` | `/api/stats` | Global dataset statistics, averages, and language distribution |
| `GET` | `/api/health` | Healthcheck and indexed repository count |
| `GET` | `/docs` | Interactive OpenAPI Swagger UI documentation |

---

## 7. Local Setup & Execution

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/your-username/github-search-engine.git
cd github-search-engine

# Create & activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Ingest Dataset & Build Search Index
```bash
# Ingest sample (e.g. 15,000 repositories) and build vector cache
python scripts/run_etl.py --sample-size 15000
```

### 3. Run Backend API Server
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Run Streamlit UI
```bash
streamlit run src/frontend/app.py --server.port 8501
```

---

## 8. Project Structure

```
github-search-engine/
├── data/                          # Persistent SQLite database and vector cache
│   ├── github_search.db
│   └── search_index.pkl
├── docker-compose.yml             # Container orchestration (PostgreSQL + API + UI)
├── Dockerfile                     # Container definition
├── requirements.txt               # Dependencies
├── README.md                      # Architecture, Q&A, and documentation
├── repo_metadata.json             # GitHub repository metadata dataset (3.28 GB)
├── scripts/
│   ├── run_etl.py                 # Ingestion & index build CLI script
│   └── test_search_query.py       # Terminal query verification script
├── src/
│   ├── config.py                  # App configuration & search weights
│   ├── api/
│   │   ├── main.py                # FastAPI application entrypoint
│   │   ├── routes.py              # REST API route handlers
│   │   └── schemas.py             # Pydantic request/response schemas
│   ├── db/
│   │   ├── init_db.py             # DB schema & extension initialization
│   │   ├── models.py              # SQLAlchemy Repository model
│   │   └── session.py             # Database engine & session maker
│   ├── etl/
│   │   ├── pipeline.py            # Streaming ijson batch ETL pipeline
│   │   ├── schema.py              # Raw & Processed data schemas
│   │   ├── spark_etl.py          # Distributed Apache Spark ETL module
│   │   └── transformers.py       # Sanitization, activity scoring, text synthesis
│   ├── frontend/
│   │   └── app.py                 # Streamlit interactive UI application with pagination
│   └── search/
│       ├── bm25.py                # Lexical BM25 indexing & retrieval
│       ├── embedder.py            # Dense Sentence Transformer wrapper
│       ├── engine.py              # Hybrid Search Engine orchestrator
│       └── ranker.py              # Multi-factor scoring & RRF algorithms
└── tests/
    ├── test_api.py                # FastAPI endpoint integration tests
    ├── test_etl.py                # Cleaning & activity scoring unit tests
    └── test_search.py             # Vector & BM25 retrieval tests
```

---

## 9. Benchmark Queries

| Natural Language Query | Top Match | Why Semantic + Activity Ranking Outperforms Keyword Search |
|---|---|---|
| `"fast web framework for async microservices"` | `tiangolo/fastapi` | Understands "microservices" and "async" concepts, ranking active modern frameworks above older sync libraries. |
| `"web automation testing framework for games and apps"` | `microsoft/playwright`, `wix/Detox` | Discovers end-to-end testing tools without needing exact string matches in title. |
| `"computer vision deep learning for object detection"` | `ultralytics/yolov5`, `opencv/opencv` | Connects general query intent with specialized domain tags (`yolo`, `darknet`, `cv`). |
| `"export csv and excel spreadsheet in browser"` | `jmaister/excellentexport` | Directly identifies browser-side spreadsheet export libraries. |

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
