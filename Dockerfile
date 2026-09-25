FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first for better Docker layer caching
COPY requirements.txt .

# Install CPU-only PyTorch and compatible NumPy.
# PyTorch is installed separately so CUDA packages are never pulled in.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.7.1+cpu \
        numpy==1.26.4 && \
    pip install --no-cache-dir -r requirements.txt

# Verify the ML dependency stack before downloading models.
RUN python -c "\
import torch; \
import numpy; \
import transformers; \
import sentence_transformers; \
print('[Dockerfile] PyTorch:', torch.__version__); \
print('[Dockerfile] NumPy:', numpy.__version__); \
print('[Dockerfile] Transformers:', transformers.__version__); \
print('[Dockerfile] SentenceTransformers:', sentence_transformers.__version__)"

# Pre-download the dense embedding model.
# This prevents a model download/cold start when the container starts.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu'); \
print('[Dockerfile] Dense model cached.')"

# Pre-download the sparse BM25 model.
RUN python -c "\
from fastembed import SparseTextEmbedding; \
list(SparseTextEmbedding('Qdrant/bm25').embed(['warmup'])); \
print('[Dockerfile] Sparse BM25 model cached.')"

# Copy application source after dependencies/models for better caching.
COPY . .

# Application data directory
RUN mkdir -p data

EXPOSE 8000
EXPOSE 8501

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]