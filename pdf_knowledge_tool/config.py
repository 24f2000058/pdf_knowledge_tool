import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# --- Model Configuration ---
# Options: "transformers" or "ollama"
LLM_BACKEND = os.getenv("LLM_BACKEND", "transformers")

# Model Name
# Using standard Transformers model
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
# Note: Ensure this repo exists and is public on HF.



# --- Retrieval Configuration ---
# Safety limit for total tokens in context
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "2000"))

# Number of initial candidates to fetch from Hybrid Search
TOP_N_RETRIEVAL = int(os.getenv("TOP_N_RETRIEVAL", "30"))

# Number of results to keep after Re-ranking (if enabled)
TOP_M_RERANK = int(os.getenv("TOP_M_RERANK", "5"))

# Enable/Disable Reranker
# Set to "true" or "1" to enable
USE_RERANKER = os.getenv("USE_RERANKER", "false").lower() in ("true", "1", "yes")

# Minimum vector similarity score to consider a chunk relevant
MIN_SIMILARITY_THRESHOLD = float(os.getenv("MIN_SIMILARITY_THRESHOLD", "0.2"))

# --- Preview Configuration ---
# Enable/Disable Docling Page Previews
ENABLE_PAGE_PREVIEWS = os.getenv("ENABLE_PAGE_PREVIEWS", "true").lower() in ("true", "1", "yes")

# Max pages to render per document to avoid storage bloat
PREVIEW_MAX_PAGES = int(os.getenv("PREVIEW_MAX_PAGES", "30"))

# Ollama API URL
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")

from pathlib import Path

# --- Paths ---
DATA_DIR = Path("./data")
METADATA_DB_PATH = DATA_DIR / "metadata.json"
CHROMA_DB_PATH = DATA_DIR / "chroma_db"
IMAGES_DIR = DATA_DIR / "images"
