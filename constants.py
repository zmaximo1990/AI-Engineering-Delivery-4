import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")


def _env_text(name: str) -> str:
    return os.environ.get(name, "").strip()


def _int_env(name: str, default: int) -> int:
    raw = _env_text(name)
    if not raw:
        return default
    return int(raw)


def _float_env(name: str, default: float) -> float:
    raw = _env_text(name)
    if not raw:
        return default
    return float(raw)


def _set_env(name: str, default: set[str]) -> set[str]:
    raw = _env_text(name)
    if not raw:
        return set(default)
    return {part.strip() for part in raw.split(",") if part.strip()}


def _project_path(name: str) -> Path:
    path = Path(name)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path


def _bounded_int(name: str, default: int, low: int, high: int) -> int:
    return min(high, max(low, _int_env(name, default)))


def _bounded_float(name: str, default: float, low: float, high: float) -> float:
    return min(high, max(low, _float_env(name, default)))


DEFAULT_MAX_TOKENS = _int_env("DEFAULT_MAX_TOKENS", 512)
DEFAULT_TEMPERATURE = _float_env("DEFAULT_TEMPERATURE", 0.0)
RETRY_ATTEMPTS = _int_env("RETRY_ATTEMPTS", 3)
TRUNCATED_REASONS = _set_env("TRUNCATED_REASONS", {"length", "max_tokens"})
GEMINI_MAX_TOKENS = _int_env("GEMINI_MAX_TOKENS", 1024)

# Spec: chunks of at least 500 tokens with 50 tokens of overlap.
CHUNK_SIZE = max(500, _int_env("CHUNK_SIZE", 500))
CHUNK_OVERLAP = min(_int_env("CHUNK_OVERLAP", 50), CHUNK_SIZE - 1)
MODEL_ENCODING = _env_text("MODEL_ENCODING") or "cl100k_base"

# Chunks a vector-only search returns (semantic_search and the default
# similarity retriever). Bounded to 3–5 so the prompt stays small.
# Unlike HYBRID_K, this does not fuse BM25 with vectors.
TOP_K = _bounded_int("TOP_K", 4, 3, 5)
# Documents kept after the hybrid retriever fuses BM25 and vector search.
# Independent of TOP_K. Defaults for the top 5 combined results.
HYBRID_K = max(1, _int_env("HYBRID_K", 5))
# Relevance score in [0, 1]. Higher keeps only closer matches. 0 keeps them all.
SCORE_THRESHOLD = _bounded_float("SCORE_THRESHOLD", 0.5, 0.0, 1.0)

HUGGINGFACE_EMBEDDING_MODEL = (
    _env_text("HUGGINGFACE_EMBEDDING_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
)
OPENAI_EMBEDDING_MODEL = _env_text("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small"
EMBEDDING_PROVIDER = _env_text("EMBEDDING_PROVIDER") or "huggingface"
EMBEDDING_BATCH_SIZE = max(1, _int_env("EMBEDDING_BATCH_SIZE", 100))
EMBEDDING_DIMENSION = _int_env("EMBEDDING_DIMENSION", 0) or None
INDEX_NAME = _env_text("INDEX_NAME") or "delivery4-index"
NAMESPACE = _env_text("NAMESPACE") or "delivery4"
VECTOR_STORE = (_env_text("VECTOR_STORE") or "pinecone").lower()
COLLECTION_NAME = _env_text("COLLECTION_NAME") or "techcorp_policies"
DATA_DIR = _project_path(_env_text("DATA_DIR") or "data")
VECTORSTORE_DIR = _project_path(_env_text("VECTORSTORE_DIR") or "vectorstore")
DEFAULT_PROVIDER = _env_text("LLM_PROVIDER") or "openai"


# TODO: create a Config or Settings class