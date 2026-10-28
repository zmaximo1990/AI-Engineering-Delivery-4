"""Runtime settings loaded from the environment and the project ``.env`` file."""

import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Self

from pydantic import BeforeValidator, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PROJECT_DIR = Path(__file__).resolve().parent

# Leading document id in a file name, for example NE-100.
DOC_ID = re.compile(r"^(NE-\d+)", re.IGNORECASE)


def _comma_separated_set(value: object) -> set[str]:
    """Parse a comma-separated string, or copy a set, list, or tuple of strings."""
    if isinstance(value, str):
        parts = value.split(",")
    elif isinstance(value, (set, frozenset, list, tuple)):
        parts = [str(part) for part in value]
    else:
        raise ValueError("expected a comma-separated string or a set of strings")
    return {part.strip() for part in parts if part.strip()}


class Settings(BaseSettings):
    """Process settings. Empty environment values fall back to the field default."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    default_max_tokens: int = 512
    default_temperature: float = 0.0
    retry_attempts: int = 3
    truncated_reasons: Annotated[
        set[str], NoDecode, BeforeValidator(_comma_separated_set)
    ] = Field(default_factory=lambda: {"length", "max_tokens"})
    gemini_max_tokens: int = 1024

    # Chunks of at least 500 tokens with 50 tokens of overlap.
    chunk_size: int = 500
    chunk_overlap: int = 50
    model_encoding: str = "cl100k_base"

    # Chunks a vector-only search returns (semantic_search and the default
    # similarity retriever). Bounded to 3–5 so the prompt stays small.
    # Unlike hybrid_k, this does not fuse BM25 with vectors.
    top_k: int = 4
    # Documents kept after the hybrid retriever fuses BM25 and vector search.
    # Independent of top_k. Defaults for the top 5 combined results.
    hybrid_k: int = 5
    # Relevance score in [0, 1]. Higher keeps only closer matches. 0 keeps them all.
    score_threshold: float = 0.5

    huggingface_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_provider: str = "huggingface"
    embedding_batch_size: int = 100
    embedding_dimension: int | None = None
    index_name: str = "delivery4-index"
    namespace: str = "delivery4"
    vector_store: str = "pinecone"
    collection_name: str = "techcorp_policies"
    data_dir: Path = PROJECT_DIR / "data"
    vectorstore_dir: Path = PROJECT_DIR / "vectorstore"
    default_provider: str = Field(default="openai", validation_alias="LLM_PROVIDER")

    @model_validator(mode="before")
    @classmethod
    def _blank_as_unset(cls, data: object) -> object:
        """Treat blank strings as missing so the field default is used."""
        if not isinstance(data, dict):
            return data
        cleaned: dict[object, object] = {}
        for key, value in data.items():
            if isinstance(value, str):
                value = value.strip()
                if value == "":
                    continue
            cleaned[key] = value
        return cleaned

    @field_validator("chunk_size")
    @classmethod
    def _at_least_500(cls, value: int) -> int:
        return max(500, value)

    @field_validator("top_k")
    @classmethod
    def _clamp_top_k(cls, value: int) -> int:
        return min(5, max(3, value))

    @field_validator("hybrid_k")
    @classmethod
    def _at_least_one(cls, value: int) -> int:
        return max(1, value)

    @field_validator("score_threshold")
    @classmethod
    def _clamp_score_threshold(cls, value: float) -> float:
        return min(1.0, max(0.0, value))

    @field_validator("embedding_batch_size")
    @classmethod
    def _at_least_one_batch(cls, value: int) -> int:
        return max(1, value)

    @field_validator("embedding_dimension", mode="before")
    @classmethod
    def _zero_dimension_is_unset(cls, value: object) -> object:
        if value == 0 or value == "0":
            return None
        return value

    @field_validator("vector_store", mode="before")
    @classmethod
    def _lower_vector_store(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("data_dir", "vectorstore_dir", mode="before")
    @classmethod
    def _resolve_project_path(cls, value: object) -> Path:
        path = value if isinstance(value, Path) else Path(str(value).strip())
        if not path.is_absolute():
            path = PROJECT_DIR / path
        return path

    @model_validator(mode="after")
    def _cap_chunk_overlap(self) -> Self:
        cap = self.chunk_size - 1
        if self.chunk_overlap > cap:
            object.__setattr__(self, "chunk_overlap", cap)
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, loaded once."""
    return Settings()


settings = get_settings()
