"""Embedding clients used to index and search the vector store."""

import logging
import os
from abc import ABC, abstractmethod

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings as HuggingFaceEmbeddingModel
from langchain_openai import OpenAIEmbeddings as OpenAIEmbeddingModel
from pydantic import SecretStr

from settings import settings
from schemas import EmbeddingsProvider

logger = logging.getLogger(__name__)


class BaseEmbeddings(ABC):
    """Shared contract for the embedding model used to index and search."""

    provider: EmbeddingsProvider
    default_model: str

    def __init__(self, model: str | None = None) -> None:
        self.model_name = model or self.default_model
        self._embeddings_model = self._build()

    @abstractmethod
    def _build(self) -> Embeddings:
        """Build the LangChain embeddings client for this provider."""
        ...

    def get_embeddings_model(self) -> Embeddings:
        """Return the LangChain embeddings model."""
        return self._embeddings_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents with the provider client."""
        return self._embeddings_model.embed_documents(texts)


class HuggingFaceEmbeddings(BaseEmbeddings):
    """Local sentence-transformers embeddings."""

    provider = EmbeddingsProvider.HUGGINGFACE
    default_model = settings.huggingface_embedding_model

    def _build(self) -> Embeddings:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        logger.info("Building HuggingFaceEmbeddings (model=%s)", self.model_name)
        return HuggingFaceEmbeddingModel(model_name=self.model_name)


class OpenAIEmbeddings(BaseEmbeddings):
    """OpenAI embeddings. Requires OPENAI_API_KEY."""

    provider = EmbeddingsProvider.OPENAI
    default_model = settings.openai_embedding_model
    api_key_env = "OPENAI_API_KEY"

    def _build(self) -> Embeddings:
        logger.info("Building OpenAIEmbeddings (model=%s)", self.model_name)
        return OpenAIEmbeddingModel(
            model=self.model_name,
            openai_api_key=self._require_api_key().get_secret_value(),
        )

    def _require_api_key(self) -> SecretStr:
        """Return the API key from the environment, or raise if it is missing."""
        value = os.environ.get(self.api_key_env, "").strip()
        if not value:
            raise ValueError(
                f"{self.api_key_env} is not configured. "
                "Set it in the environment before using this provider."
            )
        return SecretStr(value)
