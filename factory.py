import logging
from typing import Optional

from settings import settings
from embeddings import BaseEmbeddings, HuggingFaceEmbeddings, OpenAIEmbeddings
from llm import AnthropicClient, BaseLLMClient, GeminiClient, OpenAIClient
from schemas import EmbeddingsProvider, LLMConfig, Provider
from vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)


class LLMFactory:
    """Factory that builds an LLM client from an LLMConfig."""

    @staticmethod
    def create_client(config: LLMConfig) -> BaseLLMClient:
        """Build the client that matches ``config.provider``."""
        logger.info("Creating LLM client for provider=%s", config.provider.value)

        kwargs = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
        }

        client: Optional[BaseLLMClient] = None
        if config.provider == Provider.OPENAI:
            client = OpenAIClient(**kwargs)
        elif config.provider == Provider.ANTHROPIC:
            client = AnthropicClient(**kwargs)
        elif config.provider == Provider.GEMINI:
            # Gemini spends part of the budget on thinking tokens.
            kwargs["max_tokens"] = max(config.max_tokens, settings.gemini_max_tokens)
            client = GeminiClient(**kwargs)
        else:
            raise ValueError(f"Invalid provider: {config.provider}")
        logger.info("LLM client created: %s", type(client).__name__)
        return client


class EmbeddingsFactory:
    """Factory that builds an embeddings client from an EmbeddingsProvider."""

    @staticmethod
    def create(
        provider: EmbeddingsProvider | None = None,
        model: str | None = None,
    ) -> BaseEmbeddings:
        """Build the embeddings client that matches ``provider``.

        When ``provider`` is omitted, ``EMBEDDING_PROVIDER`` selects it.
        """
        selected = provider or EmbeddingsFactory.embedding_provider_from_env()
        logger.info("Creating embeddings client for provider=%s", selected.value)

        client: Optional[BaseEmbeddings] = None
        if selected == EmbeddingsProvider.OPENAI:
            client = OpenAIEmbeddings(model=model)
        elif selected == EmbeddingsProvider.HUGGINGFACE:
            client = HuggingFaceEmbeddings(model=model)
        else:
            raise ValueError(f"Invalid embedding provider: {selected}")
        logger.info("Embeddings client created: %s", type(client).__name__)
        return client


    @staticmethod
    def embedding_provider_from_env() -> EmbeddingsProvider:
        """
        Return the embeddings provider from the environment.
        Raise an error if the provider is not valid.
        Allowed providers are:
        - openai
        - huggingface
        """

        try:
            return EmbeddingsProvider(settings.embedding_provider)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in EmbeddingsProvider)
            raise ValueError(f"EMBEDDING_PROVIDER must be one of: {allowed}") from exc


class VectorStoreFactory:
    """Factory that returns one vector store for the process."""

    _manager: VectorStoreManager | None = None

    @staticmethod
    def get_manager() -> VectorStoreManager:
        """Return the process-wide store so the embedding model is loaded once.

        ``VECTOR_STORE`` selects Pinecone or Chroma. Later calls return the
        same instance.
        """
        if VectorStoreFactory._manager is None:
            # Imported here because vectorstore imports EmbeddingsFactory.
            from vectorstore import ChromaStoreManager, PineconeStoreManager

            if settings.vector_store == "pinecone":
                VectorStoreFactory._manager = PineconeStoreManager()
            elif settings.vector_store == "chroma":
                VectorStoreFactory._manager = ChromaStoreManager()
            else:
                raise ValueError("VECTOR_STORE must be one of: pinecone, chroma")
            logger.info(
                "Vector store created: %s",
                type(VectorStoreFactory._manager).__name__,
            )
        return VectorStoreFactory._manager
