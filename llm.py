import logging
import os
from abc import ABC, abstractmethod

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from settings import settings
from schemas import LLMProvider

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Common interface for the LangChain chat model behind the RAG chain."""

    default_model: str
    provider: LLMProvider
    api_key_env: str

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = settings.default_max_tokens,
        temperature: float = settings.default_temperature,
    ) -> None:
        self.model = model or self.default_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._chain_model = self._build_chain_model()

    @abstractmethod
    def _build_chain_model(self) -> BaseChatModel:
        """Build the provider chat model used by the LCEL chain."""
        ...

    def get_chain_model(self) -> BaseChatModel:
        """Return the LangChain chat model for ``with_structured_output``."""
        return self._chain_model

    def _require_api_key(self) -> SecretStr:
        """Return the API key from the environment, or raise if it is missing."""
        value = os.environ.get(self.api_key_env, "").strip()
        if not value:
            raise ValueError(
                f"{self.api_key_env} is not configured. "
                "Set it in the environment before using this provider."
            )
        return SecretStr(value)


class OpenAIClient(BaseLLMClient):
    default_model = "gpt-4o-mini"
    provider = LLMProvider.OPENAI
    api_key_env = "OPENAI_API_KEY"

    def _build_chain_model(self) -> BaseChatModel:
        logger.info("Building ChatOpenAI (model=%s)", self.model)
        return ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self._require_api_key().get_secret_value(),
        )


class AnthropicClient(BaseLLMClient):
    default_model = "claude-haiku-4-5-20251001"
    provider = LLMProvider.ANTHROPIC
    api_key_env = "ANTHROPIC_API_KEY"

    def _build_chain_model(self) -> BaseChatModel:
        logger.info("Building ChatAnthropic (model=%s)", self.model)
        return ChatAnthropic(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self._require_api_key().get_secret_value(),
        )


class GeminiClient(BaseLLMClient):
    default_model = "models/gemini-3.8-flash"
    provider = LLMProvider.GEMINI
    api_key_env = "GEMINI_API_KEY"

    def _build_chain_model(self) -> BaseChatModel:
        logger.info("Building ChatGoogleGenerativeAI (model=%s)", self.model)
        return ChatGoogleGenerativeAI(
            model=self.model,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            google_api_key=self._require_api_key().get_secret_value(),
        )
