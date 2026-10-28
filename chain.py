"""LCEL RAG chain: retrieve, attach context, generate, assemble sources."""

import asyncio
import logging

from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from openai import LengthFinishReasonError
from pydantic import ValidationError

from llm import BaseLLMClient
from constants import DEFAULT_PROVIDER, RETRY_ATTEMPTS, TRUNCATED_REASONS
from factory import LLMFactory
from prompts import prompt
from retriever import Retriever
from schemas import LLMAnswer, LLMConfig, Provider, RAGResponse

logger = logging.getLogger(__name__)


def _finish_reason(raw) -> str | None:
    metadata = getattr(raw, "response_metadata", None) or {}
    reason = metadata.get("finish_reason") or metadata.get("stop_reason")
    if reason is None:
        return None
    normalized = str(reason).lower()
    if normalized.endswith("max_tokens"):
        return "max_tokens"
    return normalized


def _reject_incomplete(result: dict) -> LLMAnswer:
    """Reject truncated or invalid output before it is treated as a final object.

    Raising here lets ``.with_retry()`` call the model again. A refusal is a
    valid answer: only an empty or truncated payload is rejected.
    """
    raw = result.get("raw")
    reason = _finish_reason(raw)
    logger.info("Validating structured output (finish_reason=%s)", reason)

    if reason in TRUNCATED_REASONS:
        logger.warning(
            "Model output truncated (finish_reason=%s); retrying",
            reason,
        )
        raise OutputParserException(
            f"Model output truncated before completion (finish_reason={reason})"
        )

    parsing_error = result.get("parsing_error")
    if parsing_error is not None:
        logger.warning("Structured output failed validation: %s", parsing_error)
        if isinstance(parsing_error, ValidationError):
            raise parsing_error
        raise OutputParserException(str(parsing_error))

    parsed = result.get("parsed")
    if parsed is None:
        logger.warning("Structured output missing parsed object; retrying")
        raise OutputParserException("Model returned no parsed object")

    if not isinstance(parsed, LLMAnswer):
        parsed = LLMAnswer.model_validate(parsed)

    if not parsed.answer.strip():
        logger.warning("Structured output answer is empty; retrying")
        raise OutputParserException("answer is empty")

    return parsed


async def retrieve_documents(inputs: dict) -> list[Document]:
    """Search the index and log each fragment that will enter the prompt."""

    question: str = inputs["question"]

    # retriever: Retriever = await asyncio.to_thread(Retriever)
    retriever: Retriever = Retriever()
    documents: list[Document] = await retriever.ainvoke(question)
    # documents: list[Document] = retriever.invoke(question)

    logger.info("Retrieved %d fragments", len(documents))
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")
        chunk_index = document.metadata.get("chunk_index", "?")
        logger.info(
            "Fragment %d: source=%s chunk=%s chars=%d\n%s",
            index,
            source,
            chunk_index,
            len(document.page_content),
            document.page_content,
        )
    return documents


def format_documents(documents: list[Document]) -> str:
    """Turn retrieved chunks into the CONTEXT block of the prompt."""
    if not documents:
        return "(no relevant fragments were retrieved)"
    blocks = []
    for document in documents:
        source = document.metadata.get("source", "unknown")
        blocks.append(f"[Source: {source}]\n{document.page_content}")
    return "\n\n---\n\n".join(blocks)


def assemble_response(payload: dict) -> RAGResponse:
    """Build the final object from the model answer and the retrieved metadata.

    Sources are read from the chunk metadata. The model does not choose them.
    """
    documents = list(payload.get("documents") or [])
    answer = payload.get("answer")
    if not isinstance(answer, LLMAnswer):
        answer = LLMAnswer.model_validate(answer)

    sources = sorted(
        {
            str(document.metadata.get("source", "unknown"))
            for document in documents
        }
    )
    logger.info(
        "Assembled response with %d fragments from %d sources",
        len(documents),
        len(sources),
    )
    return RAGResponse(
        answer=answer.answer,
        sources=sources,
        retrieved_fragments=len(documents),
    )


def compose_rag_chain(llm_generator: Runnable) -> Runnable:
    """
    Propagate the question, the documents, and the formatted context.

    Each ``RunnablePassthrough.assign`` adds a key and keeps the previous
    ones, so the closing ``RunnableLambda`` can still read the documents
    when it assembles the sources.
    """

    return (
        RunnablePassthrough.assign(
            documents=retrieve_documents, # run RAG
        )
        | RunnablePassthrough.assign(
            context=lambda inputs: format_documents(inputs["documents"]), # format RAG response as LLM prompt context
        )
        | RunnablePassthrough.assign(answer=llm_generator) # the LLM invoke
        | RunnableLambda(assemble_response) # assemble the final response after LLM answered
    )


def build_chain(config: LLMConfig) -> Runnable:
    """Compose retrieval, structured generation, and source assembly.

    Structured output replaces a ``{formato}`` prompt variable. The schema
    is ``LLMAnswer``; source files are attached after the model returns.
    """

    client: BaseLLMClient = LLMFactory.create_client(config)
    llm_generator: Runnable = (
        prompt
        | client.get_chain_model().with_structured_output(
            LLMAnswer, include_raw=True
        )
        | RunnableLambda(_reject_incomplete)
    ).with_retry(
        retry_if_exception_type=(
            OutputParserException,
            ValidationError,
            LengthFinishReasonError,
        ),
        stop_after_attempt=RETRY_ATTEMPTS,
    )
    return compose_rag_chain(llm_generator)


def _default_config() -> LLMConfig:
    try:
        provider = Provider(DEFAULT_PROVIDER)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in Provider)
        raise ValueError(f"LLM_PROVIDER must be one of: {allowed}") from exc
    return LLMConfig(provider=provider)


async def get_rag_response(
    query: str,
    config: LLMConfig | None = None,
) -> RAGResponse:
    """Search the local index and return a grounded answer with its sources."""
    cleaned = query.strip()
    if not cleaned:
        raise ValueError("query must not be empty")

    llm_config: LLMConfig = config or _default_config()

    logger.info(
        "get_rag_response started (provider=%s, %d chars)",
        llm_config.provider.value,
        len(cleaned),
    )
    result = await build_chain(llm_config).ainvoke({"question": cleaned})
    logger.info("get_rag_response finished")
    return result
