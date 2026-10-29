"""
Hybrid retriever: BM25 plus vector similarity, fused by EnsembleRetriever.
"""

import asyncio
import logging
from typing import ClassVar

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from settings import settings
from ingestion import Ingestor
from factory import VectorStoreFactory

logger = logging.getLogger(__name__)


class Retriever:
    """
    Fuse lexical and vector search and return the top-k documents.

    One instance is built per ``k`` and reused. ``aget`` runs that build in a
    worker thread so parsing and the BM25 index stay off the event loop.

    NOTE: it is the `RAGSystem` class requested from the spec.
    """

    _instances: ClassVar[dict[int, "Retriever"]] = {}
    _lock: ClassVar[asyncio.Lock | None] = None

    def __init__(self, k: int = settings.hybrid_k) -> None:
        self._validate_k(k)
        self.k = k
        self._init()

    @staticmethod
    def _validate_k(k: int) -> None:
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")

    @classmethod
    def get(cls, k: int = settings.hybrid_k) -> "Retriever":
        """Return the process-wide retriever for ``k``, building it on this thread (sync)."""
        cls._validate_k(k)
        instance = cls._instances.get(k)
        if instance is None:
            instance = cls(k)
            cls._instances[k] = instance
        return instance

    @classmethod
    async def aget(cls, k: int = settings.hybrid_k) -> "Retriever":
        """Return the process-wide retriever for ``k``.

        The first call builds it in a worker thread (async). Later calls return the
        same instance without reading the corpus again.
        """
        cls._validate_k(k)
        instance = cls._instances.get(k)
        if instance is not None:
            return instance

        if cls._lock is None:
            cls._lock = asyncio.Lock()
        async with cls._lock:
            instance = cls._instances.get(k)
            if instance is None:
                instance = await asyncio.to_thread(cls, k)
                cls._instances[k] = instance
            return instance

    def _init(self) -> None:
        # Load the documents from the ingestor
        documents = Ingestor().load_documents()
        if not documents:
            raise ValueError("No documents are available for retrieval")

        lexical_retriever = BM25Retriever.from_documents(documents)
        lexical_retriever.k = self.k

        # Get the vector retriever
        vector_retriever = VectorStoreFactory.get_manager().as_retriever(k=self.k, search_type="similarity")

        # Create the ensemble retriever
        self._ensemble = EnsembleRetriever(
            retrievers=[lexical_retriever, vector_retriever],
            # Distribute the weight between the two retrievers
            weights=[0.5, 0.5],
        )
        logger.info("Hybrid retriever ready (k=%d, weights=0.5/0.5)", self.k)

    def invoke(self, question: str) -> list[Document]:
        """Return at most ``k`` documents from the hybrid search."""
        return list(self._ensemble.invoke(question))[: self.k]

    async def ainvoke(self, question: str) -> list[Document]:
        """Return at most ``k`` documents, searching BM25 and vectors together."""
        documents = await self._ensemble.ainvoke(question)
        return list(documents)[: self.k]
