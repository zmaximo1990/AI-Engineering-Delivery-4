"""
Hybrid retriever: BM25 plus vector similarity, fused by EnsembleRetriever.
"""

import logging

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from settings import settings
from ingestion import Ingestor
from vectorstore import get_vector_store

logger = logging.getLogger(__name__)


class Retriever:
    """Fuse lexical and vector search and return the top-k documents."""

    def __init__(self, k: int = settings.hybrid_k) -> None:
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")
        self.k = k
        self._init()

    def _init(self) -> None:
        # Load the documents from the ingestor
        documents = Ingestor().load_documents()
        if not documents:
            raise ValueError("No documents are available for retrieval")

        # Create the lexical retriever. It must be created in a separate thread to avoid blocking the main thread because the BM25Retriever is a I/O operation to read files from the filesystem.
        # lexical: BM25Retriever = await asyncio.to_thread(BM25Retriever.from_documents, documents)
        lexical = BM25Retriever.from_documents(documents)
        lexical.k = self.k

        # Create the vector retriever
        vector = get_vector_store().as_retriever(k=self.k, search_type="similarity")

        # Create the ensemble retriever
        self._ensemble = EnsembleRetriever(
            retrievers=[lexical, vector],
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
