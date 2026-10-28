"""Vector store interface and the Chroma and Pinecone backends."""

import logging
import os
import sqlite3
import urllib3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

import chromadb
from chromadb.api.models import Collection as ChromaDBCollection
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import ChromaError
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeException, UnauthorizedException

from settings import settings
from embeddings import BaseEmbeddings
from factory import EmbeddingsFactory

logger = logging.getLogger(__name__)

# Disk failures, the local SQLite store, and Chroma server or internal errors.
_STORE_ERRORS = (OSError, sqlite3.Error, ChromaError)
_PINECONE_ERRORS = (PineconeException, urllib3.exceptions.HTTPError, OSError)
_RETRIEVER_SEARCH_TYPES = {"similarity", "similarity_score_threshold"}


def _l2_relevance_score(distance: float) -> float:
    """Map Chroma's squared L2 distance to a relevance score in ``[0, 1]``.

    For normalized embeddings, cosine similarity is ``1 - distance / 2``.
    """
    return max(0.0, min(1.0, 1.0 - distance / 2.0))


class VectorStoreError(Exception):
    """A vector store read or write failed, or the embedding model does not match."""


class VectorStoreManager(ABC):
    """Contract for indexing and similarity search.

    Indexing and search share one embedding model. A different model would
    make distances meaningless. Concrete clients such as Chroma or Pinecone
    implement persistence and queries.
    """

    def __init__(self, embeddings: BaseEmbeddings | None = None) -> None:
        embeddings_client: BaseEmbeddings = embeddings or EmbeddingsFactory.create()
        self.embedding_model_name = embeddings_client.model_name
        self.embeddings = embeddings_client.get_embeddings_model()
        self.init_store()

    @abstractmethod
    def init_store(self) -> VectorStore:
        """Create this provider's LangChain vector store.

        The constructor assigns the result to ``self.store``. Set provider
        configuration on the instance before calling ``super().__init__``.
        """
        ...

    @staticmethod
    def create_metadata(
        source: str,
        chunk_index: int,
        text: str,
        doc_id: str,
        asset_format: str,
        page: int | None = None,
    ) -> dict[str, Any]:
        """Build the metadata record shared by Chroma and Pinecone.

        ``page`` is omitted when the asset has no page (DOCX or plain text).
        Values stay scalar because Pinecone rejects nested objects and nulls.
        """
        metadata: dict[str, Any] = {
            "source": source,
            "doc_id": doc_id,
            "format": asset_format.lstrip(".").lower(),
            "chunk_index": chunk_index,
            "char_count": len(text),
        }
        if page is not None:
            metadata["page"] = page
        return metadata

    def _batches(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ):
        """Yield equal-sized slices for embedding and writing."""
        if not (len(ids) == len(documents) == len(metadatas)):
            raise ValueError("ids, documents, and metadatas must have the same length")
        size = settings.embedding_batch_size
        for start in range(0, len(ids), size):
            end = start + size
            yield ids[start:end], documents[start:end], metadatas[start:end]

    def _embedding_dimension(self) -> int:
        """Read the dimension from one embedding of the configured model."""
        current = len(self.embeddings.embed_query("dimension"))
        if settings.embedding_dimension is not None and settings.embedding_dimension != current:
            raise VectorStoreError(
                f"EMBEDDING_DIMENSION is {settings.embedding_dimension}, "
                f"but '{self.embedding_model_name}' returned vectors of length {current}."
            )
        return current

    @abstractmethod
    def count(self) -> int:
        """Return the number of documents in the collection."""
        ...

    @abstractmethod
    def upsert_documents(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Insert documents, or replace them when the id already exists."""
        ...

    @abstractmethod
    def semantic_search(
        self,
        query_text: str,
        n_results: int = settings.top_k,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the nearest documents. Lower distance means a closer match."""
        ...

    @abstractmethod
    def delete_by_id(self, ids: list[str]) -> None:
        """Remove documents by id."""
        ...

    def as_retriever(
        self,
        k: int = settings.top_k,
        score_threshold: float = settings.score_threshold,
        search_type: str = "similarity_score_threshold",
    ) -> VectorStoreRetriever:
        """Similarity retriever. ``k`` is clamped to the spec range of 3 to 5.

        ``score_threshold`` is a relevance score in ``[0, 1]``. A higher value
        keeps only closer fragments. ``0`` keeps every fragment the store returns.
        ``search_type="similarity"`` skips that threshold.
        """
        if search_type not in _RETRIEVER_SEARCH_TYPES:
            allowed = ", ".join(sorted(_RETRIEVER_SEARCH_TYPES))
            raise ValueError(f"search_type must be one of: {allowed}")
        top_k = min(5, max(3, k))
        threshold = min(1.0, max(0.0, score_threshold))
        search_kwargs: dict[str, Any] = {"k": top_k}
        if search_type == "similarity_score_threshold":
            search_kwargs["score_threshold"] = threshold
        logger.info(
            "Retriever ready (k=%d, search_type=%s, score_threshold=%.2f)",
            top_k,
            search_type,
            threshold,
        )
        return self.store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)


class ChromaStoreManager(VectorStoreManager):
    """CRUD wrapper around a local Chroma collection."""

    def __init__(
        self,
        persist_directory: Path | str | None = None,
        collection_name: str = settings.collection_name,
        embeddings: BaseEmbeddings | None = None,
    ) -> None:
        persist_path = Path(persist_directory or settings.vectorstore_dir)
        if not persist_path.is_absolute():
            persist_path = Path(__file__).resolve().parent / persist_path
        self.persist_path = persist_path
        self.collection_name = collection_name
        super().__init__(embeddings=embeddings)
        self.collection: ChromaDBCollection = cast(Chroma, self.store)._collection
        self._require_matching_embedding_model()
        logger.info(
            "Vector store ready at %s (collection=%s, embedding=%s, documents=%d)",
            self.persist_path,
            self.collection_name,
            self.embedding_model_name,
            self.count(),
        )

    def init_store(self) -> Chroma:
        """Open the local Chroma collection used for similarity search."""
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.persist_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.store =Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            collection_metadata={"embedding_model": self.embedding_model_name},
            # Chroma's default space is squared L2, and this collection does not
            # record that metric. The score function keeps relevance inside [0, 1].
            relevance_score_fn=_l2_relevance_score,
        )
        return self.store

    def _require_matching_embedding_model(self) -> None:
        recorded = (self.collection.metadata or {}).get("embedding_model")
        if recorded and recorded != self.embedding_model_name:
            raise VectorStoreError(
                f"This collection was indexed with '{recorded}', "
                f"but the configured embedding model is '{self.embedding_model_name}'. "
                "Use the same model for indexing and search."
            )

    def count(self) -> int:
        return self.collection.count()

    def upsert_documents(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Insert documents, or replace them when the id already exists."""
        try:
            for batch_ids, batch_documents, batch_metadatas in self._batches(
                ids, documents, metadatas
            ):
                vectors = self.embeddings.embed_documents(batch_documents)
                self.collection.upsert(
                    ids=batch_ids,
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                    embeddings=vectors,
                )
        except _STORE_ERRORS as exc:
            raise VectorStoreError("Could not write documents to the vector store.") from exc

    def semantic_search(
        self,
        query_text: str,
        n_results: int = settings.top_k,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the nearest documents. Lower distance means a closer match."""
        try:
            pairs = self.store.similarity_search_with_score(
                query_text,
                k=n_results,
                filter=where,
            )
        except _STORE_ERRORS as exc:
            raise VectorStoreError("Could not query the vector store.") from exc
        return [
            {
                "document": document.page_content,
                "metadata": document.metadata,
                "distance": float(distance),
            }
            for document, distance in pairs
        ]

    def delete_by_id(self, ids: list[str]) -> None:
        """Remove documents by id."""
        try:
            self.store.delete(ids=ids)
        except _STORE_ERRORS as exc:
            raise VectorStoreError("Could not delete documents from the vector store.") from exc


class PineconeStoreManager(VectorStoreManager):
    """CRUD wrapper around a Pinecone Serverless index."""

    def __init__(
        self,
        index_name: str | None = None,
        namespace: str | None = None,
        embeddings: BaseEmbeddings | None = None,
    ) -> None:
        self.index_name = index_name or settings.index_name
        self.namespace = namespace or settings.namespace
        super().__init__(embeddings=embeddings)
        logger.info(
            "Pinecone index ready (index=%s, namespace=%s, dimension=%d, embedding=%s, documents=%d)",
            self.index_name,
            self.namespace,
            self.dimension,
            self.embedding_model_name,
            self.count(),
        )

    def _init_client(self) -> Pinecone:
        """Create the client and confirm the API key can reach Pinecone."""
        api_key = os.environ.get("PINECONE_API_KEY", "").strip()
        if not api_key:
            raise VectorStoreError(
                "Set PINECONE_API_KEY in the environment or in the .env file."
            )
        try:
            client = Pinecone(api_key=api_key)
            client.list_indexes()
        except UnauthorizedException as exc:
            raise VectorStoreError(
                "Pinecone rejected the API key. Check PINECONE_API_KEY."
            ) from exc
        except _PINECONE_ERRORS as exc:
            raise VectorStoreError(
                "Could not connect to Pinecone. Check your network and credentials."
            ) from exc
        return client

    def init_store(self) -> PineconeVectorStore:
        """Open the Pinecone Serverless index used for similarity search."""
        self.client = self._init_client()
        self.dimension = self._embedding_dimension()
        self._ensure_index()
        self.index = self.client.Index(self.index_name)
        self.store = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings,
            namespace=self.namespace,
            text_key="text",
        )
        return self.store

    def _ensure_index(self) -> None:
        """Create a Serverless index when it is missing, and reject a dimension mismatch."""
        names = self.client.list_indexes().names()
        if self.index_name not in names:
            try:
                self.client.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
            except _PINECONE_ERRORS as exc:
                raise VectorStoreError(
                    f"Could not create Pinecone index '{self.index_name}'."
                ) from exc
            return
        try:
            described = self.client.describe_index(self.index_name)
        except _PINECONE_ERRORS as exc:
            raise VectorStoreError(
                f"Could not describe Pinecone index '{self.index_name}'."
            ) from exc
        existing = described.dimension
        if existing != self.dimension:
            raise VectorStoreError(
                f"Index '{self.index_name}' has dimension {existing}, "
                f"but '{self.embedding_model_name}' produces vectors of length {self.dimension}."
            )

    def count(self) -> int:
        try:
            stats = self.index.describe_index_stats()
        except _PINECONE_ERRORS as exc:
            raise VectorStoreError("Could not read Pinecone index stats.") from exc
        namespaces = getattr(stats, "namespaces", None) or {}
        namespace_stats = namespaces.get(self.namespace)
        if namespace_stats is None:
            return 0
        if isinstance(namespace_stats, dict):
            return int(namespace_stats.get("vector_count") or 0)
        return int(getattr(namespace_stats, "vector_count", 0) or 0)

    def upsert_documents(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Insert documents, or replace them when the id already exists.

        Pinecone stores the chunk text in metadata because the index has no
        separate document field.
        """
        try:
            for batch_ids, batch_documents, batch_metadatas in self._batches(
                ids, documents, metadatas
            ):
                vectors = self.embeddings.embed_documents(batch_documents)
                payload = []
                for doc_id, document, metadata, values in zip(
                    batch_ids, batch_documents, batch_metadatas, vectors
                ):
                    stored = dict(metadata)
                    stored["text"] = document
                    payload.append({"id": doc_id, "values": values, "metadata": stored})
                self.index.upsert(vectors=payload, namespace=self.namespace)
        except _PINECONE_ERRORS as exc:
            raise VectorStoreError("Could not write documents to the vector store.") from exc

    def semantic_search(
        self,
        query_text: str,
        n_results: int = settings.top_k,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the nearest documents for this namespace."""
        try:
            pairs = self.store.similarity_search_with_score(
                query_text,
                k=n_results,
                filter=where,
            )
        except _PINECONE_ERRORS as exc:
            raise VectorStoreError("Could not query the vector store.") from exc
        return [
            {
                "document": document.page_content,
                "metadata": document.metadata,
                "distance": float(distance),
            }
            for document, distance in pairs
        ]

    def delete_by_id(self, ids: list[str]) -> None:
        """Remove documents by id from this namespace."""
        if not ids:
            return
        try:
            self.index.delete(ids=ids, namespace=self.namespace)
        except _PINECONE_ERRORS as exc:
            raise VectorStoreError("Could not delete documents from the vector store.") from exc


# TODO move to factory.py
_manager: VectorStoreManager | None = None

def get_vector_store() -> VectorStoreManager:
    """Return the process-wide store so the embedding model is loaded once (singleton)."""
    global _manager
    if _manager is None:
        if settings.vector_store == "pinecone":
            _manager = PineconeStoreManager()
        elif settings.vector_store == "chroma":
            _manager = ChromaStoreManager()
        else:
            raise ValueError("VECTOR_STORE must be one of: pinecone, chroma")
    return _manager
