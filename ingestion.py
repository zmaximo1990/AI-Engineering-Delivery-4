"""Load assets from /data, convert them to Markdown, chunk them, and index them."""

import hashlib
import logging
from pathlib import Path

from langchain_core.documents import Document

from settings import DOC_ID, settings
from extraction import AssetExtractor
from processing import Chunkener
from factory import VectorStoreFactory
from vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)


class Ingestor:
    """Read the data directory, chunk each asset, and upsert the chunks."""

    def __init__(self, data_dir: Path | None = None, extractor: AssetExtractor | None = None, chunkener: Chunkener | None = None) -> None:
        self.data_dir = data_dir or settings.data_dir
        self.extractor = extractor or AssetExtractor()
        self.chunkener = chunkener or Chunkener()

    def list_files(self) -> list[Path]:
        """Return supported assets in the data directory, sorted by name."""
        directory = self.data_dir
        if not directory.is_dir():
            raise FileNotFoundError(f"Data directory does not exist: {directory}")
        files: list[Path] = []
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() in AssetExtractor.SUPPORTED_SUFFIXES:
                files.append(path)
            elif path.name != ".DS_Store":
                logger.warning("Skipping unsupported asset %s", path.name)
        return files

    def iter_chunks(self):
        """Yield ``(source, chunk_index, text, metadata)`` for every chunk."""

        # List all the files in the data directory that are supported
        files = self.list_files()
        if not files:
            raise FileNotFoundError(
                f"No supported assets found in {self.data_dir}"
            )

        for path in files:
            # Extract the segments from the file
            segments = self.extractor.extract(path)
            chunk_index = 0
            for segment in segments:
                # Chunk the segment
                for chunk in self.chunkener.process_document(segment.text):
                    metadata = VectorStoreManager.create_metadata(
                        source=path.name,
                        chunk_index=chunk_index,
                        text=chunk,
                        doc_id=self._doc_id(path.name),
                        asset_format=path.suffix,
                        page=segment.page,
                    )
                    yield path.name, chunk_index, chunk, metadata
                    chunk_index += 1
            logger.info("%s -> %d chunks", path.name, chunk_index)

    def load_documents(self) -> list[Document]:
        """Return the chunked corpus without embedding or writing it."""
        return [
            Document(page_content=text, metadata=metadata)
            for _source, _index, text, metadata in self.iter_chunks()
        ]

    def ingest_if_needed(self) -> int:
        """Chunk and persist the corpus when the collection is still empty.

        A non-empty collection is treated as an existing index, so the files
        are not embedded again. Returns the number of chunks written on this call.
        """
        store: VectorStoreManager = VectorStoreFactory.get_manager()
        existing = store.count()
        if existing > 0:
            logger.info(
                "Vector store already contains %d chunks; skipping ingestion. "
                "Delete the index contents to reindex.",
                existing,
            )
            return 0

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        for source, index, text, metadata in self.iter_chunks():
            ids.append(self._chunk_id(source, index, text))
            documents.append(text)
            metadatas.append(metadata)

        if not ids:
            raise ValueError("The source files did not produce any chunks")

        store.upsert_documents(ids, documents, metadatas)
        logger.info("Indexed %d chunks into the vector store", len(ids))
        return len(ids)


    def _doc_id(self, filename: str) -> str:
        """
        Return the leading NE-100 style id, or the file stem when there is none.
        """

        match = DOC_ID.match(filename)
        if match:
            return match.group(1).upper()
        return Path(filename).stem


    def _chunk_id(self, source: str, index: int, text: str) -> str:
        """
        Hash the source, index, and text so a repeated ingest replaces the same chunk.
        """

        payload = f"{source}\n{index}\n{text}".encode()
        return hashlib.sha256(payload).hexdigest()
