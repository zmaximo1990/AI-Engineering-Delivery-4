"""Convert source assets into Markdown pages."""

import re
from pathlib import Path

from markitdown import MarkItDown
from pdfminer.high_level import extract_text
from pdfminer.pdfpage import PDFPage

from schemas import DocumentPage


class AssetExtractor:
    """Convert an asset into Markdown pages.

    Document formats are implemented today. A video or another asset type can
    be added as one more strategy on this class. It must return the same
    pages, so Ingestor and Chunkener stay unchanged.
    """

    SUPPORTED_SUFFIXES = {".pdf", ".docx", ".md", ".txt"}

    def __init__(self) -> None:
        self._converter = MarkItDown()

    def extract(self, path: Path) -> list[DocumentPage]:
        """Return Markdown pages for ``path``. The page is set for PDFs."""
        suffix = path.suffix.lower()
        strategies = {
            ".pdf": self._extract_pdf,
            ".docx": self._extract_docx,
            ".md": self._extract_plain,
            ".txt": self._extract_plain,
        }
        strategy = strategies.get(suffix)
        if strategy is None:
            raise ValueError(
                f"Unsupported asset type '{suffix}' for {path.name}. "
                "Add a strategy on AssetExtractor for this type."
            )
        return strategy(path)

    def _convert(self, path: Path) -> str:
        """Convert the asset to Markdown."""
        result = self._converter.convert(str(path))
        text = getattr(result, "text_content", None)
        if text is None:
            text = getattr(result, "markdown", "")
        return (text or "").strip()

    def _extract_plain(self, path: Path) -> list[DocumentPage]:
        """Extract text from a plain text file.

        A plain text file returns a single page because it has no pages.

        Arguments:
            path: Path to the plain text file.

        Returns:
            list[DocumentPage]: One page with the file text, or an empty list.
        """
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        return [DocumentPage(text=text, page=None)]

    def _extract_docx(self, path: Path) -> list[DocumentPage]:
        """Extract text from a DOCX file.

        Arguments:
            path: Path to the DOCX file.

        Returns:
            list[DocumentPage]: One page with the document text.
        """
        text = self._convert(path)
        if not text:
            raise ValueError(f"Could not extract text from {path.name}")
        return [DocumentPage(text=text, page=None)]

    def _extract_pdf(self, path: Path) -> list[DocumentPage]:
        """Extract text from a PDF file.

        Prefer MarkItDown page breaks. Otherwise split with its PDF engine.

        Arguments:
            path: Path to the PDF file.

        Returns:
            list[DocumentPage]: One page per PDF page that contains text.
        """
        markdown = self._convert(path)
        pages = [part.strip() for part in re.split(r"\f", markdown)]
        segments = [
            DocumentPage(text=page, page=number)
            for number, page in enumerate(pages, start=1)
            if page
        ]
        if len(segments) > 1:
            return segments
        paged = self._pdfminer_pages(path)
        if len(paged) > 1:
            return paged
        if segments:
            return segments
        return paged

    def _pdfminer_pages(self, path: Path) -> list[DocumentPage]:
        """Read one PDF page at a time with pdfminer, the engine MarkItDown uses."""
        with path.open("rb") as handle:
            page_count = sum(1 for _ in PDFPage.get_pages(handle))
        segments: list[DocumentPage] = []
        for index in range(page_count):
            text = extract_text(str(path), page_numbers=[index]) or ""
            text = text.replace("\f", "").strip()
            if text:
                segments.append(DocumentPage(text=text, page=index + 1))
        return segments
