"""Convert source assets into Markdown pages."""

import json
import re
from pathlib import Path

from markitdown import MarkItDown
from pdfminer.high_level import extract_text
from pdfminer.pdfpage import PDFPage

from schemas import DocumentPage

# Heading for one object inside a JSON list. The first of these keys that
# holds a non-empty string becomes the section title of the target markdown conversion, in this order:
# term, then title, name, heading, and id. That key is not repeated in the body.
_JSON_TITLE_KEYS = ("term", "title", "name", "heading", "id")


class AssetExtractor:
    """Convert an asset into Markdown pages.

    Document formats are implemented today. A video or another asset type can
    be added as one more strategy on this class. It must return the same
    pages, so Ingestor and Chunkener stay unchanged.
    """

    SUPPORTED_SUFFIXES = {".pdf", ".docx", ".md", ".txt", ".json"}

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
            ".json": self._extract_json,
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

    def _extract_json(self, path: Path) -> list[DocumentPage]:
        """Extract text from a JSON file.

        The value is rendered as Markdown. A JSON file has no pages.

        Arguments:
            path: Path to the JSON file.

        Returns:
            list[DocumentPage]: One page with the rendered text, or an empty list.
        """
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse JSON from {path.name}") from exc
        text = self._json_to_markdown(payload).strip()
        if not text:
            return []
        return [DocumentPage(text=text, page=None)]

    def _json_to_markdown(self, value: object, level: int = 2) -> str:
        """Render a JSON value as Markdown. Blank values are omitted."""
        if self._json_blank(value):
            return ""
        if isinstance(value, dict):
            return self._json_object_markdown(value, level)
        if isinstance(value, list):
            return self._json_list_markdown(value, level)
        return self._json_scalar(value)

    def _json_object_markdown(
        self,
        value: dict,
        level: int,
        skip: str | None = None,
    ) -> str:
        """Render object keys as headings, or as a bold label when the value is scalar."""
        parts: list[str] = []
        depth = min(level, 6)
        for key, item in value.items():
            if key == skip or self._json_blank(item):
                continue
            rendered = self._json_to_markdown(item, level + 1)
            if not rendered:
                continue
            label = str(key).replace("_", " ")
            if isinstance(item, (dict, list)):
                parts.append(f"{'#' * depth} {label}\n\n{rendered}")
            else:
                parts.append(f"**{label}:** {rendered}")
        return "\n\n".join(parts)

    def _json_list_markdown(self, value: list, level: int) -> str:
        """Render a list of objects as sections, or a list of scalars as bullets."""
        items = [item for item in value if not self._json_blank(item)]
        if not items:
            return ""
        if all(isinstance(item, dict) for item in items):
            return self._json_sections(items, level)
        if all(not isinstance(item, (dict, list)) for item in items):
            return "\n".join(f"- {self._json_scalar(item)}" for item in items)
        blocks = [self._json_to_markdown(item, level) for item in items]
        return "\n\n".join(block for block in blocks if block)

    def _json_sections(self, items: list[dict], level: int) -> str:
        """Render each object as a heading plus its remaining fields."""
        sections: list[str] = []
        depth = min(level, 6)
        for item in items:
            title, skip = self._json_item_title(item)
            body = self._json_object_markdown(item, level + 1, skip=skip)
            if title and body:
                sections.append(f"{'#' * depth} {title}\n\n{body}")
            elif title:
                sections.append(f"{'#' * depth} {title}")
            elif body:
                sections.append(body)
        return "\n\n".join(sections)

    def _json_item_title(self, item: dict) -> tuple[str | None, str | None]:
        """Return a short label for a list item, and the key that supplied it."""
        for key in _JSON_TITLE_KEYS:
            raw = item.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip(), key
        return None, None

    def _json_scalar(self, value: object) -> str:
        """Return a scalar as a single line of text."""
        if isinstance(value, str):
            return value.strip()
        return str(value)

    def _json_blank(self, value: object) -> bool:
        """Return whether a JSON value has nothing to index."""
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, (list, dict)):
            return len(value) == 0
        return False

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
