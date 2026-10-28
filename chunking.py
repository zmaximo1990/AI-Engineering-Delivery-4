"""Clean source text and split it into token-sized chunks."""

import re

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import settings

# Repeated page headers from an extracted PDF.
_PAGE_HEADER = re.compile(r"(?m)^.*Pág\.\s*\d+.*\n?")
# More than one blank line between paragraphs.
_EXTRA_BREAKS = re.compile(r"\n{3,}")
# Repeated spaces and tabs. Line breaks stay so the splitter can cut on paragraphs.
_MULTI_SPACE = re.compile(r"[^\S\n]+")


class Chunkener:
    """Clean Markdown text and split it into token-sized chunks."""

    def __init__(
        self,
        model_encoding: str = settings.model_encoding,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ):
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding(model_encoding)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self.calculate_tokens,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def clean_text(self, text: str) -> str:
        """Prepare Markdown for chunking and embedding.

        Drop PDF page-header lines that contain ``Pág.`` and a page number,
        remove the ``■`` extraction artifact, collapse three or more line
        breaks to a single blank line, and collapse repeated spaces and tabs
        to one space. Line breaks stay so the splitter can still cut on
        paragraphs. Leading and trailing whitespace is stripped.
        """
        text = _PAGE_HEADER.sub("", text)
        text = text.replace("■", "")
        text = _EXTRA_BREAKS.sub("\n\n", text)
        text = _MULTI_SPACE.sub(" ", text)
        return text.strip()

    def calculate_tokens(self, text: str) -> int:
        """Measure the text in tokens."""
        return len(self.tokenizer.encode(text))

    def process_document(self, raw_text: str) -> list[str]:
        """Clean Markdown text, then split it into token-sized chunks."""
        cleaned = self.clean_text(raw_text)
        if not cleaned:
            return []
        return self.splitter.split_text(cleaned)
