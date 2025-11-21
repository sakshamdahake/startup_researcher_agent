from typing import List, Optional

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    CharacterTextSplitter,
)
from langchain_core.documents import Document


def get_recursive_splitter(chunk_size: int = 500, chunk_overlap: int = 100, separators: Optional[List[str]] = None):
    """
    General-purpose splitter: prefers paragraph > sentence > word boundaries.
    """
    separators = separators or ["\n\n", "\n", ". ", " ", ""]
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )


def get_token_splitter(chunk_size: int = 256, chunk_overlap: int = 40):
    """
    Token-aware splitter (tiktoken-backed).
    """
    return TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def get_sentence_splitter(chunk_size: int = 800, chunk_overlap: int = 100):
    """
    Sentence-aware splitter using '.' and newline-based segmentation.
    """
    return CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[". ", "\n", " "],
    )


def split_documents_with_metadata(docs: List[Document], splitter=None) -> List[Document]:
    """
    Split a list of Documents and preserve metadata.
    """
    splitter = splitter or get_recursive_splitter()
    return splitter.split_documents(docs)