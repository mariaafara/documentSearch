from abc import ABC
from typing import List, Tuple


class IndexStore(ABC):
    """Class that stores an inverted index."""

    def add_doc(self, doc_id, ngrams, **kwargs) -> None:
        """Index a document."""
        raise NotImplementedError()

    def get_docs(self, ngrams: List[str], **kwargs) -> List[Tuple[str, List[str]]]:
        """Get all docs based on token indices."""
        raise NotImplementedError()

    def add_docs(self, indices: List[Tuple[str, List[str]]], **kwargs) -> None:
        """Index documents."""
        raise NotImplementedError()
