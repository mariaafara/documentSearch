from abc import ABC

from typing import List, Tuple


class IndexStore(ABC):
    """Class that stores an inverted index."""

    def add_doc(self, doc_id, tokens) -> None:
        """Index a document."""
        raise NotImplementedError()

    def get_docs(self, tokens: List[str]) -> List[Tuple[str, List[str]]]:
        """Get all docs based on token indices."""
        raise NotImplementedError()
