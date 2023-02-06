from typing import List, Tuple

from search.index_store.index_store import IndexStore


class RedisIndexStore(IndexStore):
    """Class that stores an inverted index in the Redis database
    This class is not implemented, but it just to explain that we can extend the IndexStore to support any data store
    """

    def add_doc(self, doc_id, ngrams, **kwargs) -> None:
        """Index a document."""
        raise NotImplementedError()

    def get_docs(self, ngrams: List[str], **kwargs) -> List[Tuple[str, List[str]]]:
        """Get all docs based on token indices."""
        raise NotImplementedError()

    def add_docs(self, indices: List[Tuple[str, List[str]]], **kwargs) -> None:
        """Index documents."""
        raise NotImplementedError()
