from typing import List, Tuple
from collections import defaultdict

from search.index_store.index_store import IndexStore


class InMemoryIndexStore(IndexStore):
    """Class that stores InMemory indices.

    A dictionary will is created where all the words of the documents are mapped to the IDs of the documents
    they occur in.
    """

    def __init__(self):
        super(IndexStore).__init__()
        self.indices_dict = defaultdict(set)

    def add_doc(self, doc_id, tokens) -> None:
        for token in tokens:
            self.indices_dict[token].add(doc_id)

    def get_docs(self, tokens: List[str]) -> List[Tuple[str, List[str]]]:
        """return List[Tuple[str, List[str]]]: [(doc_id, [tokens matched in doc]),etc]"""
        result_dict = defaultdict(list)
        for token in tokens:
            docs_ids = self.indices_dict[token]
            for doc_id in docs_ids:
                result_dict[doc_id].append(token)
        return list(result_dict.items())
