import pickle
from typing import List, Tuple
from collections import defaultdict

from search.index_store.index_store import IndexStore


class InMemoryIndexStore(IndexStore):
    """Class that stores InMemory indices.

    A dictionary will is created where all the words of the documents are mapped to the IDs of the documents
    they occur in.
    """

    def __init__(self):
        super().__init__()
        self.ngrams_indices_dict = defaultdict(set)
        # {ngram: [doc_id ngram mentions in]}

    def add_doc(self, doc_id, ngrams, **kwargs) -> None:
        """Add a single indexed document to the store."""
        for ngram in ngrams:
            self.ngrams_indices_dict[ngram].add(doc_id)

    def add_docs(self, indices: List[Tuple[Tuple, List[str]]], **kwargs) -> None:
        """Add a batch of indexed documents to the store."""
        # [(ngram, [doc ids where ngram found in,..] )...]
        for index in indices:
            self.ngrams_indices_dict[index[0]].update(index[1])

    def get_docs(self, ngrams: List[Tuple], **kwargs) -> List[Tuple[Tuple, List[str]]]:
        """Lookup self.ngrams_indices_dict to get list of documents that contain the input ngrams.

        return List[Tuple[str, List[str]]]. e.g. [(doc_id, [tokens matched in doc]),etc]
        """
        result_dict = defaultdict(list)
        for ngram in ngrams:
            docs_ids = self.ngrams_indices_dict[ngram]
            for doc_id in docs_ids:
                result_dict[doc_id].append(ngram)
        return list(result_dict.items())

    def save(self):
        with open("serialized_ngrams_indices_dict.pkl", "wb") as f:
            pickle.dump(self.ngrams_indices_dict, f)

    def load(self):
        with open("serialized_ngrams_indices_dict.pkl", "rb") as f:
            self.ngrams_indices_dict = pickle.load(f)
