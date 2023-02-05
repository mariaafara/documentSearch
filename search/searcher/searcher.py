from collections import defaultdict
from typing import List, Tuple

from search.index_store.in_memory import InMemoryIndexStore
from search.index_store.index_store import IndexStore
from sklearn.metrics.pairwise import cosine_similarity

from search.processor import Processor


class Searcher:
    """Class to search for documents based on a query."""

    def __init__(self, processor: Processor, index_store: InMemoryIndexStore):
        self.processor = processor
        self.index_store = index_store

    @staticmethod
    def compute_similarity(doc_embeddings, query_embeddings):
        """Compute the similarity between document and query embeddings."""
        similarities = cosine_similarity(doc_embeddings, query_embeddings)
        return similarities

    def _prepare_query(self, query_terms):
        """"""
        queried_terms = set()
        for term in query_terms:
            processed_term, _ = self.processor.preprocess(term)
            queried_terms.update(processed_term)
        return list(queried_terms)

    def _aggregate_docs(self, raw_result):
        result_dict = defaultdict(set)
        for doc_id, matched_tokens in raw_result:
            result_dict[doc_id].update(matched_tokens)
        return {k: list(result_dict[k]) for k in result_dict}

    def search(self, query_terms: List[str]) -> List[Tuple[str, List[str]]]:
        """Filter document based on input query terms.

        :param query_terms: List of keywords/terms/phrases.
        :return: List of filtered document ids with the matched keywords in the document.
        """

        prepared_query = self._prepare_query(query_terms)
        term_result = self.index_store.get_docs(ngrams=prepared_query)
        result_dict = self._aggregate_docs(term_result)

        print("-" * 10)
        print(f"queried_terms= {prepared_query}")

        return list(result_dict.items())
