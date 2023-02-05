from collections import defaultdict
from typing import List, Tuple

import numpy as np

from search.index_store.embedding_in_memory import EmbeddingInMemoryIndexStore
from search.index_store.index_store import IndexStore
from search.processor.embedding_processor import EmbeddingProcessor
from search.searcher.searcher import Searcher


class EmbeddingSearcher(Searcher):
    """Class to search for documents based on a query."""

    def __init__(
        self,
        processor: EmbeddingProcessor,
        index_store: EmbeddingInMemoryIndexStore,
        similarity_threshold=0.25,
    ):
        super().__init__(processor, index_store)
        self.processor = processor
        self.index_store = index_store
        self.similarity_threshold = similarity_threshold

    def _prepare_query(self, query_terms):
        """Prepare the search query."""
        queried_terms = set()
        query_embeddings = []
        for term in query_terms:
            preprocessed_term, embedded_term = self.processor.preprocess(term)

            queried_terms.update(preprocessed_term)
            query_embeddings.append(embedded_term)

        return list(queried_terms), query_embeddings

    def search(self, query_terms: List[str]) -> List[Tuple[str, List[str]]]:
        """Filter document based on input query terms.

        It uses query embedding and document embedding to retrieve the semantically similar docs with their mentions.
        :param query_terms: List of keywords/terms/phrases.
        :return: List of filtered document ids with the matched keywords in the document.
        """
        prepared_query, query_embeddings = self._prepare_query(query_terms)
        # compute average the embeddings of the query ngrams
        query_average_embedding = np.mean(query_embeddings, axis=0)
        term_result = self.index_store.get_docs(
            ngrams=prepared_query,
            query_embedding=query_average_embedding,
            similarity_threshold=self.similarity_threshold,
        )
        result_dict = self._aggregate_docs(term_result)
        return list(result_dict.items())
