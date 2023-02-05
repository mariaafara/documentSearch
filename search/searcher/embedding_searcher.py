from collections import defaultdict
from typing import List, Tuple

import numpy as np

from search.index_store.embedding_in_memory import EmbeddingInMemoryIndexStore
from search.index_store.index_store import IndexStore
from sklearn.metrics.pairwise import cosine_similarity

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

    @staticmethod
    def compute_similarity(doc_embeddings, query_embeddings):
        # Compute the similarity between document and query embeddings
        similarities = cosine_similarity(doc_embeddings, query_embeddings)
        return similarities

    def search(self, query_terms: List[str]) -> List[Tuple[str, List[str]]]:
        """Filter document based on input query terms.

        It uses query embedding and document embedding to retrieve the semantically similar docs with their mentions.
        :param query_terms: List of keywords/terms/phrases.
        :return: List of filtered document ids with the matched keywords in the document.
        """
        result_dict = defaultdict(set)
        queried_terms = set()
        query_embeddings = []
        for term in query_terms:
            preprocessed_term, embedded_term = self.processor.preprocess(term)
            search_query = (
                set(preprocessed_term) - queried_terms
            )  # query new tokens or n-grams

            queried_terms.update(search_query)
            if search_query:
                query_embeddings.append(embedded_term)

                term_result = self.index_store.get_docs(ngrams=list(search_query))
                for doc_id, matched_tokens in term_result:
                    result_dict[doc_id].update(matched_tokens)

        print("-" * 10)
        print(f"queried_terms= {queried_terms}")

        selected_doc_ids = list(
            result_dict.keys()
        )  # get the id of the documents where terms from the query are matched in
        query_average_embedding = np.mean(
            query_embeddings, axis=0
        )  # average the embeddings of the query ngrams

        # compute a similarity score between the averaged query embedding and each matched document embedding
        # to rank and to filter out those of which are the least similar based on a similarity threshold
        similarities_dict = {}
        for doc_id in selected_doc_ids:
            similarity_score = self.compute_similarity(
                doc_embeddings=self.index_store.document_indices[doc_id].reshape(1, -1),
                # reshape each to 2-dimensional numpy array
                query_embeddings=query_average_embedding.reshape(1, -1),
            )

            similarities_dict[doc_id] = similarity_score[0][
                0
            ]  # store the similarity score
            # rank the documents based on their scores while excluding those which don't fulfill the similarity
            # threshold defined. The score is preserved for future need if needed.
        scores = [
            (doc_id, score)
            for doc_id, score in sorted(
                similarities_dict.items(), key=lambda x: x[1], reverse=True
            )
            if score >= self.similarity_threshold
        ]
        similar_doc_ids = [doc_id for doc_id, _ in scores]

        print(f"On ngram match: {len(selected_doc_ids)} docs are selected.")
        print(
            f"Using Semantic similarity {len(similar_doc_ids)} docs are then selected."
        )

        # only retrieve documents which are semantically similar and their ngrams that were matched in the document
        filtered_docs_with_matches = []
        for doc_id, matched_tokens in result_dict.items():
            if doc_id in similar_doc_ids:
                filtered_docs_with_matches.append((doc_id, list(matched_tokens)))

        return filtered_docs_with_matches
