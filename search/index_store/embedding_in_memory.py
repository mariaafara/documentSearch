import pickle
from typing import List, Tuple, Dict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from search.index_store.in_memory import InMemoryIndexStore


class EmbeddingInMemoryIndexStore(InMemoryIndexStore):
    """Class that stores InMemory indices.

    A dictionary will is created where all the words of the documents are mapped to the IDs of the documents
    they occur in.
    """

    def __init__(self):
        super().__init__()
        self.document_indices = {}

    def add_doc(
        self,
        doc_id: str,
        ngrams: List[Tuple[str, ...]],
        document_embedding: np.ndarray = None,
        **kwargs,
    ) -> None:
        """Add a single indexed document to the store."""
        super().add_doc(doc_id, ngrams)
        self.document_indices[doc_id] = document_embedding

    @staticmethod
    def _compute_similarity(
        doc_embedding: np.ndarray, query_embedding: np.ndarray
    ) -> np.float64:
        """Compute the similarity between document and query embeddings."""
        similarity_score = cosine_similarity(doc_embedding, query_embedding)
        return similarity_score

    def _get_similarities(
        self, doc_ids: List[str], query_embedding: np.ndarray
    ) -> Dict[str, int]:
        """Compute the similarity between the query embedding and a list of documents."""
        similarities_dict = {}
        for doc_id in doc_ids:
            similarity_score = self._compute_similarity(
                doc_embedding=self.document_indices[doc_id].reshape(1, -1),
                # reshape each to 2-dimensional numpy array
                query_embedding=query_embedding.reshape(1, -1),
            )
            # store the similarity score
            similarities_dict[doc_id] = similarity_score[0][0]
        return similarities_dict

    def _filter(
        self,
        matched_docs: List[Tuple[str, List[Tuple[str, ...]]]],
        similarities_dict: Dict[str, int],
        similarity_threshold: float,
    ) -> List[Tuple[str, List[Tuple[str, ...]]]]:
        """Filter out documents which don't fulfill the similarity threshold."""
        filtered_docs = list(
            filter(
                lambda doc: similarities_dict[doc[0]] >= similarity_threshold,
                matched_docs,
            )
        )
        return filtered_docs

    def _sort(
        self,
        filtered_docs: List[Tuple[str, List[Tuple[str, ...]]]],
        similarities_dict: Dict[str, int],
    ) -> List[Tuple[str, List[Tuple[str, ...]]]]:
        """Sort the documents based on their similarity scores."""
        return sorted(
            filtered_docs,
            key=lambda doc: similarities_dict[doc[0]],
            reverse=True,
        )

    def _filter_and_sort(
        self,
        matched_docs: List[Tuple[str, List[Tuple[str, ...]]]],
        query_embedding: np.ndarray,
        similarity_threshold: float,
    ) -> List[Tuple[str, List[Tuple[str, ...]]]]:
        """Filter and sort the most similar documents to the query embedding."""
        # compute a similarity score between the averaged query embedding and each matched document embedding.
        # to rank and to filter out those of which are the least similar based on a similarity threshold
        doc_ids = [matched_doc[0] for matched_doc in matched_docs]
        similarities_dict = self._get_similarities(doc_ids, query_embedding)
        filtered_docs = self._filter(
            matched_docs, similarities_dict, similarity_threshold
        )

        print(f"On ngram match: {len(matched_docs)} docs are selected.")
        print(f"Using Semantic similarity {len(filtered_docs)} docs are then selected.")

        return self._sort(filtered_docs, similarities_dict)

    def get_docs(
        self,
        ngrams: List[Tuple[str, ...]],
        query_embedding: np.ndarray = None,
        similarity_threshold: float = None,
        **kwargs,
    ) -> List[Tuple[str, List[Tuple[str, ...]]]]:
        """Lookup self.ngrams_indices_dict to get list of documents that contain the input ngrams."""
        matched_docs = super().get_docs(ngrams=ngrams)
        filtered_docs = self._filter_and_sort(
            matched_docs, query_embedding, similarity_threshold
        )
        return filtered_docs

    def save(self):
        super().save()
        with open("serialized_document_indices.pkl", "wb") as f:
            pickle.dump(self.document_indices, f)

    def load(self):
        super().load()
        with open("serialized_document_indices.pkl", "rb") as f:
            self.document_indices = pickle.load(f)
