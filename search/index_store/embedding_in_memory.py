import pickle
from typing import List, Tuple
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

from search.index_store.in_memory import InMemoryIndexStore


class EmbeddingInMemoryIndexStore(InMemoryIndexStore):
    """Class that stores InMemory indices.

    A dictionary will is created where all the words of the documents are mapped to the IDs of the documents
    they occur in.
    """

    def __init__(self):
        super().__init__()
        self.document_indices = {}  # {document_id: document_embedding}

    def add_doc(self, doc_id, ngrams, document_embedding=None, **kwargs) -> None:
        super().add_doc(doc_id, ngrams)
        self.document_indices[doc_id] = document_embedding

    @staticmethod
    def compute_similarity(doc_embeddings, query_embeddings):
        # Compute the similarity between document and query embeddings
        similarities = cosine_similarity(doc_embeddings, query_embeddings)
        return similarities

    def _get_similarities(self, doc_ids, query_embedding):
        similarities_dict = {}
        for doc_id in doc_ids:
            similarity_score = self.compute_similarity(
                doc_embeddings=self.document_indices[doc_id].reshape(1, -1),
                # reshape each to 2-dimensional numpy array
                query_embeddings=query_embedding.reshape(1, -1),
            )
            # store the similarity score
            # rank the documents based on their scores while excluding those which don't fulfill the similarity
            # threshold defined. The score is preserved for future need if needed.
            similarities_dict[doc_id] = similarity_score[0][0]
        return similarities_dict

    def _filter(self, matched_docs, similarities_dict, similarity_threshold):
        filtered_doc_ids = list(
            filter(
                lambda doc: doc[1] >= similarity_threshold, similarities_dict.items()
            )
        )
        filtered_doc_ids_with_score = {
            doc_id: score for doc_id, score in filtered_doc_ids
        }
        filtered_docs = list(
            filter(lambda doc: doc[0] in filtered_doc_ids_with_score, matched_docs)
        )
        return filtered_docs, filtered_doc_ids_with_score

    def _sort(self, filtered_docs, filtered_doc_ids_with_score):
        return sorted(
            filtered_docs,
            key=lambda doc: filtered_doc_ids_with_score[doc[0]],
            reverse=True,
        )

    def _filter_and_sort(self, matched_docs, query_embedding, similarity_threshold):
        # compute a similarity score between the averaged query embedding and each matched document embedding
        # to rank and to filter out those of which are the least similar based on a similarity threshold
        doc_ids = [matched_doc[0] for matched_doc in matched_docs]
        similarities_dict = self._get_similarities(doc_ids, query_embedding)
        filtered_docs, filtered_doc_ids_with_score = self._filter(
            matched_docs, similarities_dict, similarity_threshold
        )

        print(f"On ngram match: {len(matched_docs)} docs are selected.")
        print(f"Using Semantic similarity {len(filtered_docs)} docs are then selected.")

        return self._sort(filtered_docs, filtered_doc_ids_with_score)

    def get_docs(
        self,
        ngrams: List[Tuple],
        query_embedding=None,
        similarity_threshold=None,
        **kwargs,
    ):  # -> List[Tuple[str, List[str]]]:
        """return List[Tuple[str, List[str]]]: [(doc_id, [ngrams matched in doc]),etc]"""

        matched_docs = super().get_docs(
            ngrams=ngrams
        )  # [(doc_id, [ngrams matched in doc]),...]
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
