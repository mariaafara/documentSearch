from typing import List, Tuple

from search.processor.processor import Processor
import spacy

# Load the pre-trained language model
nlp = spacy.load('en_core_web_sm')


class EmbeddingProcessor(Processor):
    # @staticmethod
    # def compute_ngrams_embeddings(ngrams: List[Tuple[str]]):
    #     # Compute the n-gram embeddings using spacy lm
    #     ngram_embeddings = []
    #     for ngram in ngrams:
    #         ngram_embedding = EmbeddingProcessor.compute_embedding(" ".join(ngram))
    #         ngram_embeddings.append(ngram_embedding)
    #     return ngram_embeddings

    @staticmethod
    def compute_embedding(text: str):
        return nlp(text).vector

    def compute_document_embedding(self, text: str):
        """Embed a document."""
        return EmbeddingProcessor.compute_embedding(text)
