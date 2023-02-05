from typing import List, Tuple, Union

from search.processor.processor import Processor
import spacy

# Load the pre-trained language model
nlp = spacy.load("en_core_web_sm")


class EmbeddingProcessor(Processor):
    @staticmethod
    def compute_embedding(text: str):
        """Compute text embedding using SpaCy model."""
        return nlp(text).vector

    def compute_document_embedding(self, text: str):
        """Embed a document."""
        return EmbeddingProcessor.compute_embedding(text)

    def preprocess(self, text: str) -> (List[Tuple[str]], Union[None, List[float]]):
        """Preprocess input text."""
        ngrams = super().preprocess(text)
        embedded_doc = self.compute_document_embedding(text)
        return ngrams, embedded_doc
