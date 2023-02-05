from typing import List, Tuple, Union

import numpy as np
import spacy

from search.processor.processor import Processor

# Load the pre-trained language model
nlp = spacy.load("en_core_web_sm")


class EmbeddingProcessor(Processor):
    @staticmethod
    def _compute_embedding(text: str) -> np.ndarray:
        """Compute text embedding using SpaCy model."""
        return nlp(text).vector

    def preprocess(self, text: str) -> (List[Tuple[str, ...]], Union[None, np.ndarray]):
        """Preprocess input text."""
        ngrams = super().preprocess(text)
        embedded_doc = EmbeddingProcessor._compute_embedding(text)
        return ngrams, embedded_doc
