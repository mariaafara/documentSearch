from typing import List, Tuple
import nltk
import string
import numpy as np
from nltk.corpus import stopwords

nltk.download("stopwords")

import spacy

# Load the pre-trained language model
nlp = spacy.load('en_core_web_sm')


class Processor:

    def __init__(self, n=2):
        self.n = n

    @staticmethod
    def _lowercase(text) -> str:
        # convert the document to lowercase
        return text.lower()

    @staticmethod
    def _punctuation_filter(text) -> str:
        return text.translate(str.maketrans("", "", string.punctuation))

    @staticmethod
    def _new_lines_filter(text) -> str:
        return text.replace("\n", " ")

    @staticmethod
    def _tokenize(text) -> List[str]:
        return text.split()
        # return nltk.word_tokenize(document)

    @staticmethod
    def _stopwords_filter(tokens: List[str]) -> List[str]:
        stop_words = set(stopwords.words("english"))
        return [token for token in tokens if token.lower() not in stop_words]

    @staticmethod
    def _lemmatize(tokens: List[str]):
        pass

    @staticmethod
    def compute_embeddings(ngrams: List[Tuple[str]]):
        # Compute the n-gram embeddings using spacy lm
        ngram_embeddings = []
        for ngram in ngrams:
            ngram_embedding = nlp(" ".join(ngram)).vector
            ngram_embeddings.append(ngram_embedding)
        return ngram_embeddings

    def preprocess(self, text: str) -> List[str]:
        text = Processor._punctuation_filter(text)
        text = Processor._new_lines_filter(text)
        text = Processor._lowercase(text)
        # Tokenize the document into words
        tokens = Processor._tokenize(text)
        # Remove stopwords
        tokens = Processor._stopwords_filter(tokens)

        ngrams = set()
        # Generate n-grams (tri-gram)
        for i in range(1, self.n + 1):
            ngrams.update(nltk.ngrams(tokens, i))  # List of Tuples of (strs)
            # print("\n -->", i, ngrams)
        return list(ngrams)

    def postprocess(self, ngrams: List[Tuple[str]]):
        """Combine contiguous N-grams."""
        pass
