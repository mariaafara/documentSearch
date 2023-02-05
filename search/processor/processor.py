from typing import List, Tuple, Union
import nltk
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()


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
    def _lemmatize(tokens: List[str]) -> List[str]:
        return [lemmatizer.lemmatize(token) for token in tokens]

    @staticmethod
    def _get_ngrams(tokens: List[str], n):
        """Generate n-grams."""
        ngrams = set()
        for i in range(1, n + 1):
            ngrams.update(nltk.ngrams(tokens, i))
        # if len(tokens) < n:
        #     ngrams = set((nltk.ngrams(tokens, len(tokens))))
        # else:
        #     ngrams = set((nltk.ngrams(tokens, n)))
        return ngrams  # List of Tuples of (strs)

    def compute_document_embedding(self, text: str):
        """Embed a document."""
        return None

    def preprocess(self, text: str) -> (List[Tuple[str]], Union[None, List[float]]):
        # Remove punctuations
        text = Processor._punctuation_filter(text)
        # Remove new lines
        text = Processor._new_lines_filter(text)
        # Lowercase
        text = Processor._lowercase(text)
        # Tokenize the document into words
        tokens = Processor._tokenize(text)
        # Remove stopwords
        tokens = Processor._stopwords_filter(tokens)
        # Lemmatize
        tokens = Processor._lemmatize(tokens)

        ngrams = Processor._get_ngrams(tokens, self.n)
        # embedded_ngrams = Processor.compute_ngrams_embeddings(list(ngrams))
        embedded_doc = self.compute_document_embedding(text)
        return ngrams, embedded_doc

    def postprocess(self, ngrams: List[Tuple[str]]):
        """Combine contiguous N-grams."""
        pass
