from typing import List
import nltk
import string

from nltk.corpus import stopwords

nltk.download("stopwords")


class Processor:
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
    def process(text: str) -> List[str]:
        text = Processor._punctuation_filter(text)
        text = Processor._new_lines_filter(text)
        text = Processor._lowercase(text)
        tokens = Processor._tokenize(text)
        tokens = Processor._stopwords_filter(tokens)
        return tokens
