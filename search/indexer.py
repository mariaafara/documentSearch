from abc import ABC

from typing import List

from search.document import Document
from search.index_store.index_store import IndexStore
from search.processor import Processor


class Indexer:
    """Class that indexes documents."""

    def __init__(self, processor: Processor, index_store: IndexStore):
        self.processor = processor
        self.index_store = index_store

    def index_docs(self, documents: List[Document]) -> None:
        for document in documents:
            tokenized_doc = Processor.process(document.text)
            self.index_store.add_doc(document.id, tokenized_doc)
