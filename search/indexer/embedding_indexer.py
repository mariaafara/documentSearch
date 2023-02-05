from typing import List

from search.document import Document
from search.index_store.embedding_in_memory import EmbeddingInMemoryIndexStore
from search.index_store.index_store import IndexStore
from search.indexer.indexer import Indexer
from search.processor.embedding_processor import EmbeddingProcessor
from search.processor.processor import Processor


class EmbeddingIndexer(Indexer):
    """Class that indexes documents."""

    def __init__(self, processor: EmbeddingProcessor, index_store: EmbeddingInMemoryIndexStore):
        super().__init__(processor, index_store)
        self.processor = processor
        self.index_store = index_store

    def index_docs(self, documents: List[Document]) -> None:
        for document in documents:
            preprocessed_doc, embedded_doc = self.processor.preprocess(document.text)
            self.index_store.add_doc(doc_id=document.id, ngrams=preprocessed_doc, document_embedding=embedded_doc)
            # print(self.index_store.ngrams_indices_dict)
