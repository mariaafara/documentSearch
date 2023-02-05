from search.document import Document
from search.index_store.in_memory import InMemoryIndexStore
from search.indexer.indexer import Indexer
from search.processor.processor import Processor


def test_indexer():
    documents = [
        Document("", "id_1", "", text="token1, token2 token3."),
        Document("", "id_2", "", text="Token2 token4: token5!"),
        Document("", "id_3", "", text="token2 @token5 Token6?"),
    ]

    expected_indices_dict = {
        ("token1",): {"id_1"},
        ("token2",): {"id_1", "id_2", "id_3"},
        ("token3",): {"id_1"},
        ("token4",): {"id_2"},
        ("token5",): {"id_2", "id_3"},
        ("token6",): {"id_3"},
    }
    processor = Processor(n=1)
    index_store = InMemoryIndexStore()
    indexer = Indexer(processor, index_store)
    indexer.index_docs(documents)

    assert expected_indices_dict.keys() == index_store.ngrams_indices_dict.keys()

    for token, docs in index_store.ngrams_indices_dict.items():
        assert expected_indices_dict[token] == docs
