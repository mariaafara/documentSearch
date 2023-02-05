from sklearn.metrics.pairwise import cosine_similarity

from search.index_store.embedding_in_memory import EmbeddingInMemoryIndexStore
from search.index_store.in_memory import InMemoryIndexStore
from search.processor.embedding_processor import EmbeddingProcessor
from search.processor.processor import Processor
from search.searcher.embedding_searcher import EmbeddingSearcher
from search.searcher.searcher import Searcher
from deepdiff import DeepDiff


def test_searcher():
    in_memory_index_store = InMemoryIndexStore()
    in_memory_index_store.add_doc(
        doc_id="id_1", ngrams=[("token1",), ("token2",), ("token3",)]
    )
    in_memory_index_store.add_doc(
        doc_id="id_2", ngrams=[("token2",), ("token4",), ("token5",)]
    )
    in_memory_index_store.add_doc(
        doc_id="id_3", ngrams=[("token2",), ("token5",), ("token6",)]
    )

    processor = Processor(n=1)
    searcher = Searcher(processor, in_memory_index_store)

    assert (
        DeepDiff(
            searcher.search(query_terms=["token1 token7", "token2"]),
            [
                ("id_1", [("token1",), ("token2",)]),
                ("id_2", [("token2",)]),
                ("id_3", [("token2",)]),
            ],
            ignore_order=True,
        )
        == {}
    )
    assert (
        DeepDiff(
            searcher.search(query_terms=["token1"]),
            [("id_1", [("token1",)])],
            ignore_order=True,
        )
        == {}
    )
    assert (
        DeepDiff(
            searcher.search(query_terms=["token8"]),
            [],
            ignore_order=True,
        )
        == {}
    )
    assert (
        DeepDiff(
            searcher.search(
                query_terms=["token5"],
            ),
            [
                ("id_2", [("token5",)]),
                ("id_3", [("token5",)]),
            ],
            ignore_order=True,
        )
        == {}
    )
