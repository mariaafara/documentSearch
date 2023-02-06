from search.index_store.embedding_in_memory import EmbeddingInMemoryIndexStore
from search.index_store.in_memory import InMemoryIndexStore


def test_in_memory_index_store():
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

    # assert sorted(in_memory_index_store.get_docs(ngrams=[("token2",)])) == sorted(
    #     [("id_1", ["token2"]), ("id_2", ["token2"]), ("id_3", ["token2"])]
    # )
    assert sorted(in_memory_index_store.get_docs(ngrams=[("token1",)])) == sorted(
        [("id_1", [("token1",)])]
    )
    # assert sorted(in_memory_index_store.get_docs(tokens=["token5"])) == sorted(
    #     [("id_2", ["token5"]), ("id_3", ["token5"])]
    # )
    # assert sorted(in_memory_index_store.get_docs(tokens=["token1", "token2"])) == sorted(
    #     [("id_1", ["token1", "token2"]), ("id_2", ["token2"]), ("id_3", ["token2"])]
    # )


def test_filter():
    matched_docs = [
        ("id1", ("token1")),
        ("id2", ("token2", "token4")),
        ("id3", ("token3")),
    ]
    filtered_doc_ids_with_score = {"id1": 0.3, "id2": 0.1, "id3": 0.5}
    assert EmbeddingInMemoryIndexStore()._filter(
        matched_docs, filtered_doc_ids_with_score, 0.2
    ) == [("id1", ("token1")), ("id3", ("token3"))]


def test_sort():
    filtered_docs = [("id1", ("token1")), ("id3", ("token3"))]
    filtered_doc_ids_with_score = {"id1": 0.3, "id3": 0.5}
    assert EmbeddingInMemoryIndexStore()._sort(
        filtered_docs, filtered_doc_ids_with_score
    ) == [("id3", ("token3")), ("id1", ("token1"))]
