from search.index_store.in_memory import InMemoryIndexStore


def test_in_memory_index_store():
    index_store = InMemoryIndexStore()
    index_store.add_doc(doc_id="id_1", ngrams=["token1", "token2", "token3"])
    index_store.add_doc(doc_id="id_2", ngrams=["token2", "token4", "token5"])
    index_store.add_doc(doc_id="id_3", ngrams=["token2", "token5", "token6"])

    assert sorted(index_store.get_docs(tokens=["token2"])) == sorted(
        [("id_1", ["token2"]), ("id_2", ["token2"]), ("id_3", ["token2"])]
    )
    assert sorted(index_store.get_docs(tokens=["token1"])) == sorted(
        [("id_1", ["token1"])]
    )
    assert sorted(index_store.get_docs(tokens=["token5"])) == sorted(
        [("id_2", ["token5"]), ("id_3", ["token5"])]
    )
    assert sorted(index_store.get_docs(tokens=["token1", "token2"])) == sorted(
        [("id_1", ["token1", "token2"]), ("id_2", ["token2"]), ("id_3", ["token2"])]
    )
