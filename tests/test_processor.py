from search.processor.processor import Processor
from search.processor.embedding_processor import EmbeddingProcessor


def test_get_ngrams():
    processor = Processor()
    doc = "Orange."
    ngrams = processor._get_ngrams(processor._tokenize(doc), n=2)
    assert ngrams == {('Orange.',)}

    doc = "Orange Country"
    ngrams = processor._get_ngrams(processor._tokenize(doc), n=2)
    assert ngrams == {('Orange', 'Country')}

    doc = "Orange Country"
    ngrams = processor._get_ngrams(processor._tokenize(doc), n=3)
    assert ngrams == {('Orange', 'Country')}

    doc = "Orange Country is"
    ngrams = processor._get_ngrams(processor._tokenize(doc), n=2)
    assert ngrams == {('Orange', 'Country'), ('Country', 'is')}


def test_embedding_processor():
    doc = "Orange County is a great place to live."
    processor = EmbeddingProcessor()
    ngrams = processor._get_ngrams(processor._tokenize(doc), 3)
    print(ngrams)
    ngrams_embedding = processor.compute_ngrams_embeddings(ngrams)
    assert len(ngrams) == len(ngrams_embedding)
    # print(ngrams_embedding)

    # print(ngrams_embedding[0])
    # print(ngrams_embedding[0].shape)

    # doc_embedding = processor.compute_document_embedding(doc)
    # print(doc_embedding)
    # print(doc_embedding.shape)
