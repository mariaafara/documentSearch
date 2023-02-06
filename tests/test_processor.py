from search.processor.processor import Processor
from search.processor.embedding_processor import EmbeddingProcessor


def test_get_ngrams():
    processor = Processor()
    doc = "Orange."
    ngrams = processor._get_ngrams(processor._tokenize(doc), n=2)
    assert ngrams == [('Orange.',)]

    doc = "Orange Country"
    ngrams = processor._get_ngrams(processor._tokenize(doc), n=2)
    assert sorted(ngrams) == sorted([('Orange', 'Country'), ('Country',), ('Orange',)])

    doc = "Orange Country"
    ngrams = processor._get_ngrams(processor._tokenize(doc), n=3)
    assert sorted(ngrams) == sorted([('Orange',), ('Orange', 'Country'), ('Country',)])

    doc = "Orange Country is"
    ngrams = processor._get_ngrams(processor._tokenize(doc), n=2)
    assert sorted(ngrams) == sorted([('Country', 'is'), ('is',), ('Orange',), ('Country',), ('Orange', 'Country')])
