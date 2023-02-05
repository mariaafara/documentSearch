import json
from typing import Dict

import pandas as pd
from tqdm import tqdm

from search.document import Document
from search.index_store import InMemoryIndexStore, EmbeddingInMemoryIndexStore
from search.indexer import EmbeddingIndexer, Indexer
from search.processor import EmbeddingProcessor, Processor
from search.searcher import Searcher, EmbeddingSearcher


def load_documents(excel_path) -> Dict[str, Document]:
    documents_df = pd.read_excel(excel_path, sheet_name="documents")
    documents_df = documents_df[~documents_df.text.isnull()]
    documents = {}
    for i, row in tqdm(documents_df.iterrows(), desc="Loading documents"):
        documents[row["id"]] = Document(
            id=row["id"],
            extracted_at=row["extracted"],
            lang=row["lang"],
            text=row["text"],
        )
    return documents


def save(company_id, search_result):
    filtered_docs_records = []
    if search_result:
        for doc_id, mentions in tqdm(search_result, desc="Store search results."):
            filtered_docs_record = {
                "extracted": documents[doc_id].extracted_at,
                "id": documents[doc_id].id,
                "lang": documents[doc_id].lang,
                "text": documents[doc_id].text,
                "mentions": mentions,
            }

            filtered_docs_records.append(filtered_docs_record)

        pd.DataFrame(filtered_docs_records).to_csv(
            f"data/output/{company_id}.csv", sep=",", index=False
        )


if __name__ == "__main__":
    with open("data/input/companies.json", "rb") as jf:
        companies = json.load(jf)

    search_type = "semantic"  # or exact_match

    documents = load_documents("data/input/documents.xlsx")

    n_gram = 4
    similarity_threshold = 0.4
    if search_type == "exact-match":
        processor = Processor(n=n_gram)
        in_memory_index_store = InMemoryIndexStore()
        indexer = Indexer(processor, in_memory_index_store)
        searcher = Searcher(processor, in_memory_index_store)

    else:
        processor = EmbeddingProcessor(n=n_gram)
        in_memory_index_store = EmbeddingInMemoryIndexStore()
        indexer = EmbeddingIndexer(processor, in_memory_index_store)
        searcher = EmbeddingSearcher(
            processor, in_memory_index_store, similarity_threshold=similarity_threshold
        )

    indexer.index_docs(list(documents.values())[:])
    # in_memory_index_store.load()
    for company_id, company_terms in tqdm(companies.items(), desc="Search companies."):
        print(f"searching for company {company_id} with keywords: {company_terms}")
        search_result = searcher.search(query_terms=company_terms)
        save(company_id, search_result)

        print("*" * 10)
