from typing import Dict

import pandas as pd
import json
from search.document import Document
from search.index_store.in_memory import InMemoryIndexStore
from search.indexer import Indexer
from search.processor import Processor
from search.searcher import Searcher


def load_documents(excel_path) -> Dict[str, Document]:
    print("Loading documents...")
    documents_df = pd.read_excel(excel_path, sheet_name="documents")
    documents_df = documents_df[~documents_df.text.isnull()]
    documents = {}
    for i, row in documents_df.iterrows():
        documents[row["id"]] = Document(
            id=row["id"],
            extracted_at=row["extracted"],
            lang=row["lang"],
            text=row["text"],
        )
    return documents


if __name__ == "__main__":
    with open("data/input/companies.json", "rb") as jf:
        companies = json.load(jf)

    documents = load_documents("data/input/documents.xlsx")
    processor = Processor()
    in_memory_index_store = InMemoryIndexStore()
    indexer = Indexer(processor, in_memory_index_store)
    indexer.index_docs(list(documents.values()))

    searcher = Searcher(processor, in_memory_index_store)
    for company_id, company_terms in companies.items():
        search_result = searcher.search(query_terms=company_terms)
        filtered_docs_records = []
        if search_result:
            filtered_docs_record = {}
            for doc_id, mentions in search_result:
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

        print(
            "Search Query for company ",
            company_id,
            "of terms: ",
            company_terms,
            "appeared in ",
            len(search_result),
        )
        print("*" * 10)
