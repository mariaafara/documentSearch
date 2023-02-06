import logging
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from tqdm import tqdm

from search.index_store import EmbeddingInMemoryIndexStore
from search.indexer import EmbeddingIndexer
from search.processor import EmbeddingProcessor
from search.searcher import EmbeddingSearcher

app = FastAPI()

log = logging.getLogger("SearchEngine")


@app.on_event("startup")
async def startup_event():
    app.state.index_store = EmbeddingInMemoryIndexStore()
    processor = EmbeddingProcessor(n=3)
    app.state.indexer = EmbeddingIndexer(
        processor=processor, index_store=app.state.index_store
    )
    app.state.searcher = EmbeddingSearcher(
        processor=processor, index_store=app.state.index_store, similarity_threshold=0.2
    )
    try:
        app.state.index_store.load(path="/documentSearch/data")
    except Exception as e:
        logging.info(e)


@app.on_event("shutdown")
def shutdown_event():
    try:
        app.state.index_store.save(path="/documentSearch/data")
    except Exception as e:
        log.error(e)


class Document(BaseModel):
    """Model that represents a document."""

    extracted_at: str
    id: str
    lang: Optional[str] = "english"
    text: str


@app.post("/index", status_code=200, tags=["index"])
def index_docs(documents: List[Document]):
    indexer: EmbeddingIndexer = app.state.indexer
    indexer.index_docs(documents)
    return JSONResponse(
        content={"msg": f"{len(documents)} have been indexed"}, status_code=200
    )


@app.post("/query", status_code=200, tags=["query"])
def run_query(query: List[str]):
    searcher: EmbeddingSearcher = app.state.searcher
    query_result = searcher.search(query)
    return JSONResponse(
        content={res[0]: res[1] for res in query_result}, status_code=200
    )


@app.post("/initiate", status_code=200, tags=["index"])
def initiate():
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

    indexer: EmbeddingIndexer = app.state.indexer
    documents = load_documents("/documentSearch/data/input/documents.xlsx")
    indexer.index_docs(list(documents.values()))
    return JSONResponse(
        content={"msg": f"{len(documents)} have been indexed"}, status_code=200
    )
