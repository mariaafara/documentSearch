# Document Filtering

This project aims to build a system that filters documents based on a set of specified company
names/keywords. The system takes two input files: one with the documents to be filtered and another with the
companies of interest and their associated
keywords. The output will be a set of CSV files, with one file per company, containing the filtered documents and the
mentions found in each document.

## Input files

The input files consist of a JSON file with companies, IDs, and their keywords, and a compressed CSV file with the
documents to filter.

- The JSON file has the following format:
  ```python
  {"ID001": {["Apple", "Apple Inc"]}, "ID002":["Orange",...], ...}
  ```

- The compressed CSV file with the documents to filter contains the following columns:
    - extracted: string <!--timestamp-->
    - id: string
    - lang: string
    - text: string

- The expected output is a set of CSV files, with one file per company, containing the filtered documents and the
  mentions
  found in each document based on the input query. The columns of the output file are as follows:
    - extracted: string <!--timestamp-->
    - id: string
    - lang: string
    - text: string
    - mentions: list of strings

----

----

## Proposed Solution

This solution implements a hybrid document search engine that combines N-gram extraction and semantic embeddings to
compare similarity between documents and a query. The process involves extracting N-grams from company keywords,
computing embeddings for each N-gram and calculating the query embedding from their average. An inverted index is used
to find which documents contain the N-grams (based on exact match). Cosine similarity is then computed between the query
embedding and selected documents, which are ranked in ascending order and deselected if their similarity score is below
a threshold. The remaining documents are returned with the N-grams that initially led to their selection.

---

### Pipeline

- indexing
- searching/filtering

#### Preprocessing

Both input query (keywords) and documents to be filtered are clean and pre-processed.

##### Cleaning and preprocessing steps are:

- removing punctuations
- removing new lines
- lowercase
- tokenization
- removing stop words
- lemmatization
- ngrams generation

---

#### Indexing

An indexStore is initiated which aims at creating an inverted index which is a data structure used to store the
mapping between terms (ngrams) and the documents they appear in. It is called an inverted index because it inverts the
original mapping from documents to terms. Instead of mapping from documents to terms, it maps from terms to documents.
The inverted index stores a list of documents for each term (ngram), allowing for fast and efficient retrieval of the
documents containing a specific term (ngram).

e.g.:

```python
ngrams_indices_dict = {"apple": [doc_id1, doc_id1, ...],
                       "france telcom": [doc_id5, doc_id7, doc_id11, ...], ...}
```

We are also storing the document embedding of each document that was indexed.

e.g.:

  ```python
  document_indices = {doc_id5: doc_id5_embedding,
                      doc_id10: doc_id10_embedding, ...}
  ```

---

#### Searching

The process starts by extracting N-grams from each company keyword, then computing an embedding for each N-gram. The
query embedding is calculated based on the average of the N-gram embeddings.

Next, an inverted index is used to lookup which documents contains any of the N-grams extracted (query ngrams). This
step is based on exact match, which means that the documents will only be selected if they contain an exact match of the
N-grams.

Once the relevant documents have been selected, a cosine similarity is computed between the query embedding and each of
the selected documents. The documents are then ranked in ascending order based on the similarity score and those that
have a similarity score less than the input similarity threshold are deselected.

Finally, the remaining documents are returned in order along with the N-grams that led to their initial selection.

---

---

## Run with docker

### Build the image

```bash
docker build -t doc_search .
```

### Run the container in the bash shell

```bash
docker run -it -v $(pwd)/data/:/documentSearch/data doc_search /bin/bash
```

- `-it` specifies that you want to run the container in an interactive mode
- `-v` option is used to specify the volume mount. The first part of the volume mount, `$(pwd)/data/`, is the host
  directory that you want to mount. The second part, `/documentSearch/data`, is the container directory where the host
  directory will
  be mounted.
- the final argument, `/bin/bash`, specifies the command to run in the container. It allows you to run commands inside
  the
  container. You can exit the shell by typing `exit`.