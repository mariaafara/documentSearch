from collections import defaultdict
from typing import List, Tuple

from search.index_store.index_store import IndexStore


class Searcher:
    """Class to search for documents based on a query."""

    def __init__(self, processor, index_store: IndexStore):
        self.processor = processor
        self.index_store = index_store

    def search(self, query_terms: List[str]) -> List[Tuple[str, List[str]]]:
        """Filter document based on input query terms.

        :param query_terms: List of keywords/terms/phrases.
        :return: List of filtered document ids with the matched keywords in the document.
        """
        result_dict = defaultdict(set)
        queried_terms = set()
        for term in query_terms:
            processed_term = set(self.processor.preprocess(term))
            search_query = processed_term - queried_terms  # query new tokens or n-grams

            # print(processed_term, "--", search_query)
            queried_terms.update(search_query)
            if search_query:
                term_result = self.index_store.get_docs(tokens=list(search_query))
                for doc_id, matched_tokens in term_result:
                    result_dict[doc_id].update(matched_tokens)
        print("-" * 10)
        print(f"queried_terms= {queried_terms}")

        filtered_docs_with_matches = {}
        for doc_id, matched_tokens in result_dict.items():
            filtered_docs_with_matches[doc_id] = list(matched_tokens)

        return list(filtered_docs_with_matches.items())
