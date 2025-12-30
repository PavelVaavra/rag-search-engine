import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
    
def normalize(scores):
    # If no scores are given don't print anything
    # If the minimum and maximum scores are the same, print a list of 1.0 values
    # Otherwise, print the new normalized scores using the min-max normalization described above
    # Print in the following format using print(f"* {score:.4f}") to only print 4 decimal places:
    maximum = max(scores)
    minimum = min(scores)
    if maximum == minimum:
        normalized_scores = [1.0 for i in range(len(scores))]
    else:
        normalized_scores = [(score - minimum) / (maximum - minimum) for score in scores]

    for normalize_score in normalized_scores:
        print(f"* {normalize_score:.4f}")