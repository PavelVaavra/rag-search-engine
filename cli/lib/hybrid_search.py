import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch, get_documents
from .search_utils import DATA_PATH


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex(DATA_PATH)
        self.idx.load()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)
    
    def _tuple_to_list_bm25_search(self, lst_of_tuples):
        # [(1771, ('Paddington', 10.489448461845111))]
        # convert tuple to list
        keyword_results_lst = []
        for keyword_result in lst_of_tuples:
            lst = []
            lst.append(keyword_result[0])
            lst.append(list(keyword_result[1]))
            keyword_results_lst.append(lst)

        return keyword_results_lst

    def weighted_search(self, query, alpha, limit=5):
        keyword_results = self._tuple_to_list_bm25_search(self._bm25_search(query, 500 * limit))

        keyword_scores = [keyword_result[1][1] for keyword_result in keyword_results]
        keyword_scores_normalized = normalize(keyword_scores)
        for i in range(len(keyword_results)):
            keyword_results[i][1][1] = keyword_scores_normalized[i]

        # [{'id': 2784, 'title': 'Legends of the Fall', 
        #   'description': 'Sick of betrayals the United States government perpetrated on the Native Americans, Colonel William ', 
        #   'score': np.float32(0.5236), 'metadata': {}}]
        semantic_results = self.semantic_search.search_chunks(query, 500 * limit)
        semantic_scores = [semantic_result["score"] for semantic_result in semantic_results]
        semantic_scores_normalized = normalize(semantic_scores)
        for i in range(len(semantic_results)):
            semantic_results[i]["score"] = semantic_scores_normalized[i]

        # Combine the results from both searches as follows:
        # Normalize all the keyword and semantic scores using the normalize method you implemented earlier.
        # Create a dictionary mapping document IDs to the documents themselves and their keyword and semantic scores (I just used a new dictionary to 
        # store this information)
        # Add a third score (the hybrid score) to each document - calculate this using the hybrid_score function described above.
        # Return the results sorted by the hybrid score in descending order.
        # { id: [keyword_score, semantic_score, hybrid_score, title, description] }
        id_to_scores = self._combine_keyword_semantic(keyword_results, semantic_results)

        for key, value in id_to_scores.items():
            id_to_scores[key][2] = hybrid_score(value[0], value[1], alpha)
            id_to_scores[key][3] = f"{self.semantic_search.document_map[key]['title']}"
            id_to_scores[key][4] = f"{self.semantic_search.document_map[key]['description'][:100]}..."

        id_to_scores = dict(sorted(id_to_scores.items(), key=lambda item: item[1][2], reverse=True))
        return dict(list(id_to_scores.items())[:limit])
    
    def _combine_keyword_semantic(self, keyword_results, semantic_results):
        id_to_scores = {}
        for i in range(len(keyword_results)):
            id = keyword_results[i][0]
            score = keyword_results[i][1][1]
            try:
                id_to_scores[id][0] = score
            except KeyError:
                id_to_scores[id] = [score, 0, 0, "", ""]

            id = semantic_results[i]["id"]
            score = semantic_results[i]["score"]
            try:
                id_to_scores[id][1] = score
            except KeyError:
                id_to_scores[id] = [0, score, 0, "", ""]

        return id_to_scores

    def rrf_search(self, query, k, limit=10):
        keyword_results = self._tuple_to_list_bm25_search(self._bm25_search(query, 500 * limit))
        keyword_results.sort(key=lambda item: item[1][1], reverse=True)
        for i in range(len(keyword_results)):
            keyword_results[i][1][1] = i + 1

        semantic_results = self.semantic_search.search_chunks(query, 500 * limit)
        semantic_results.sort(key=lambda item: item["score"], reverse=True)
        for i in range(len(semantic_results)):
            semantic_results[i]["score"] = i + 1

        # { id: [keyword_score, semantic_score, hybrid_score, title, description] }
        id_to_scores = self._combine_keyword_semantic(keyword_results, semantic_results)

        for key, value in id_to_scores.items():
            rrf_score_keyword = 0 if value[0] == 0 else rrf_score(value[0], k)
            rrf_score_semantic = 0 if value[1] == 0 else rrf_score(value[1], k)
            id_to_scores[key][2] = rrf_score_keyword + rrf_score_semantic
            id_to_scores[key][3] = f"{self.semantic_search.document_map[key]['title']}"
            id_to_scores[key][4] = f"{self.semantic_search.document_map[key]['description']}"

        id_to_scores = dict(sorted(id_to_scores.items(), key=lambda item: item[1][2], reverse=True))
        return dict(list(id_to_scores.items())[:limit])

def rrf_score(rank, k=60):
    return 1 / (k + rank)
    
def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score
    
def normalize(scores):
    maximum = max(scores)
    minimum = min(scores)
    if maximum == minimum:
        return [1.0 for i in range(len(scores))]
    return [(score - minimum) / (maximum - minimum) for score in scores]

def weighted_search(query, alpha, limit):
    documents = get_documents(DATA_PATH)
    hybrid_search = HybridSearch(documents)

    return hybrid_search.weighted_search(query, alpha, limit)

def rrf_search(query, k, limit):
    documents = get_documents(DATA_PATH)
    hybrid_search = HybridSearch(documents)

    return hybrid_search.rrf_search(query, k, limit)