import os

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = "/home/pavel/workspace/github.com/PavelVaavra/rag-search-engine"
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
INDEX_FILE_NAME = "index.pkl"
DOCMAP_FILE_NAME = "docmap.pkl"
TERM_FREQUENCIES_FILE_NAME = "term_frequencies.pkl"
DOC_LENGTHS_FILE_NAME = "doc_lengths.pkl"

BM25_K1 = 1.5
BM25_B = 0.75