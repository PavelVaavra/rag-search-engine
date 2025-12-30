import json
import string
import math
import pickle
import os

from pathlib import Path
from collections import Counter
from nltk.stem import PorterStemmer

from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    DATA_PATH,
    STOPWORDS_PATH,
    CACHE_DIR,
    INDEX_FILE_NAME,
    DOCMAP_FILE_NAME,
    TERM_FREQUENCIES_FILE_NAME,
    DOC_LENGTHS_FILE_NAME,
    BM25_K1,
    BM25_B
)

def process_str(text):
    """lowercase, delete punctuation, split words, 
    remove empty tokens and stopwords, stemming"""
    translator = str.maketrans("", "", string.punctuation)

    with open(STOPWORDS_PATH, "r") as file:
        stop_words = file.read().splitlines()

    stemmer = PorterStemmer()
    
    processed_s = text.lower().translate(translator).split()
    processed_s = [stemmer.stem(s) for s in processed_s if s != "" and s not in stop_words]

    return processed_s

def get_movies_by_keyword(keyword):
    keyword_tokens = process_str(keyword)

    try:
        inverted_idx = get_inverted_idx_load(DATA_PATH)
    except Exception as e:
        print(e)


    found_movies = []
    # instead, iterate over each token in the query, and use the inverted index to get any matching documents for each token. Once you have 5 results, 
    # stop searching and just return them. Print the resulting document titles and IDs.
    try:
        for keyword_token in keyword_tokens:
            docs = inverted_idx.get_documents(keyword_token)
            if docs:
                for doc in docs:
                    found_movies.append({"id": doc, "title": inverted_idx.docmap[doc]["title"]})
                    if len(found_movies) == DEFAULT_SEARCH_LIMIT:
                        raise StopIteration
    except StopIteration:
        pass
        
    for movie in found_movies:
        print(f"{movie["id"]} {movie["title"]}")

class InvertedIndex():
    def __init__(self, path):
        # path to json data
        self.path = path
        # a dictionary mapping tokens (strings) to sets of document IDs (integers).
        self.index = {}
        # a dictionary mapping document IDs to their full document objects.
        self.docmap = {}
        # a dictionary of document IDs to Counter objects for keeping track of how many times each term appears in each document
        self.term_frequencies = {}
        # a dictionary of document IDs to their lengths
        self.doc_lengths = {}

    def __add_document(self, doc_id, text):
        """tokenize the input text, then add each token to the index with the document ID"""
        text_tokens = process_str(text)
        self.term_frequencies[doc_id].update(text_tokens)
        self.doc_lengths[doc_id] = len(text_tokens)
        for text_token in text_tokens:
            try:
                self.index[text_token].add(doc_id)
            except KeyError:
                self.index[text_token] = {doc_id}

    def __get_avg_doc_length(self):
        number_docs = len(self.doc_lengths)
        if number_docs == 0:
            return 0.0
        
        accumulated_length = 0
        for length in self.doc_lengths.values():
            accumulated_length += length

        return accumulated_length / number_docs

    def get_documents(self, term):
        """get the set of document IDs for a given token, and return them as a list, sorted in ascending order"""
        # lowercase term
        try:
            return sorted(self.index[term.lower()])
        except KeyError:
            return []

    def build(self):
        """iterate over all the movies from self.path and add them to both the index and the docmap"""
        # When adding the movie data to the index with __add_document(), concatenate the title and the description and use that as the input text. 
        # For example: f"{m['title']} {m['description']}"
        with open(self.path, "r") as file:
            movies = json.load(file)["movies"]

        for movie in movies:
            self.term_frequencies[movie["id"]] = Counter()
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")

            self.docmap[movie["id"]] = movie

    def get_tf(self, doc_id, term):
        """return the times the token appears in the document with the given ID"""
        # if the term doesn't exist in that document, return 0
        # be sure to tokenize the term, but assume that there is only one token. If there's more than one, raise an exception.
        term_token = process_str(term)
        if len(term_token) != 1:
            raise Exception("term argument is more than one token")
        term_token = term_token[0]
        
        try:
            return self.term_frequencies[doc_id][term_token]
        except KeyError:
            return 0
        
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        # Length normalization factor
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        
        tf = self.get_tf(doc_id, term)
        # Apply to term frequency
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)
        
    def get_bm25_idf(self, term):
        """"""
        term_token = process_str(term)
        if len(term_token) != 1:
            raise Exception("term argument is more than one token")
        term_token = term_token[0]

        N = len(self.docmap)
        df = len(self.get_documents(term_token))        
        return math.log((N - df + 0.5) / (df + 0.5) + 1)
    
    def bm25(self, doc_id, term):
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf
    
    def bm25_search(self, query, limit):
        # Tokenize the query.
        # Initialize a scores dictionary, to map document IDs to their total BM25 scores.
        # For each document in the index, calculate the total BM25 score:
        # For each query token, add its BM25 score to the document's running total.
        # Store the total score in the scores dictionary.
        # Sort the documents by score in descending order.
        # Return the top limit documents along with their scores.
        query_tokens = process_str(query)
        scores = {}
        for doc_id in self.docmap.keys():
            bm25_sum = 0
            for query_token in query_tokens:
                bm25_sum += self.bm25(doc_id, query_token)
            scores[doc_id] = (self.docmap[doc_id]["title"], bm25_sum)
        
        sorted_scores = sorted(scores.items(), key=lambda item: item[1][1], reverse=True)
        return sorted_scores[:limit]
                

    def save(self):
        """save the index and docmap attributes to disk using the pickle module's dump function"""
        # Use the file path/name cache/index.pkl for the index.
        # Use the file path/name cache/docmap.pkl for the docmap.
        # Have this method create the cache directory if it doesn't exist (before trying to write files into it).
        Path("cache").mkdir(parents=True, exist_ok=True)

        with open(os.path.join(CACHE_DIR, INDEX_FILE_NAME), "wb") as file:
            pickle.dump(self.index, file)

        with open(os.path.join(CACHE_DIR, DOCMAP_FILE_NAME), "wb") as file:
            pickle.dump(self.docmap, file)

        with open(os.path.join(CACHE_DIR, TERM_FREQUENCIES_FILE_NAME), "wb") as file:
            pickle.dump(self.term_frequencies, file)

        with open(os.path.join(CACHE_DIR, DOC_LENGTHS_FILE_NAME), "wb") as file:
            pickle.dump(self.doc_lengths, file)

    def load(self):
        """load the index and docmap from disk using the pickle module's load function"""
        # use cache/index.pkl for the index
        # use cache/docmap.pkl for the docmap
        # raise an error if the files don't exist
        try:
            with open(os.path.join(CACHE_DIR, INDEX_FILE_NAME), "rb") as file:
                self.index = pickle.load(file)
        except FileNotFoundError:
            raise Exception("cache/index.pkl is missing")
        
        try:
            with open(os.path.join(CACHE_DIR, DOCMAP_FILE_NAME), "rb") as file:
                self.docmap = pickle.load(file)
        except FileNotFoundError:
            raise Exception("cache/docmap.pkl is missing")
        
        try:
            with open(os.path.join(CACHE_DIR, TERM_FREQUENCIES_FILE_NAME), "rb") as file:
                self.term_frequencies = pickle.load(file)
        except FileNotFoundError:
            raise Exception("cache/term_frequencies.pkl is missing")
        
        try:
            with open(os.path.join(CACHE_DIR, DOC_LENGTHS_FILE_NAME), "rb") as file:
                self.doc_lengths = pickle.load(file)
        except FileNotFoundError:
            raise Exception("cache/doc_lengths.pkl is missing")

def build_idx():
    inverted_idx = InvertedIndex(DATA_PATH)
    inverted_idx.build()
    inverted_idx.save()

def get_tf(id, term):
    try:
        inverted_idx = get_inverted_idx_load(DATA_PATH)
    except Exception as e:
        print(e)

    return inverted_idx.get_tf(id, term)

def bm25_tf_command(id, term, k1=BM25_K1, b=BM25_B):
    try:
        inverted_idx = get_inverted_idx_load(DATA_PATH)
    except Exception as e:
        print(e)

    return inverted_idx.get_bm25_tf(id, term, k1, b)

def get_idf(term):
    try:
        inverted_idx = get_inverted_idx_load(DATA_PATH)
    except Exception as e:
        print(e)

    doc_count = len(inverted_idx.docmap)
    term_doc_count = len(inverted_idx.get_documents(process_str(term)[0]))
    return math.log((doc_count + 1) / (term_doc_count + 1))

def bm25_idf_command(term):
    try:
        inverted_idx = get_inverted_idx_load(DATA_PATH)
    except Exception as e:
        print(e)

    return inverted_idx.get_bm25_idf(term)

def get_tfidf(id, term):
    return get_tf(id, term) * get_idf(term)

def get_inverted_idx_load(DATA_PATH):
    inverted_idx = InvertedIndex(DATA_PATH)
    try:
        inverted_idx.load()
        return inverted_idx
    except Exception as e:
        raise e
    
def get_bm25_search_command(query, limit):
    try:
        inverted_idx = get_inverted_idx_load(DATA_PATH)
    except Exception as e:
        print(e)

    return inverted_idx.bm25_search(query, limit)