import json
import string
from nltk.stem import PorterStemmer

def process_str(text):
    """lowercase, delete punctuation, split words, 
    remove empty tokens and stopwords, stemming"""
    translator = str.maketrans("", "", string.punctuation)

    with open("data/stopwords.txt", "r") as file:
        stop_words = file.read().splitlines()

    stemmer = PorterStemmer()
    
    processed_s = text.lower().translate(translator).split()
    processed_s = [stemmer.stem(s) for s in processed_s if s != "" and s not in stop_words]

    return processed_s

def get_movies_by_keyword(keyword):
    keyword_tokens = process_str(keyword)

    inverted_idx = InvertedIndex("data/movies.json")
    try:
        inverted_idx.load()
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
                    if len(found_movies) == 5:
                        raise StopIteration
    except StopIteration:
        pass
        
    for movie in found_movies:
        print(f"{movie["id"]} {movie["title"]}")

import pickle
from pathlib import Path
from collections import Counter

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

    def __add_document(self, doc_id, text):
        """tokenize the input text, then add each token to the index with the document ID"""
        text_tokens = process_str(text)
        self.term_frequencies[doc_id].update(text_tokens)
        for text_token in text_tokens:
            try:
                self.index[text_token].add(doc_id)
            except KeyError:
                self.index[text_token] = {doc_id}

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
        
        try:
            return self.term_frequencies[doc_id][term_token[0]]
        except KeyError:
            return 0

    def save(self):
        """save the index and docmap attributes to disk using the pickle module's dump function"""
        # Use the file path/name cache/index.pkl for the index.
        # Use the file path/name cache/docmap.pkl for the docmap.
        # Have this method create the cache directory if it doesn't exist (before trying to write files into it).
        Path("cache").mkdir(parents=True, exist_ok=True)

        with open("cache/index.pkl", "wb") as file:
            pickle.dump(self.index, file)

        with open("cache/docmap.pkl", "wb") as file:
            pickle.dump(self.docmap, file)

        with open("cache/term_frequencies.pkl", "wb") as file:
            pickle.dump(self.term_frequencies, file)

    def load(self):
        """load the index and docmap from disk using the pickle module's load function"""
        # use cache/index.pkl for the index
        # use cache/docmap.pkl for the docmap
        # raise an error if the files don't exist
        try:
            with open("cache/index.pkl", "rb") as file:
                self.index = pickle.load(file)
        except FileNotFoundError:
            raise Exception("cache/index.pkl is missing")
        
        try:
            with open("cache/docmap.pkl", "rb") as file:
                self.docmap = pickle.load(file)
        except FileNotFoundError:
            raise Exception("cache/docmap.pkl is missing")
        
        try:
            with open("cache/term_frequencies.pkl", "rb") as file:
                self.term_frequencies = pickle.load(file)
        except FileNotFoundError:
            raise Exception("cache/term_frequencies.pkl is missing")

def build_idx():
    inverted_idx = InvertedIndex("data/movies.json")
    inverted_idx.build()
    inverted_idx.save()

def get_tf(id, term):
    inverted_idx = InvertedIndex("data/movies.json")
    try:
        inverted_idx.load()
    except Exception as e:
        print(e)

    return inverted_idx.get_tf(id, term)

import math

def get_idf(term):
    inverted_idx = InvertedIndex("data/movies.json")
    try:
        inverted_idx.load()
    except Exception as e:
        print(e)

    doc_count = len(inverted_idx.docmap)
    term_doc_count = len(inverted_idx.get_documents(process_str(term)[0]))
    return math.log((doc_count + 1) / (term_doc_count + 1))

def get_tfidf(id, term):
    return get_tf(id, term) * get_idf(term)