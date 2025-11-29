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
    
    with open("data/movies.json", "r") as file:
        movies = json.load(file)["movies"]

    found_movies = []
    for movie in movies:
        movie_tokens = process_str(movie["title"])
        
        try:
            for movie_token in movie_tokens:
                for keyword_token in keyword_tokens:
                    if keyword_token in movie_token:
                        found_movies.append(movie["title"])
                        # terminate both for loops
                        raise StopIteration
        except StopIteration:
            pass

        if len(found_movies) == 5:
            break
        
    for i, movie in enumerate(found_movies):
        print(f"{i + 1} {movie}")

import pickle
from pathlib import Path

class InvertedIndex():
    def __init__(self, path):
        # path to json data
        self.path = path
        # a dictionary mapping tokens (strings) to sets of document IDs (integers).
        self.index = {}
        # a dictionary mapping document IDs to their full document objects.
        self.docmap = {}

    def __add_document(self, doc_id, text):
        """tokenize the input text, then add each token to the index with the document ID"""
        text_tokens = process_str(text)
        for text_token in text_tokens:
            try:
                self.index[text_token.lower()].add(doc_id)
            except KeyError:
                self.index[text_token.lower()] = {doc_id}

    def get_documents(self, term):
        """get the set of document IDs for a given token, and return them as a list, sorted in ascending order"""
        # lowercase term
        try:
            return sorted(self.index[term.lower()])
        except IndexError:
            return []

    def build(self):
        """iterate over all the movies from self.path and add them to both the index and the docmap"""
        # When adding the movie data to the index with __add_document(), concatenate the title and the description and use that as the input text. 
        # For example: f"{m['title']} {m['description']}"
        with open(self.path, "r") as file:
            movies = json.load(file)["movies"]

        for movie in movies:
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")

            self.docmap[movie["id"]] = movie

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

def build_idx():
    inverted_idx = InvertedIndex("data/movies.json")
    inverted_idx.build()
    inverted_idx.save()
    docs = inverted_idx.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")
