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