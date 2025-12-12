from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json

from .search_utils import (
    CACHE_DIR,
    MOVIE_EMBEDDINGS_FILE_NAME,
    DATA_PATH
)

class SemanticSearch():
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if not text or text.isspace():
            raise ValueError("text is empty or only whitespace")
        
        return self.model.encode([text])[0]
    
    # Add a new build_embeddings(self, documents) method. documents is a list of dictionaries, each representing a movie.
    # Set self.documents equal to the documents argument.
    # For each document, add an entry to self.document_map where the key is the id of the document and the value is the document itself.
    # Create a string representation of each document (movie) and store them all in a list. Each string should have this format: f"{doc['title']}: {doc['description']}"
    # Use the model's encode method on the list of movie strings. Set the show_progress_bar argument on encode to True so you can see the progress (it takes a while). Store the result as self.embeddings.
    # Save the embeddings into cache/movie_embeddings.npy using np.save.
    # Return self.embeddings from the method.
    def build_embeddings(self, documents):
        self.documents = documents
        movie_strings = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            movie_strings.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)

        with open(os.path.join(CACHE_DIR, MOVIE_EMBEDDINGS_FILE_NAME),"wb") as file:
            np.save(file, self.embeddings)

        return self.embeddings
    
    # We want to build the embeddings only once, and thereafter load them from disk. Add a new load_or_create_embeddings(self, documents) method to 
    # the SemanticSearch class.
    # Populate self.documents and self.document_map in this method as well, just like in build_embeddings.
    # Check if the cache/movie_embeddings.npy file exists.
    # If it does, call np.load to load the embeddings from that file into self.embeddings.
    # Verify that the length of self.embeddings is equal to the length of documents. If it is, return the cached self.embeddings.
    # Otherwise, return the result of rebuilding the embeddings from scratch with self.build_embeddings(documents).
    def load_or_create_embeddings(self, documents):
        if not os.path.exists(os.path.join(CACHE_DIR, MOVIE_EMBEDDINGS_FILE_NAME)):
            return self.build_embeddings(documents)
        
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc

        with open(os.path.join(CACHE_DIR, MOVIE_EMBEDDINGS_FILE_NAME),"rb") as file:
            self.embeddings = np.load(file)

        if len(self.embeddings) == len(self.documents):
            return self.embeddings
                                 

def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def embed_text(text):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

# Add a new top-level verify_embeddings function. It should:
# Create an instance of SemanticSearch.
# Load the documents from movies.json into a list.
# Call load_or_create_embeddings with those movie documents.
# Print the number of documents and the shape of the embeddings:
def verify_embeddings():
    semantic_search = SemanticSearch()
    with open(DATA_PATH, "r") as file:
        documents = json.load(file)["movies"]

    embeddings = semantic_search.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")