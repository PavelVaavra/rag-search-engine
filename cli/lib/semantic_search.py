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
    
    def search(self, query, limit):
        # Generate an embedding for the query using self.generate_embedding(query).
        # Calculate cosine similarity between the query embedding and each document embedding.
        # Create a list of (similarity_score, document) tuples.
        # Sort the list by similarity score in descending order.
        # Return the top results (up to limit) as a list of dictionaries, each containing:
        #     score: The cosine similarity score
        #     title: The movie title
        #     description: The movie description
        if len(self.embeddings) == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        embedding = self.generate_embedding(query)
        similarity = []
        for i, doc in enumerate(self.document_map.values()):
            similarity.append((cosine_similarity(self.embeddings[i], embedding), doc))

        sorted_similarity = sorted(similarity, key=lambda item: item[0], reverse=True)

        sorted_similarity = sorted_similarity[:limit]
        result = []
        for item in sorted_similarity:
            result_item = {}
            result_item["score"] = item[0]
            result_item["title"] = item[1]["title"]
            result_item["description"] = item[1]["description"]
            result.append(result_item)

        return result


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

def embed_query_text(query):
    # Create an instance of the SemanticSearch class.
    # Call your existing generate_embedding method with the provided query (this method already handles whitespace stripping and validation).
    # Print information about the query and its embedding:
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def search(query, limit):
    # It should accept a positional query string argument.
    # Accept an optional --limit argument (default 5).
    # It should create a SemanticSearch instance.
    # Load movies and load/create embeddings.
    # Call the search method with the query and limit.
    # Print the results in this format:
    semantic_search = SemanticSearch()
    with open(DATA_PATH, "r") as file:
        documents = json.load(file)["movies"]

    semantic_search.load_or_create_embeddings(documents)

    top_similarities = semantic_search.search(query, limit)
    
    for i, movie in enumerate(top_similarities):
        print(f"{i + 1}. {movie["title"]} (score: {movie["score"]:.4f})")
        print(f"{movie["description"][:100]}...")
        print("===========================")

def chunk(text, size, overlap):
    words = text.split()
    chunks = []
    while len(words) > size:
        chunks.append(words[:size])
        words = words[size:]
    chunks.append(words)

    if len(chunks) > 1 and overlap > 0:
        for i in range(1, len(chunks)):
            chunks[i] = chunks[i - 1][-overlap:] + chunks[i]
    
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {" ".join(chunk)}")