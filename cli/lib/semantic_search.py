from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
import re

from .search_utils import (
    CACHE_DIR,
    MOVIE_EMBEDDINGS_FILE_NAME,
    DATA_PATH,
    CHUNK_EMBEDDINGS_FILE_NAME,
    CHUNK_METADATA_FILE_NAME
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
    
class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self):
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        chunks = []
        chunk_metadata = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            if doc["description"] == "":
                continue
            doc_chunks = semantic_chunk(doc["description"], 4, 1)
            chunks.extend(doc_chunks)
            doc_chunk_metadata = []
            for i in range(len(doc_chunks)):
                doc_chunk_metadata_item = {}
                doc_chunk_metadata_item["movie_idx"] = doc["id"]
                doc_chunk_metadata_item["chunk_idx"] = i + 1
                doc_chunk_metadata_item["total_chunks"] = len(doc_chunks)
                doc_chunk_metadata.append(doc_chunk_metadata_item)
            chunk_metadata.extend(doc_chunk_metadata)


        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        with open(os.path.join(CACHE_DIR, CHUNK_EMBEDDINGS_FILE_NAME),"wb") as file:
            np.save(file, self.chunk_embeddings)

        with open(os.path.join(CACHE_DIR, CHUNK_METADATA_FILE_NAME),"w") as file:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(chunks)}, file, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents):
        if not os.path.exists(os.path.join(CACHE_DIR, CHUNK_EMBEDDINGS_FILE_NAME)) and \
            not os.path.exists(os.path.join(CACHE_DIR, CHUNK_METADATA_FILE_NAME)):
            return self.build_chunk_embeddings(documents)
        
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc

        with open(os.path.join(CACHE_DIR, CHUNK_EMBEDDINGS_FILE_NAME),"rb") as file:
            self.chunk_embeddings = np.load(file)

        with open(os.path.join(CACHE_DIR, CHUNK_METADATA_FILE_NAME),"r") as file:
            self.chunk_metadata = json.load(file)["chunks"]

        return self.chunk_embeddings
    
    def search_chunks(self, query, limit=10):
        embedding = self.generate_embedding(query)
        chunk_scores = []
        for i in range(len(self.chunk_embeddings)):
            chunk_scores_item = {}
            chunk_scores_item["chunk_idx"] = self.chunk_metadata[i]["chunk_idx"]
            chunk_scores_item["movie_idx"] = self.chunk_metadata[i]["movie_idx"]
            chunk_scores_item["score"] = cosine_similarity(self.chunk_embeddings[i], embedding)
            chunk_scores.append(chunk_scores_item)

        idx_to_scores = {}
        for chunk_score in chunk_scores:
            try:
                if idx_to_scores[chunk_score["movie_idx"]] < chunk_score["score"]:
                    idx_to_scores[chunk_score["movie_idx"]] = chunk_score["score"]
            except KeyError:
                idx_to_scores[chunk_score["movie_idx"]] = chunk_score["score"]

        idx_to_scores = dict(sorted(idx_to_scores.items(), key=lambda item: item[1], reverse=True))
        idx_to_scores = dict(list(idx_to_scores.items())[:limit])

        results = []
        for i, idx_to_score in enumerate(idx_to_scores.items()):
            movie_idx = idx_to_score[0]
            results.append(
                {
                    "id": movie_idx,
                    "title": self.document_map[movie_idx]["title"],
                    "description": self.document_map[movie_idx]["description"][:100],
                    "score": round(idx_to_score[1], 4),
                    "metadata": {}
                }
            )

        return results

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
    documents = get_documents(DATA_PATH)

    embeddings = semantic_search.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_chunks():
    # Load the movie documents
    # Initialize a ChunkedSemanticSearch instance
    # Load or build the chunk embeddings
    # Print info about the embeddings in this format:
    chunked_semantic_search = ChunkedSemanticSearch()
    documents = get_documents(DATA_PATH)

    embeddings = chunked_semantic_search.load_or_create_chunk_embeddings(documents)

    print(f"Generated {len(embeddings)} chunked embeddings")

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
    documents = get_documents(DATA_PATH)

    semantic_search.load_or_create_embeddings(documents)

    top_similarities = semantic_search.search(query, limit)
    
    for i, movie in enumerate(top_similarities):
        print(f"{i + 1}. {movie["title"]} (score: {movie["score"]:.4f})")
        print(f"{movie["description"][:100]}...")
        print("===========================")

def search_chunked(query, limit):
    # Load the movie documents using load_movies().
    # Initialize a ChunkedSemanticSearch instance.
    # Load or create chunk embeddings.
    # Get the results using the search_chunks method with the given query and limit arguments.
    # Print results in the following format:
    chunked_semantic_search = ChunkedSemanticSearch()
    documents = get_documents(DATA_PATH)

    chunked_semantic_search.load_or_create_chunk_embeddings(documents)

    top_similarities = chunked_semantic_search.search_chunks(query, limit)

    for i, movie in enumerate(top_similarities):
        print(f"\n{i + 1}. {movie["title"]} (score: {movie["score"]:.4f})")
        print(f"   {movie["description"]}...")

def chunk(text, size, overlap):
    words = text.split()
    chunks = []
    while len(words) > size:
        chunks.append(words[:size])
        words = words[size - overlap:]
    chunks.append(words)
    
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {" ".join(chunk)}")

def semantic_chunk(text, max_size, overlap):
    # Strip leading and trailing whitespace from the input text before using the regex to split sentences.
    # If there's nothing left after stripping, return an empty list.
    # After splitting sentences, if there's only one sentence and it doesn't end with a punctuation mark like ., !, or ?, treat the 
    # whole text as one sentence.
    # Strip leading and trailing whitespace from each sentence before appending it to the chunk list.
    # Only use chunks that still have content after the stripping.
    text = text.strip()
    if not text:
        return []
    
    raw_sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [raw_sentence.strip() for raw_sentence in raw_sentences if raw_sentence.strip() is not None]

    if len(sentences) == 1 and sentences[0][-1] not in (".", "!", "?"):
        return sentences

    chunks = []
    while len(sentences) > max_size:
        chunks.append(" ".join(sentences[:max_size]))
        sentences = sentences[max_size - overlap:]
    chunks.append(" ".join(sentences))
    return chunks

def semantic_chunk_print(chunks, text_len):
    print(f"Semantically chunking {text_len} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")

def get_documents(path):
    with open(path, "r") as file:
        return json.load(file)["movies"]