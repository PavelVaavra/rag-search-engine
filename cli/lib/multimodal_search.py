from PIL import Image
from sentence_transformers import SentenceTransformer

from .search_utils import DATA_PATH, DEFAULT_SEARCH_LIMIT
from .semantic_search import get_documents, cosine_similarity

class MultimodalSearch:
    def __init__(self, documents, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        
        self.texts = []
        for doc in self.documents:
            self.texts.append(f"{doc["title"]}: {doc["description"]}")

        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, path):
        image = Image.open(path)
        return self.model.encode([image])[0]
    
    def search_with_image(self, path):
        img_embedding = self.embed_image(path)

        similarity_results = []
        for i, text_embedding in enumerate(self.text_embeddings):
            similarity_results.append({
                **self.documents[i], 
                "similarity_score": cosine_similarity(text_embedding, img_embedding)
            })

        similarity_results.sort(key=lambda item: item["similarity_score"], reverse=True)

        return similarity_results[:DEFAULT_SEARCH_LIMIT]
    
def verify_image_embedding(path):
    documents = get_documents(DATA_PATH)
    multimodal_search = MultimodalSearch(documents)
    embedding = multimodal_search.embed_image(path)

    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(path):
    documents = get_documents(DATA_PATH)
    multimodal_search = MultimodalSearch(documents)

    return multimodal_search.search_with_image(path)