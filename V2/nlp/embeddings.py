from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    def encode(self, texts, to_tensor=True):
        return self.model.encode(texts, convert_to_tensor=to_tensor)