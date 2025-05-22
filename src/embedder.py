import numpy as np
from sentence_transformers import SentenceTransformer
from abc import ABC, abstractmethod
from langchain.embeddings import HuggingFaceInstructEmbeddings


class Embedder(ABC):
    @abstractmethod
    def get_embed(self, text):
        pass


class LLMEmbedder(Embedder):
    def __init__(self, model_name="ai-forever/ruElectra-small"):
        self.emb_model = HuggingFaceInstructEmbeddings(
            model_name=model_name
        )
        self.model_name = model_name

    def get_embed(self, text):
        return np.array(self.emb_model.embed_documents([text])).reshape(-1)


class LaBSEEmbedder(Embedder):
    def __init__(self, model_name='sergeyzh/LaBSE-ru-turbo'):
        self.emb_model = SentenceTransformer('sergeyzh/LaBSE-ru-turbo')
        self.model_name = model_name

    def get_embed(self, text):
        return np.array(self.emb_model.encode([text])).reshape(-1)
