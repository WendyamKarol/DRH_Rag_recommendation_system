# src/embeddings/faiss_store.py

import faiss
import numpy as np
from typing import List
import pickle
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.embeddings.base import BaseEmbeddingStore
from src.models import TrainingDocument
from config.config import config


class FAISSEmbeddingStore(BaseEmbeddingStore):
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=config.model.embedding_model,
            openai_api_key=config.openai_api_key
        )
        self.index = faiss.IndexFlatL2(config.model.embedding_dimension)
        self.documents = []
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.retrieval.chunk_size,
            chunk_overlap=config.retrieval.chunk_overlap
        )

    def add_documents(self, documents: List[TrainingDocument]) -> None:
        chunks = []
        for doc in documents:
            parts = self.text_splitter.split_text(doc.content)
            for part in parts:
                self.documents.append(doc)
                chunks.append(part)

        vectors = self.embeddings.embed_documents(chunks)
        self.index.add(np.array(vectors).astype("float32"))

    def search(self, query_text: str, k: int) -> List[TrainingDocument]:
        query_vec = self.embeddings.embed_query(query_text)
        distances, indices = self.index.search(np.array([query_vec]).astype("float32"), k)
        results = [self.documents[i] for i in indices[0] if i < len(self.documents)]
        return results

    def save(self, path: str) -> None:
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.docs", "wb") as f:
            pickle.dump(self.documents, f)

    def load(self, path: str) -> None:
        self.index = faiss.read_index(f"{path}.index")
        with open(f"{path}.docs", "rb") as f:
            self.documents = pickle.load(f)
