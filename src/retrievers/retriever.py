# src/retrievers/retriever.py

from src.retrievers.base import BaseRetriever, RetrievalContext
from src.embeddings.faiss_store import FAISSEmbeddingStore
from src.models import TrainingDocument
from config.config import COMPETENCY_KEYWORDS


class Retriever(BaseRetriever):
    def __init__(self, embedding_store: FAISSEmbeddingStore):
        self.embedding_store = embedding_store

    def retrieve(self, context: RetrievalContext, k: int = 5) -> list[TrainingDocument]:
        query = self.build_query(context)
        return self.embedding_store.search(query, k)

    def build_query(self, context: RetrievalContext) -> str:
        query_parts = []

        for gap in context.competency_gaps:
            keywords = COMPETENCY_KEYWORDS.get(gap.competency, [])
            query_parts.extend(keywords)

        # Ajouter quelques mots de l'évaluation brute (découpage simple)
        query_parts.extend(context.employee_evaluation.evaluation.lower().split())

        # Limite de 20 mots pertinents
        return " ".join(query_parts[:20])
