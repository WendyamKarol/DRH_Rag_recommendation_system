from typing import List
from src.agents.base_agent import AgentRAG
from src.embeddings.faiss_store import FAISSEmbeddingStore
from src.generators.generator import Generator
from src.models import TrainingDocument
from config.config import config

class BestPracticesAgent(AgentRAG):
    def __init__(self):
        self.index = FAISSEmbeddingStore()
        #self.index.load(f"{config.data.faiss_index_path}_best")  # Exemple : data/faiss_index_best
        self.index.load(config.data.faiss_index_paths["BEST_PRACTICES"])

        self.generator = Generator()

    def retrieve(self, query: str) -> List[TrainingDocument]:
        return self.index.search(query_text=query, k=config.retrieval.top_k)

    def generate_partial_recommendation(self, evaluation_text: str, documents: List[TrainingDocument]) -> str:
        content_snippets = "\n".join([f"- {doc.content[:200]}" for doc in documents])

        prompt = f"""
Tu es un coach en performance RH. Voici une évaluation RH :

Évaluation : {evaluation_text}
Extraits de bonnes pratiques :
{content_snippets}

Donne une courte recommandation basée sur les bonnes pratiques observées.
"""

        return self.generator._call_openai(prompt)

