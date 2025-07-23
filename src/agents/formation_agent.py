from src.agents.base_agent import AgentRAG
from src.embeddings.faiss_store import FAISSEmbeddingStore
from src.generators.generator import Generator
from src.models import TrainingDocument
from config.config import config
from typing import List


class FormationAgent(AgentRAG):
    def __init__(self):
        self.index = FAISSEmbeddingStore()
        self.index.load(config.data.faiss_index_paths["FORMATION"])
        self.generator = Generator()
    
    def retrieve(self, query:str) -> list[TrainingDocument]:
        return self.index.search(query_text=query,k=config.retrieval.top_k)
    
    
    def generate_partial_recommendation(self, evaluation_text: str, documents: List[TrainingDocument]) -> str:
        # On construit un prompt minimal pour une reco partielle
        content_snippets = "\n".join([f"- {doc.content[:200]}" for doc in documents])
        prompt = f"""
Tu es un expert en formation professionnelle. Voici une évaluation RH :

Évaluation : {evaluation_text}
Documents liés :
{content_snippets}

Donne une courte recommandation de formation (3-5 lignes) pour cet employé.
"""
        # Appel LLM via méthode privée
        return self.generator._call_openai(prompt)



