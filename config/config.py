import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    openai_model: str = "gpt-4"
    embedding_model: str = "text-embedding-ada-002"
    temperature: float = 0.7
    max_tokens: int = 2000
    embedding_dimension: int = 1536


@dataclass
class DataConfig:
    employees_file: str = "data/employee.json"
    training_corpus_file: str = "data/formation.json"
    
     # Configuration FAISS
    faiss_index_path: str = "data/faiss_index"

    faiss_index_paths: dict = field(default_factory=lambda: {
    "FORMATION": "data/faiss_index_formation",
    "BEST_PRACTICES": "data/faiss_index_best",
    "CASE_STUDY": "data/faiss_index_case"
})



@dataclass
class RetrievalConfig:
    top_k: int = 5
    similarity_threshold: float = 0.5
    chunk_size: int = 500
    chunk_overlap: int = 50


@dataclass
class SystemConfig:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    retrieval: RetrievalConfig = RetrievalConfig()

    def get_priority_level(self, score: int) -> str:
        if score < 65:
            return "Élevée"
        elif score < 75:
            return "Moyenne"
        else:
            return "Faible"


config = SystemConfig()

# Mappings globaux
COMPETENCY_KEYWORDS = {
    "gestion_projet": ["gestion de projet", "planification", "organisation"],
    "communication": ["communication", "présentation", "écoute"],
    "leadership": ["leadership", "management", "équipe"],
    "technique": ["technique", "digital", "informatique"],
    "stress": ["stress", "pression", "émotion"],
    "autonomie": ["autonomie", "initiative", "proactif"],
    "analyse": ["analyse", "réflexion", "synthèse"],
    "temps": ["temps", "délais", "planning"],
    "equipe": ["équipe", "collaboration", "synergie"],
    "innovation": ["innovation", "créativité", "nouveauté"]
}

DOCUMENT_TYPES = {
    "FORMATION": "programme de formation",
    "BEST_PRACTICES": "meilleures pratiques",
    "CASE_STUDY": "étude de cas"
}
