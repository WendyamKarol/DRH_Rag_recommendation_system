# src/rag_system.py

from src.embeddings.faiss_store import FAISSEmbeddingStore
from src.retrievers import Retriever, RetrievalContext
from src.generators import Generator, GenerationContext
from src.models import EmployeeEvaluation, TrainingRecommendation
from config.config import config,COMPETENCY_KEYWORDS


class TrainingRAGSystem:
    def __init__(self):
        self.embedding_store = FAISSEmbeddingStore()
        self.retriever = Retriever(self.embedding_store)
        self.generator = Generator()

    def load_training_corpus(self, filepath: str):
        import json
        from src.models import TrainingDocument

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents = [
            TrainingDocument(
                content=doc['content'],
                type=doc['type'],
                source=doc['source'],
                doc_id=f"doc_{i}"
            )
            for i, doc in enumerate(data)
        ]

        self.embedding_store.add_documents(documents)
    

    def load_employee_evaluations(self, filepath: str) -> list[EmployeeEvaluation]:
        """Charge les évaluations d'un fichier JSON"""
        import json
        from src.models import EmployeeEvaluation 

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        evaluations = []
        for item in data:
            try:
                evaluations.append(EmployeeEvaluation(**item))
            except Exception as e:
                print(f"Erreur sur {item}: {e}")
        return evaluations


    def process_evaluation(self, evaluation: EmployeeEvaluation) -> TrainingRecommendation:
        gaps = self._extract_competency_gaps(evaluation)
        context_retrieval = RetrievalContext(
            employee_evaluation=evaluation,
            competency_gaps=gaps
        )
        documents = self.retriever.retrieve(context_retrieval, k=config.retrieval.top_k)

        context_generation = GenerationContext(
            employee_evaluation=evaluation,
            competency_gaps=gaps,
            retrieved_documents=documents,
            priority_level=config.get_priority_level(evaluation.score)
        )

        return self.generator.generate(context_generation)

    def _extract_competency_gaps(self, evaluation: EmployeeEvaluation):
        from src.models import CompetencyGap
        text = evaluation.evaluation.lower()
        gaps = []

        for key, keywords in COMPETENCY_KEYWORDS.items():
            found = [kw for kw in keywords if kw in text]
            if found:
                gaps.append(CompetencyGap(competency=key, keywords_found=found))

        return gaps
