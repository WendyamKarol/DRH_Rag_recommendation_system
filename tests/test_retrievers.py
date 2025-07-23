# tests/test_retrievers.py

from src.embeddings.faiss_store import FAISSEmbeddingStore
from src.models import EmployeeEvaluation, TrainingDocument, CompetencyGap
from src.retrievers.retriever import Retriever
from src.retrievers.base import RetrievalContext


def test_retriever():
    # Initialiser un store avec deux documents
    store = FAISSEmbeddingStore()
    docs = [
        TrainingDocument(
            content="Cette formation en communication vous aide à mieux collaborer et écouter activement.",
            type="programme de formation",
            source="test_source_1",
            doc_id="doc1"
        ),
        TrainingDocument(
            content="Améliorez vos compétences en gestion de projet avec cette formation structurée.",
            type="programme de formation",
            source="test_source_2",
            doc_id="doc2"
        )
    ]
    store.add_documents(docs)

    # Évaluation simulée
    evaluation = EmployeeEvaluation(
        employe="Jean Martin",
        evaluation="Besoin de renforcement en communication",
        score=70
    )

    # Lacune simulée
    gaps = [CompetencyGap(competency="communication", keywords_found=["communication"])]

    context = RetrievalContext(
        employee_evaluation=evaluation,
        competency_gaps=gaps
    )

    # Exécution
    retriever = Retriever(store)
    results = retriever.retrieve(context, k=2)

    assert len(results) > 0, "Le retriever doit retourner au moins un document"
    assert isinstance(results[0], TrainingDocument), "Les résultats doivent être des TrainingDocument"



def test_retriever_high_priority_query():
    store = FAISSEmbeddingStore()
    docs = [
        TrainingDocument(
            content="Cette formation vous apprend à gérer le stress au travail et améliorer votre bien-être.",
            type="programme de formation",
            source="test_source_3",
            doc_id="doc3"
        ),
        TrainingDocument(
            content="Leadership et management : comment motiver vos équipes et prendre de bonnes décisions.",
            type="programme de formation",
            source="test_source_4",
            doc_id="doc4"
        )
    ]
    store.add_documents(docs)

    evaluation = EmployeeEvaluation(
        employe="Lucie Petit",
        evaluation="Besoin d'accompagnement en gestion du stress et leadership.",
        score=60  
    )

    gaps = [
        CompetencyGap(competency="stress", keywords_found=["stress"]),
        CompetencyGap(competency="leadership", keywords_found=["leadership"])
    ]

    context = RetrievalContext(
        employee_evaluation=evaluation,
        competency_gaps=gaps
    )

    retriever = Retriever(store)
    results = retriever.retrieve(context, k=2)

    assert len(results) > 0, "Le retriever doit retourner des documents"
    contents = [doc.content.lower() for doc in results]
    assert any("stress" in c or "leadership" in c for c in contents), "Les résultats doivent être liés au stress ou au leadership"
