# tests/test_embeddings.py

from src.embeddings.faiss_store import FAISSEmbeddingStore
from src.models import TrainingDocument


def test_faiss_store_add_and_search():
    store = FAISSEmbeddingStore()

    # Simuler un petit corpus de documents
    docs = [
        TrainingDocument(
            content="Ce programme de formation en gestion de projet vous aide à planifier et piloter efficacement.",
            type="programme de formation",
            source="test_source_1",
            doc_id="doc1"
        ),
        TrainingDocument(
            content="Apprenez à améliorer la communication interpersonnelle et à gérer les conflits.",
            type="programme de formation",
            source="test_source_2",
            doc_id="doc2"
        )
    ]

    # Ajouter les documents
    store.add_documents(docs)

    # Recherche par texte
    results = store.search("gestion de projet", k=1)

    assert len(results) > 0, "La recherche devrait retourner au moins un document"
    assert isinstance(results[0], TrainingDocument), "Le résultat doit être un TrainingDocument"
