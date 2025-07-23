# tests/test_rag_system.py

from unittest.mock import patch
from src.rag_system import TrainingRAGSystem
from src.models import EmployeeEvaluation, TrainingDocument


@patch("src.generators.simple_openai_generator.SimpleOpenAIGenerator._call_openai")
def test_process_evaluation_pipeline(mock_call):
    # Simuler une réponse JSON valide de l'API OpenAI
    mock_call.return_value = """
    {
      "plan": "Améliorer la communication interpersonnelle.",
      "formations": [
        {
          "titre": "Formation Communication",
          "type": "Programme de formation",
          "duree": "6 heures",
          "pertinence": "Cible les besoins identifiés"
        }
      ]
    }
    """

    # Initialiser le système
    rag = TrainingRAGSystem()

    # Charger un corpus fictif minimal
    rag.embedding_store.add_documents([
        TrainingDocument(
            content="Cette formation améliore les compétences en communication et gestion d'équipe.",
            type="programme de formation",
            source="test",
            doc_id="doc1"
        )
    ])

    # Créer une évaluation d'exemple
    evaluation = EmployeeEvaluation(
        employe="Nora Moreau",
        evaluation="Besoin de renforcer la communication",
        score=67
    )

    # Exécuter le traitement
    recommendation = rag.process_evaluation(evaluation)

    # Vérifications
    assert recommendation.employe == "Nora Moreau"
    assert recommendation.plan_developpement != ""
    assert len(recommendation.formations_recommandees) > 0
    assert recommendation.formations_recommandees[0]["titre"] == "Formation Communication"