from unittest.mock import patch
from src.generators.generator import Generator
from src.generators.base import GenerationContext
from src.models import (
    EmployeeEvaluation,
    CompetencyGap,
    TrainingDocument
)


@patch("src.generators.generator.Generator._call_openai")
def test_generate_recommendation(mock_call_openai):
    # Simule une réponse JSON renvoyée par l’API OpenAI
    mock_call_openai.return_value = """
    {
      "plan": "Renforcer les compétences en communication.",
      "formations": [
        {
          "titre": "Formation communication efficace",
          "type": "Programme de formation",
          "duree": "6 heures",
          "pertinence": "Permet d'améliorer les échanges interpersonnels"
        }
      ]
    }
    """

    # Évaluation RH simulée
    evaluation = EmployeeEvaluation(
        employe="Julie Lambert",
        evaluation="Doit renforcer sa communication",
        score=65
    )

    # Lacune détectée
    competency_gaps = [
        CompetencyGap(competency="communication", keywords_found=["communication"])
    ]

    # Document de formation directement (TrainingDocument)
    training_doc = TrainingDocument(
        doc_id="doc1",
        type="programme de formation",
        source="fichier_test.json",
        content="Améliorer sa communication orale et écrite pour mieux interagir en milieu professionnel."
    )

    # Contexte de génération
    context = GenerationContext(
        employee_evaluation=evaluation,
        competency_gaps=competency_gaps,
        retrieved_documents=[training_doc],  # ✅ directement des TrainingDocument
        priority_level="Moyenne"
    )

    # Génération de la recommandation
    generator = Generator()
    recommendation = generator.generate(context)

    # Assertions
    assert recommendation.employe == "Julie Lambert"
    assert recommendation.niveau_priorite == "Moyenne"
    assert recommendation.plan_developpement.startswith("Renforcer")
    assert len(recommendation.formations_recommandees) == 1
    assert recommendation.formations_recommandees[0]["titre"] == "Formation communication efficace"
