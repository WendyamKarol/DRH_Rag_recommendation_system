#from src.agents.formation_agent import FormationAgent
# from src.agents.best_practices_agent import BestPracticesAgent
from src.agents.case_study_agent import CaseStudyAgent

from src.models import EmployeeEvaluation

# 1. Choisir ton agent ici
#agent = FormationAgent()
# agent = BestPracticesAgent()
agent = CaseStudyAgent()

# 2. Simuler une évaluation RH
evaluation = EmployeeEvaluation(
    employe="Julie Lambert",
    evaluation="Julie a du mal à gérer son temps et à bien s'exprimer en réunion.",
    score=62
)

# 3. Récupération des documents similaires
documents = agent.retrieve(evaluation.evaluation)

# 4. Génération d'une recommandation partielle
recommandation = agent.generate_partial_recommendation(
    evaluation_text=evaluation.evaluation,
    documents=documents
)

# 5. Affichage
print(f"\n===== RECOMMANDATION de {agent.__class__.__name__} =====")
print(recommandation)

