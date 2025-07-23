
from typing import List
from src.agents.formation_agent import FormationAgent
from src.agents.best_practices_agent import BestPracticesAgent
from src.agents.case_study_agent import CaseStudyAgent
from src.models import EmployeeEvaluation
from src.generators.generator import Generator
from config.config import config



class MultiAgentRAG:
    def __init__(self):
        self.agents = [
            FormationAgent(),
            BestPracticesAgent(),
            CaseStudyAgent()
        ]
        self.generator = Generator()

    def generate_final_report(self, evaluation: EmployeeEvaluation) -> str:
        partials = []

        for agent in self.agents:
            docs = agent.retrieve(evaluation.evaluation)
            partial = agent.generate_partial_recommendation(
                evaluation_text=evaluation.evaluation,
                documents=docs
            )
            partials.append(partial.strip())

        combined_prompt = self._build_fusion_prompt(evaluation, partials)
        return self.generator._call_openai(combined_prompt)

    def _build_fusion_prompt(self, evaluation: EmployeeEvaluation, partials: List[str]) -> str:
        partial_section = "\n\n".join([f"- {p}" for p in partials])
        return f"""
Tu es un expert RH senior.

Voici une évaluation d'employé :
"{evaluation.evaluation}"

Voici des recommandations générées par trois spécialistes :
{partial_section}

Fusionne ces recommandations en un **rapport global clair et structuré** (6-10 lignes max). Mets en évidence :
- Les points prioritaires
- Le plan d’action RH
- Les conseils personnalisés

Commence directement sans dire "Voici le rapport" ni numéroter les sections.
"""
