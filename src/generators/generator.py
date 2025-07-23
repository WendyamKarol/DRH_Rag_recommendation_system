import json
from datetime import datetime
from openai import OpenAI
from src.generators.base import BaseGenerator, GenerationContext
from src.models import TrainingRecommendation, RetrievedDocument
from config.config import config


class Generator(BaseGenerator):
    def __init__(self):
        self.client = OpenAI(api_key=config.openai_api_key)
        self.model = config.model.openai_model
        self.temperature = config.model.temperature
        self.max_tokens = config.model.max_tokens

    def generate(self, context: GenerationContext) -> TrainingRecommendation:
        prompt = self._build_prompt(context)
        response = self._call_openai(prompt)
        data = self._parse_response(response)

        # Envelopper les documents dans des RetrievedDocument (simples)
        retrieved_docs = [
            RetrievedDocument(document=doc, similarity_score=1.0, relevance_rank=i+1)
            for i, doc in enumerate(context.retrieved_documents)
        ]

        return TrainingRecommendation(
            employe=context.employee_evaluation.employe,
            evaluation_originale=context.employee_evaluation.evaluation,
            score=context.employee_evaluation.score,
            niveau_priorite=context.priority_level,
            competency_gaps=context.competency_gaps,
            retrieved_documents=retrieved_docs,
            plan_developpement=data.get("plan", ""),
            formations_recommandees=data.get("formations", []),
            generated_at=datetime.now()
        )

    def _build_prompt(self, context: GenerationContext) -> str:
        doc_snippets = "\n".join([
            f"- {doc.content[:200]}" for doc in context.retrieved_documents
        ])

        return f"""
Tu es un expert RH. Analyse cette évaluation et propose un plan d'action.

Évaluation : {context.employee_evaluation.evaluation}
Score : {context.employee_evaluation.score}
Lacunes : {[gap.competency for gap in context.competency_gaps]}
Documents pertinents :
{doc_snippets}

Réponds uniquement avec ce format JSON :
{{
  "plan": "Résumé clair du plan de développement",
  "formations": [
    {{
      "titre": "Nom de la formation",
      "type": "Programme de formation",
      "duree": "X heures",
      "pertinence": "Pourquoi c’est utile"
    }}
  ]
}}
"""

    def _call_openai(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Tu es un assistant RH"},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content

    def _parse_response(self, response_text: str) -> dict:
        try:
            cleaned = response_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
            return json.loads(cleaned)
        except Exception:
            return {
                "plan": "Plan générique basé sur les lacunes",
                "formations": []
            }
