from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
from src.models import EmployeeEvaluation, CompetencyGap, RetrievedDocument,TrainingDocument


@dataclass
class GenerationContext:
    employee_evaluation: EmployeeEvaluation
    competency_gaps: List[CompetencyGap]
    retrieved_documents: List[TrainingDocument]
    priority_level: str


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, context: GenerationContext):
        pass
