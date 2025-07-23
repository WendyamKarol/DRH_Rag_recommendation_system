from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
from src.models import EmployeeEvaluation, CompetencyGap, TrainingDocument


@dataclass
class RetrievalContext:
    employee_evaluation: EmployeeEvaluation
    competency_gaps: List[CompetencyGap]


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, context: RetrievalContext, k: int) -> List[TrainingDocument]:
        pass

    @abstractmethod
    def build_query(self, context: RetrievalContext) -> str:
        pass
