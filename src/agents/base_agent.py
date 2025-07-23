from abc import ABC, abstractmethod
from typing import List
from src.models import TrainingDocument


class AgentRAG(ABC):
    @abstractmethod
    def retrieve(self,query : str)->List[TrainingDocument]:
        pass

    @abstractmethod
    def generate_partial_recommendation(self,Evaluation_text:str,documents:List[TrainingDocument])->str:
        pass
