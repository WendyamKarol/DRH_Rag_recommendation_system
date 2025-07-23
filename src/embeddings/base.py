
from abc import ABC, abstractmethod
from typing import List
from src.models import TrainingDocument


class BaseEmbeddingStore(ABC):
    @abstractmethod
    def add_documents(self, documents: List[TrainingDocument]) -> None:
        pass

    @abstractmethod
    def search(self, query_text: str, k: int) -> List[TrainingDocument]:
        pass
 