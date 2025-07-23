from dataclasses import dataclass, field
from datetime import datetime
from typing import List


@dataclass
class EmployeeEvaluation:
    employe: str
    evaluation: str
    score: int


@dataclass
class TrainingDocument:
    content: str
    type: str
    source: str
    doc_id: str


@dataclass
class CompetencyGap:
    competency: str
    keywords_found: List[str] = field(default_factory=list)

@dataclass
class RetrievedDocument:
    document: TrainingDocument
    similarity_score: float
    relevance_rank: int


@dataclass
class TrainingRecommendation:
    employe: str
    evaluation_originale: str
    score: int
    niveau_priorite: str
    competency_gaps: List[CompetencyGap]
    retrieved_documents: List[RetrievedDocument]
    plan_developpement: str
    formations_recommandees: List[dict]
    generated_at: datetime = field(default_factory=datetime.now)


@dataclass
class BatchProcessingResult:
    total_processed: int
    successful: int = 0
    failed: int = 0
    recommendations: List[TrainingRecommendation] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    