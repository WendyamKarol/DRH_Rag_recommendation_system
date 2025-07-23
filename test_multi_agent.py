from src.agents.multi_agent import MultiAgentRAG
from src.models import EmployeeEvaluation

evaluation = EmployeeEvaluation(
    employe="Julie Lambert",
    evaluation="Julie a des difficultés à s'exprimer en public et à organiser ses tâches efficacement.",
    score=63
)

agent = MultiAgentRAG()
rapport = agent.generate_final_report(evaluation)

print("\n===== RAPPORT GLOBAL =====")
print(rapport)
