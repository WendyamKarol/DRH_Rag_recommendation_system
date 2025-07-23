import argparse
import logging
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler
from config.config import config
from src.rag_system import TrainingRAGSystem
from src.models import EmployeeEvaluation
from datetime import datetime

console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

REPORT_DIR = "reports"


def display_header():
    console.print("\n[bold cyan]📊 Initialisation du système...[/bold cyan]\n")


def run_single(name: str, system: TrainingRAGSystem):
    """Génère un rapport pour un employé à partir de son nom uniquement"""
    evals = load_employee_evaluations()
    employee = next((e for e in evals if e.employe == name), None)

    if not employee:
        console.print(f"[red]❌ Aucun employé trouvé avec le nom '{name}'[/red]")
        return

    recommendation = system.process_evaluation(employee)
    report = format_recommendation_report(recommendation)

    Path(REPORT_DIR).mkdir(exist_ok=True)
    filename = f"{employee.employe.replace(' ', '_')}.txt"
    with open(Path(REPORT_DIR) / filename, "w", encoding="utf-8") as f:
        f.write(report)

    console.print(f"[green]✅ Rapport généré : {filename}[/green]")


def run_batch(system: TrainingRAGSystem):
    """Génère des rapports pour tous les employés"""
    evals = load_employee_evaluations()
    Path(REPORT_DIR).mkdir(exist_ok=True)

    for e in evals:
        recommendation = system.process_evaluation(e)
        report = format_recommendation_report(recommendation)
        filename = f"{e.employe.replace(' ', '_')}.txt"
        with open(Path(REPORT_DIR) / filename, "w", encoding="utf-8") as f:
            f.write(report)
        console.print(f"[green]✅ {e.employe} → rapport : {filename}[/green]")


def load_employee_evaluations():
    import json
    with open(config.data.employees_file, encoding="utf-8") as f:
        data = json.load(f)
    return [
        EmployeeEvaluation(
            employe=entry["employe"],
            evaluation=entry["evaluation"],
            score=entry["score"]
        )
        for entry in data
    ]


def format_recommendation_report(rec) -> str:
    formatted = [
        "=" * 70,
        "RAPPORT DE RECOMMANDATIONS DE FORMATION",
        "=" * 70,
        "",
        f"Employé: {rec.employe}",
        f"Date: {rec.generated_at.strftime('%d/%m/%Y')}",
        f"Score actuel: {rec.score}/100",
        f"Niveau de priorité: {rec.niveau_priorite}",
        "",
        "ÉVALUATION INITIALE:",
        "-" * 50,
        rec.evaluation_originale.strip(),
        "",
        "COMPÉTENCES À DÉVELOPPER:",
        "-" * 50,
        ", ".join(gap.competency.replace("_", " ").title() for gap in rec.competency_gaps) or "Aucune lacune spécifique détectée",
        "",
        "FORMATIONS RECOMMANDÉES:",
        "-" * 50,
    ]

    for i, f in enumerate(rec.formations_recommandees, 1):
        formatted += [
            f"\n{i}. {f.get('titre', 'Formation')}",
            f"   Type: {f.get('type', 'Non spécifié')}",
            f"   Durée: {f.get('duree', 'À déterminer')}",
            f"   Pertinence: {f.get('pertinence', '')}"
        ]
        if f.get("objectifs"):
            formatted.append("   Objectifs:")
            formatted += [f"   • {obj}" for obj in f["objectifs"]]

    formatted += [
        "",
        "PLAN DE DÉVELOPPEMENT PERSONNALISÉ:",
        "-" * 50,
        rec.plan_developpement.strip(),
        "",
        "=" * 70
    ]

    return "\n".join(formatted)


def main():
    display_header()
    system = TrainingRAGSystem()
    system.load_training_corpus(config.data.training_corpus_file)

    parser = argparse.ArgumentParser(description="Générer des recommandations de formation.")
    subparsers = parser.add_subparsers(dest="mode")

    sub_single = subparsers.add_parser("single", help="Générer un rapport pour un employé")
    sub_single.add_argument("--name", required=True, help="Nom exact de l'employé")

    sub_batch = subparsers.add_parser("batch", help="Générer un rapport pour tous les employés")

    args = parser.parse_args()

    if args.mode == "single":
        run_single(args.name, system)
    elif args.mode == "batch":
        run_batch(system)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
