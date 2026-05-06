"""Unified CLI for the RAG Evaluation Suite.

Entry point for all evaluation operations: run evaluations, A/B experiments,
generate test sets, compare runs, and launch the dashboard.
"""

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from src.logging import get_logger, setup_logging

console = Console()
logger = get_logger("cli")


@click.group()
def cli():
    """RAG Evaluation Suite — multi-layer evaluation for RAG pipelines."""
    pass


@cli.command()
@click.option("--test-set", required=True, help="Path to test set file (JSON/JSONL/CSV)")
@click.option("--mode", type=click.Choice(["static", "pipeline"]), default="static",
              help="'static' uses test set contexts, 'pipeline' queries live RAG")
@click.option("--metrics", default=None,
              help="Comma-separated metrics (e.g., 'retrieval,faithfulness,relevancy'). Default: all")
@click.option("--output-dir", default=None, help="Directory to save results")
@click.option("--format", "output_format", type=click.Choice(["json", "html", "both"]),
              default="json", help="Output format")
@click.option("--client-type", default=None,
              help="RAG client type: 'rag2', 'http', or a fully-qualified class path. "
                   "Overrides config. Only used in pipeline mode.")
def eval(test_set: str, mode: str, metrics: str, output_dir: str, output_format: str,
         client_type: str):
    """Run evaluation on a test set."""
    from src.pipeline.evaluator import Evaluator
    from src.reporting.reporter import save_html_report, save_json_report

    metrics_list = metrics.split(",") if metrics else None

    console.print(f"[bold blue]Running evaluation[/bold blue]")

    logger.info("Starting evaluation", extra={
        "test_set": test_set,
        "mode": mode,
        "metrics": metrics_list or "default",
        "client_type": client_type or "config-default",
    })
    console.print(f"  Test set: {test_set}")
    console.print(f"  Mode: {mode}")
    console.print(f"  Metrics: {metrics_list or 'all defaults'}")
    if client_type:
        console.print(f"  Client: {client_type}")

    # Create RAG client via factory if pipeline mode and client_type specified
    rag_client = None
    if mode == "pipeline" and client_type:
        from src.pipeline.client_factory import create_rag_client
        rag_client = create_rag_client(client_type=client_type)

    evaluator = Evaluator(rag_client=rag_client)
    summary = evaluator.run(
        test_set_path=test_set,
        mode=mode,
        metrics=metrics_list,
    )

    # Save reports
    if output_format in ("json", "both"):
        path = save_json_report(summary, output_dir=output_dir)
        console.print(f"  [green]JSON saved:[/green] {path}")
        logger.info("Evaluation JSON report saved", extra={"path": str(path)})

    if output_format in ("html", "both"):
        path = save_html_report(summary, output_dir=output_dir)
        console.print(f"  [green]HTML saved:[/green] {path}")
        logger.info("Evaluation HTML report saved", extra={"path": str(path)})

    # Print summary table
    _print_summary(summary.aggregated_scores, summary.num_cases, summary.total_latency_ms)


@cli.command()
@click.option("--test-set", required=True, help="Path to test set file")
@click.option("--config-a", required=True, help="JSON string of config overrides for variant A")
@click.option("--config-b", required=True, help="JSON string of config overrides for variant B")
@click.option("--label-a", default="Config A", help="Label for variant A")
@click.option("--label-b", default="Config B", help="Label for variant B")
@click.option("--output-dir", default=None, help="Directory to save results")
@click.option("--client-type", default=None,
              help="RAG client type: 'rag2', 'http', or a fully-qualified class path.")
def experiment(
    test_set: str,
    config_a: str,
    config_b: str,
    label_a: str,
    label_b: str,
    output_dir: str,
    client_type: str,
):
    """Run A/B experiment comparing two RAG configurations."""
    from src.pipeline.experiment import ExperimentRunner
    from src.reporting.reporter import save_experiment_report

    try:
        config_a_dict = json.loads(config_a)
        config_b_dict = json.loads(config_b)
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON config:[/red] {e}")
        sys.exit(1)

    console.print(f"[bold blue]Running A/B experiment[/bold blue]")
    console.print(f"  {label_a}: {config_a_dict}")
    console.print(f"  {label_b}: {config_b_dict}")
    if client_type:
        console.print(f"  Client: {client_type}")

    logger.info("Starting A/B experiment", extra={
        "test_set": test_set,
        "config_a": config_a_dict,
        "config_b": config_b_dict,
        "label_a": label_a,
        "label_b": label_b,
        "client_type": client_type or "config-default",
    })

    # Create RAG client via factory if client_type specified
    rag_client = None
    if client_type:
        from src.pipeline.client_factory import create_rag_client
        rag_client = create_rag_client(client_type=client_type)

    runner = ExperimentRunner(rag_client=rag_client)
    result = runner.run(
        test_set_path=test_set,
        config_a=config_a_dict,
        config_b=config_b_dict,
        label_a=label_a,
        label_b=label_b,
    )

    # Save result
    path = save_experiment_report(result.model_dump(mode="json"), output_dir=output_dir)
    console.print(f"  [green]Result saved:[/green] {path}")
    logger.info("Experiment result saved", extra={"path": str(path)})

    # Print summary
    _print_experiment_summary(result)


@cli.command("generate-testset")
@click.option("--docs-path", required=True, help="Path to documents directory")
@click.option("--num-questions", default=3, help="Questions per document")
@click.option("--output", default="data/test_sets/generated.json", help="Output file path")
def generate_testset(docs_path: str, num_questions: int, output: str):
    """Generate synthetic test set from documents."""
    from src.datasets.generator import TestSetGenerator

    console.print(f"[bold blue]Generating test set[/bold blue]")
    console.print(f"  Documents: {docs_path}")
    console.print(f"  Questions per doc: {num_questions}")

    generator = TestSetGenerator()
    cases = generator.generate_from_directory(
        docs_path=docs_path,
        num_questions_per_doc=num_questions,
    )

    generator.save_test_set(cases, output)
    console.print(f"  [green]Generated {len(cases)} test cases → {output}[/green]")


@cli.command()
@click.option("--run-a", required=True, help="Path to first result JSON")
@click.option("--run-b", required=True, help="Path to second result JSON")
@click.option("--label-a", default="Baseline", help="Label for first run")
@click.option("--label-b", default="Current", help="Label for second run")
def compare(run_a: str, run_b: str, label_a: str, label_b: str):
    """Compare two evaluation runs."""
    from src.reporting.comparator import compare_files, format_comparison_table

    console.print(f"[bold blue]Comparing runs[/bold blue]")
    result = compare_files(run_a, run_b, label_a=label_a, label_b=label_b)

    # Print comparison table
    console.print(format_comparison_table(result))

    console.print(f"\n[green]Improved:[/green] {result['num_improved']} | "
                  f"[red]Regressed:[/red] {result['num_regressed']} | "
                  f"Stable: {result['num_stable']}")


@cli.command()
@click.option("--run-id", default="latest", help="Run ID or 'latest'")
def report(run_id: str):
    """View an evaluation report."""
    results_dir = Path(__file__).parent / "data" / "results"
    if not results_dir.exists():
        console.print("[red]No results directory found.[/red]")
        return

    json_files = sorted(results_dir.glob("eval_run_*.json"), reverse=True)
    if not json_files:
        console.print("[red]No evaluation results found.[/red]")
        return

    if run_id == "latest":
        file_path = json_files[0]
    else:
        matches = [f for f in json_files if run_id in f.name]
        if not matches:
            console.print(f"[red]No result matching '{run_id}'[/red]")
            return
        file_path = matches[0]

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    console.print(f"[bold]Report: {file_path.name}[/bold]\n")
    _print_summary(
        data.get("aggregated_scores", {}),
        data.get("num_cases", 0),
        data.get("total_latency_ms", 0),
    )


@cli.command()
@click.option("--port", default=8501, help="Dashboard port")
def dashboard(port: int):
    """Launch Streamlit evaluation dashboard."""
    import subprocess
    dashboard_path = Path(__file__).parent / "src" / "reporting" / "dashboard.py"
    console.print(f"[bold blue]Launching dashboard on port {port}[/bold blue]")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port", str(port),
    ])


def _print_summary(scores: dict, num_cases: int, total_ms: float) -> None:
    """Print a rich summary table of evaluation scores.

    Args:
        scores: Aggregated metric scores.
        num_cases: Number of test cases.
        total_ms: Total evaluation time in milliseconds.
    """
    table = Table(title=f"Evaluation Results ({num_cases} cases, {total_ms:.0f}ms)")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right", style="bold")
    table.add_column("Bar", min_width=20)

    for metric, score in sorted(scores.items()):
        # Normalize to 0-20 char bar
        if score > 1.0:
            bar_len = int(score / 5.0 * 20)
        else:
            bar_len = int(score * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        table.add_row(metric, f"{score:.4f}", bar)

    console.print(table)


def _print_experiment_summary(result) -> None:
    """Print A/B experiment summary.

    Args:
        result: ExperimentResult object.
    """
    table = Table(title=f"A/B Experiment ({result.num_cases} cases)")
    table.add_column("", style="bold")
    table.add_column("Config A", justify="center")
    table.add_column("Config B", justify="center")

    table.add_row("Mean Score",
                  f"{result.scores_a.get('mean', 0):.4f}",
                  f"{result.scores_b.get('mean', 0):.4f}")
    table.add_row("Win Count",
                  str(result.scores_a.get("win_count", 0)),
                  str(result.scores_b.get("win_count", 0)))
    table.add_row("Win Rate",
                  f"{result.scores_a.get('win_rate', 0):.1%}",
                  f"{result.scores_b.get('win_rate', 0):.1%}")

    console.print(table)

    stat = result.statistical_tests
    p_val = stat.get("p_value", 1.0)
    sig = stat.get("significant", False)
    console.print(f"\n  Statistical test: {stat.get('test', 'N/A')} | "
                  f"p-value: {p_val:.6f} | "
                  f"Significant: {'[green]YES[/green]' if sig else '[yellow]NO[/yellow]'}")
    console.print(f"  [bold]Winner: {result.winner or 'tie'}[/bold]")


if __name__ == "__main__":
    cli()
