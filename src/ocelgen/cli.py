"""Typer CLI for ocelgen — generate, validate, and inspect OCEL 2.0 event logs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from ocelgen.export.manifest import write_manifest
from ocelgen.export.normative import write_normative_model
from ocelgen.export.ocel_json import write_ocel_json
from ocelgen.generation.engine import PATTERN_REGISTRY, generate
from ocelgen.validation.schema import validate_ocel_file

app = typer.Typer(
    name="ocelgen",
    help="Mock OCEL 2.0 event log generator for LangChain multi-agent runs.",
    no_args_is_help=True,
)
console = Console()


@app.command("generate")
def generate_cmd(
    pattern: Annotated[
        str, typer.Option("-p", "--pattern", help="Workflow pattern")
    ] = "sequential",
    runs: Annotated[
        int, typer.Option("-n", "--runs", help="Number of runs")
    ] = 100,
    noise: Annotated[
        float, typer.Option("-N", "--noise", help="Noise rate (0.0–1.0)")
    ] = 0.2,
    max_deviations: Annotated[
        int, typer.Option("--max-deviations", help="Max deviations per run")
    ] = 3,
    seed: Annotated[
        int | None, typer.Option("--seed", help="Random seed")
    ] = None,
    output: Annotated[
        Path, typer.Option("-o", "--output", help="Output .jsonocel path")
    ] = Path("output.jsonocel"),
) -> None:
    """Generate mock OCEL 2.0 event logs."""
    if pattern not in PATTERN_REGISTRY:
        available = list(PATTERN_REGISTRY.keys())
        console.print(f"[red]Unknown pattern '{pattern}'. Available: {available}[/red]")
        raise typer.Exit(1)

    console.print(f"Generating [bold]{runs}[/bold] {pattern} runs (noise={noise}, seed={seed})...")

    result = generate(
        pattern_name=pattern,
        num_runs=runs,
        noise_rate=noise,
        max_deviations_per_run=max_deviations,
        seed=seed,
    )

    # Write three output files
    ocel_path = output
    normative_path = output.parent / "normative_model.json"
    manifest_path = output.parent / "manifest.json"

    write_ocel_json(result.log, ocel_path)
    write_normative_model(result.template, normative_path)
    write_manifest(result, manifest_path, seed=seed)

    # Summary
    console.print()
    table = Table(title="Generation Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Pattern", result.template.name)
    table.add_row("Total runs", str(result.total_runs))
    table.add_row("Conformant runs", str(result.conformant_runs))
    table.add_row("Deviant runs", str(result.deviant_runs))
    table.add_row("Total events", str(len(result.log.events)))
    table.add_row("Total objects", str(len(result.log.objects)))
    table.add_row("Total deviations", str(len(result.deviations)))
    console.print(table)

    console.print()
    console.print(f"[green]OCEL log:[/green]        {ocel_path}")
    console.print(f"[green]Normative model:[/green] {normative_path}")
    console.print(f"[green]Manifest:[/green]        {manifest_path}")


@app.command()
def validate(
    path: Annotated[Path, typer.Argument(help="Path to .jsonocel file")],
) -> None:
    """Validate an OCEL 2.0 JSON file against the official schema."""
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        raise typer.Exit(1)

    console.print(f"Validating [bold]{path}[/bold]...")
    errors = validate_ocel_file(path)

    if errors:
        console.print(f"[red]Found {len(errors)} validation error(s):[/red]")
        for err in errors:
            console.print(f"  - {err}")
        raise typer.Exit(1)
    else:
        console.print("[green]Valid OCEL 2.0 JSON.[/green]")


@app.command("list-patterns")
def list_patterns() -> None:
    """List available workflow patterns."""
    table = Table(title="Available Patterns")
    table.add_column("Name", style="bold")
    table.add_column("Description")
    table.add_column("Steps", justify="right")

    for name, cls in PATTERN_REGISTRY.items():
        pattern = cls()
        template = pattern.build_template()
        table.add_row(name, pattern.description, str(len(template.steps)))

    console.print(table)
