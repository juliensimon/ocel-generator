"""Typer CLI for ocelgen — generate, validate, and inspect OCEL 2.0 event logs."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from ocelgen.export.manifest import write_manifest
from ocelgen.export.normative import write_normative_model
from ocelgen.export.ocel_json import write_ocel_json
from ocelgen.generation.engine import PATTERN_REGISTRY, generate
from ocelgen.scenarios.domain import DomainScenario
from ocelgen.validation.schema import validate_ocel_file


def _get_registry(config: Path | None) -> dict[str, DomainScenario]:
    """Build the scenario registry, merging custom YAML config if provided."""
    from ocelgen.scenarios.loader import build_registry

    return build_registry(config)


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
    runs: Annotated[int, typer.Option("-n", "--runs", help="Number of runs")] = 100,
    noise: Annotated[float, typer.Option("-N", "--noise", help="Noise rate (0.0–1.0)")] = 0.2,
    max_deviations: Annotated[
        int, typer.Option("--max-deviations", help="Max deviations per run")
    ] = 3,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed")] = None,
    output: Annotated[Path, typer.Option("-o", "--output", help="Output .jsonocel path")] = Path(
        "output.jsonocel"
    ),
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


@app.command("list-domains")
def list_domains(
    config: Annotated[
        Path | None,
        typer.Option(
            "--config", "-c", help="YAML file or directory with custom domain definitions"
        ),
    ] = None,
) -> None:
    """List available domain scenarios for enriched generation."""
    registry = _get_registry(config)

    table = Table(title="Available Domain Scenarios")
    table.add_column("Name", style="bold", no_wrap=True)
    table.add_column("Pattern")
    table.add_column("Runs", justify="right")
    table.add_column("Noise", justify="right")
    table.add_column("Description")

    for name, scenario in registry.items():
        table.add_row(
            name,
            scenario.pattern,
            str(scenario.runs),
            f"{scenario.noise:.0%}",
            scenario.description,
        )

    console.print(table)
    console.print(f"\n[bold]{len(registry)}[/bold] domains available.")


@app.command("enrich")
def enrich_cmd(
    path: Annotated[Path, typer.Argument(help="Path to .jsonocel file to enrich")],
    domain: Annotated[str, typer.Option("--domain", "-d", help="Domain scenario name")] = "",
    model: Annotated[
        str, typer.Option("--model", "-m", help="LLM model for enrichment")
    ] = "google/gemini-2.0-flash-001",
    base_url: Annotated[
        str | None,
        typer.Option(
            "--base-url", help="OpenAI-compatible API base URL (e.g. http://localhost:8080/v1)"
        ),
    ] = None,
    output: Annotated[Path | None, typer.Option("-o", "--output", help="Output path")] = None,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config", "-c", help="YAML file or directory with custom domain definitions"
        ),
    ] = None,
) -> None:
    """Enrich an OCEL 2.0 trace with LLM-generated content."""
    from pydantic import TypeAdapter
    from rich.progress import Progress

    from ocelgen.enrichment.client import LLMClient
    from ocelgen.enrichment.enricher import enrich_log
    from ocelgen.export.ocel_json import ocel_log_to_dict
    from ocelgen.models.ocel import OcelLog
    from ocelgen.scenarios.registry import get_scenario

    registry = _get_registry(config)

    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        raise typer.Exit(1)

    if domain not in registry:
        console.print(
            f"[red]Unknown domain '{domain}'. Use 'list-domains' to see available domains.[/red]"
        )
        raise typer.Exit(1)

    scenario = get_scenario(domain, registry=registry)

    console.print(f"Loading [bold]{path}[/bold]...")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    log = TypeAdapter(OcelLog).validate_python(data)

    console.print(f"Enriching with [bold]{model}[/bold] for domain [bold]{domain}[/bold]...")
    kwargs: dict[str, str] = {"model": model}
    if base_url:
        kwargs["base_url"] = base_url
    client = LLMClient(**kwargs)

    with Progress() as progress:
        run_count = len([o for o in log.objects if o.type == "run"])
        task = progress.add_task("Enriching runs...", total=run_count)
        enrich_log(log, scenario, client=client, progress=progress, progress_task=task)

    out_path = output or path.parent / f"enriched-{path.name}"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ocel_log_to_dict(log), f, indent=2, ensure_ascii=False)

    console.print(f"[green]Enriched log written to:[/green] {out_path}")


@app.command("upload")
def upload_cmd(
    path: Annotated[Path, typer.Argument(help="Path to enriched .jsonocel file")],
    domain: Annotated[str, typer.Option("--domain", "-d", help="Domain scenario name")] = "",
    namespace: Annotated[str, typer.Option("--namespace", "-n", help="HF namespace")] = "",
    collection: Annotated[
        str, typer.Option("--collection", help="Collection slug")
    ] = "open-agent-traces",
    config: Annotated[
        Path | None,
        typer.Option(
            "--config", "-c", help="YAML file or directory with custom domain definitions"
        ),
    ] = None,
) -> None:
    """Upload an enriched trace to Hugging Face Hub."""
    from pydantic import TypeAdapter

    from ocelgen.models.ocel import OcelLog
    from ocelgen.scenarios.registry import get_scenario
    from ocelgen.upload.flatten import flatten_log
    from ocelgen.upload.hf_upload import build_repo_name, prepare_upload_files, upload_to_hub

    registry = _get_registry(config)

    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        raise typer.Exit(1)

    if domain not in registry:
        console.print(f"[red]Unknown domain '{domain}'.[/red]")
        raise typer.Exit(1)

    if not namespace:
        console.print("[red]--namespace is required.[/red]")
        raise typer.Exit(1)

    scenario = get_scenario(domain, registry=registry)

    console.print(f"Loading [bold]{path}[/bold]...")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    log = TypeAdapter(OcelLog).validate_python(data)

    console.print("Flattening to tabular format...")
    rows = flatten_log(log, domain=domain)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        from ocelgen.generation.engine import GenerationResult

        result = GenerationResult(
            log=log,
            template=PATTERN_REGISTRY[scenario.pattern]().build_template(),
            total_runs=scenario.runs,
            conformant_runs=scenario.runs,
            deviant_runs=0,
        )

        prepare_upload_files(
            rows=rows,
            log=log,
            template=result.template,
            result=result,
            scenario=scenario,
            namespace=namespace,
            output_dir=tmp_path,
        )

        console.print(f"Uploading to [bold]{build_repo_name(namespace, domain)}[/bold]...")
        url = upload_to_hub(tmp_path, namespace, domain)
        console.print(f"[green]Uploaded:[/green] {url}")


@app.command("pipeline")
def pipeline_cmd(
    domain: Annotated[
        str | None, typer.Option("--domain", "-d", help="Single domain to process")
    ] = None,
    all_domains: Annotated[bool, typer.Option("--all", help="Process all domains")] = False,
    namespace: Annotated[str, typer.Option("--namespace", "-n", help="HF namespace")] = "",
    model: Annotated[
        str, typer.Option("--model", "-m", help="LLM model for enrichment")
    ] = "google/gemini-2.0-flash-001",
    base_url: Annotated[
        str | None,
        typer.Option(
            "--base-url", help="OpenAI-compatible API base URL (e.g. http://localhost:8080/v1)"
        ),
    ] = None,
    collection: Annotated[
        str, typer.Option("--collection", help="Collection slug")
    ] = "open-agent-traces",
    skip_upload: Annotated[
        bool, typer.Option("--skip-upload", help="Generate and enrich but don't upload")
    ] = False,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config", "-c", help="YAML file or directory with custom domain definitions"
        ),
    ] = None,
) -> None:
    """End-to-end pipeline: generate, enrich, and upload agent trace datasets."""
    from rich.progress import Progress

    from ocelgen.enrichment.client import LLMClient
    from ocelgen.enrichment.enricher import enrich_log
    from ocelgen.scenarios.registry import get_scenario
    from ocelgen.upload.flatten import flatten_log
    from ocelgen.upload.hf_upload import (
        build_repo_name,
        create_or_update_collection,
        prepare_upload_files,
        upload_to_hub,
    )

    registry = _get_registry(config)

    if not namespace:
        console.print("[red]--namespace is required.[/red]")
        raise typer.Exit(1)

    if not domain and not all_domains:
        console.print("[red]Specify --domain <name> or --all.[/red]")
        raise typer.Exit(1)

    domains: list[str] = list(registry.keys()) if all_domains else [domain]  # type: ignore[list-item]

    for d in domains:
        if d not in registry:
            console.print(f"[red]Unknown domain '{d}'.[/red]")
            raise typer.Exit(1)

    kwargs: dict[str, str] = {"model": model}
    if base_url:
        kwargs["base_url"] = base_url
    client = LLMClient(**kwargs)
    uploaded_repos: list[str] = []

    for d in domains:
        scenario = get_scenario(d, registry=registry)
        console.rule(f"[bold]{scenario.name}[/bold] ({scenario.pattern})")

        console.print(f"Generating {scenario.runs} runs...")
        result = generate(
            pattern_name=scenario.pattern,
            num_runs=scenario.runs,
            noise_rate=scenario.noise,
            seed=scenario.seed,
        )
        console.print(
            f"  {result.total_runs} runs, {len(result.log.events)} events, "
            f"{result.deviant_runs} deviant"
        )

        console.print(f"Enriching with {model}...")
        with Progress() as progress:
            task = progress.add_task(f"Enriching {d}...", total=scenario.runs)
            enrich_log(result.log, scenario, client=client, progress=progress, progress_task=task)

        rows = flatten_log(result.log, domain=d)
        console.print(f"  Flattened to {len(rows)} rows")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            prepare_upload_files(
                rows=rows,
                log=result.log,
                template=result.template,
                result=result,
                scenario=scenario,
                namespace=namespace,
                output_dir=tmp_path,
                seed=scenario.seed,
            )

            if not skip_upload:
                repo_id = build_repo_name(namespace, d)
                console.print(f"Uploading to [bold]{repo_id}[/bold]...")
                url = upload_to_hub(tmp_path, namespace, d)
                uploaded_repos.append(repo_id)
                console.print(f"  [green]{url}[/green]")
            else:
                console.print("  [yellow]Skipping upload (--skip-upload)[/yellow]")

    if uploaded_repos and not skip_upload:
        console.rule("[bold]Collection[/bold]")
        console.print("Creating/updating collection...")
        col_url = create_or_update_collection(namespace, collection, uploaded_repos)
        console.print(f"[green]Collection:[/green] {col_url}")

    console.print("\n[bold green]Done![/bold green]")
