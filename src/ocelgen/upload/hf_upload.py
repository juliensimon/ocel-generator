"""Upload agent trace datasets to Hugging Face Hub."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

from ocelgen.export.manifest import build_manifest
from ocelgen.export.normative import template_to_dict
from ocelgen.export.ocel_json import ocel_log_to_dict
from ocelgen.generation.engine import GenerationResult
from ocelgen.models.ocel import OcelLog
from ocelgen.models.workflow import WorkflowTemplate
from ocelgen.scenarios.domain import DomainScenario
from ocelgen.upload.readme import generate_dataset_card


def build_repo_name(namespace: str, domain_name: str) -> str:
    """Build the HF repo name for a domain."""
    return f"{namespace}/agent-traces-{domain_name}"


def prepare_upload_files(
    rows: list[dict],
    log: OcelLog,
    template: WorkflowTemplate,
    result: GenerationResult,
    scenario: DomainScenario,
    namespace: str,
    output_dir: Path,
    seed: int | None = None,
) -> list[Path]:
    """Write all files to output_dir, ready for upload. Returns list of file paths."""
    files: list[Path] = []

    # Parquet
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = data_dir / "train.parquet"
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, parquet_path)
    files.append(parquet_path)

    # OCEL files
    ocel_dir = output_dir / "ocel"
    ocel_dir.mkdir(parents=True, exist_ok=True)

    ocel_path = ocel_dir / "output.jsonocel"
    with open(ocel_path, "w", encoding="utf-8") as f:
        json.dump(ocel_log_to_dict(log), f, indent=2, ensure_ascii=False)
    files.append(ocel_path)

    normative_path = ocel_dir / "normative_model.json"
    with open(normative_path, "w", encoding="utf-8") as f:
        json.dump(template_to_dict(template), f, indent=2, ensure_ascii=False)
    files.append(normative_path)

    manifest_path = ocel_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(build_manifest(result, seed=seed), f, indent=2, ensure_ascii=False)
    files.append(manifest_path)

    # README
    readme_path = output_dir / "README.md"
    card = generate_dataset_card(
        scenario=scenario,
        namespace=namespace,
        num_events=len(rows),
        num_objects=len(log.objects),
    )
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(card)
    files.append(readme_path)

    return files


def upload_to_hub(
    output_dir: Path,
    namespace: str,
    domain_name: str,
) -> str:
    """Upload prepared files to HF Hub. Returns the repo URL."""
    api = HfApi()
    repo_id = build_repo_name(namespace, domain_name)

    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="dataset",
    )

    return f"https://huggingface.co/datasets/{repo_id}"


def create_or_update_collection(
    namespace: str,
    collection_slug: str,
    repo_ids: list[str],
) -> str:
    """Create or update an HF collection with the given dataset repos."""
    api = HfApi()

    try:
        collections = api.list_collections(owner=namespace)
        existing = None
        for col in collections:
            if col.slug and collection_slug in col.slug:
                existing = col
                break
    except Exception:
        existing = None

    if existing is None:
        collection = api.create_collection(
            title="Open Agent Traces",
            namespace=namespace,
            description=(
                "A collection of 10 LLM-enriched synthetic agent trace datasets "
                "covering diverse domains and workflow patterns. "
                "Generated with ocelgen using OCEL 2.0 standard."
            ),
            exists_ok=True,
        )
        collection_slug_full = collection.slug
    else:
        collection_slug_full = existing.slug

    for repo_id in repo_ids:
        try:
            api.add_collection_item(
                collection_slug=collection_slug_full,
                item_id=repo_id,
                item_type="dataset",
                exists_ok=True,
            )
        except Exception:
            pass

    return f"https://huggingface.co/collections/{collection_slug_full}"
