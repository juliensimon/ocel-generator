"""Upload agent trace datasets to Hugging Face Hub as a single multi-config dataset."""

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


def build_repo_id(namespace: str) -> str:
    """Build the HF repo ID for the unified dataset."""
    return f"{namespace}/open-agent-traces"


# Keep old name for backward compatibility with tests
def build_repo_name(namespace: str, domain_name: str) -> str:
    """Build repo name (deprecated — use build_repo_id for unified repo)."""
    return f"{namespace}/agent-traces-{domain_name}"


def prepare_domain_files(
    rows: list[dict],
    log: OcelLog,
    template: WorkflowTemplate,
    result: GenerationResult,
    scenario: DomainScenario,
    output_dir: Path,
    seed: int | None = None,
) -> list[Path]:
    """Write files for a single domain config into the unified dataset structure.

    Layout:
      output_dir/data/{domain}/train.parquet
      output_dir/ocel/{domain}/output.jsonocel
      output_dir/ocel/{domain}/normative_model.json
      output_dir/ocel/{domain}/manifest.json
    """
    domain = scenario.name
    files: list[Path] = []

    # Parquet — HF auto-detects configs from data/{config_name}/ dirs
    data_dir = output_dir / "data" / domain
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = data_dir / "train.parquet"
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, parquet_path)
    files.append(parquet_path)

    # OCEL files per domain
    ocel_dir = output_dir / "ocel" / domain
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

    return files


# Keep old function for backward compat with existing tests
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
    """Write all files to output_dir (legacy per-domain layout). Returns list of file paths."""
    files: list[Path] = []

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = data_dir / "train.parquet"
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, parquet_path)
    files.append(parquet_path)

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

    from ocelgen.upload.readme import generate_dataset_card

    readme_path = output_dir / "README.md"
    card = generate_dataset_card(
        scenarios=[scenario],
        namespace=namespace,
        domain_stats={scenario.name: {"num_events": len(rows), "num_objects": len(log.objects)}},
    )
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(card)
    files.append(readme_path)

    return files


def upload_unified_dataset(
    output_dir: Path,
    namespace: str,
) -> str:
    """Upload the unified multi-config dataset to HF Hub. Returns the repo URL."""
    api = HfApi()
    repo_id = build_repo_id(namespace)

    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="dataset",
    )

    return f"https://huggingface.co/datasets/{repo_id}"


# Keep old function for backward compat
def upload_to_hub(
    output_dir: Path,
    namespace: str,
    domain_name: str,
) -> str:
    """Upload prepared files to HF Hub (legacy per-domain). Returns the repo URL."""
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
    import logging

    logger = logging.getLogger(__name__)
    api = HfApi()

    try:
        collections = api.list_collections(owner=namespace)
        existing = None
        for col in collections:
            if col.slug and collection_slug in col.slug:
                existing = col
                break
    except Exception as exc:
        logger.warning("Failed to list collections for '%s': %s", namespace, exc)
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

    failed_items: list[str] = []
    for repo_id in repo_ids:
        try:
            api.add_collection_item(
                collection_slug=collection_slug_full,
                item_id=repo_id,
                item_type="dataset",
                exists_ok=True,
            )
        except Exception as exc:
            logger.warning("Failed to add '%s' to collection: %s", repo_id, exc)
            failed_items.append(repo_id)

    if failed_items:
        logger.warning("%d/%d items failed to add to collection", len(failed_items), len(repo_ids))

    return f"https://huggingface.co/collections/{collection_slug_full}"
