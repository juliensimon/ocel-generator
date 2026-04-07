"""Load domain scenarios from YAML configuration files."""

from __future__ import annotations

from pathlib import Path

import yaml

from ocelgen.scenarios.domain import DomainScenario

_VALID_PATTERNS = {"sequential", "supervisor", "parallel"}
_REQUIRED_FIELDS = {"name", "description", "pattern", "runs", "noise", "seed"}


def load_domains_from_yaml(path: Path) -> dict[str, DomainScenario]:
    """Read a YAML file and return validated domain scenarios keyed by name.

    The file must have a top-level ``domains`` key containing a list of
    domain definitions whose fields match :class:`DomainScenario`.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML in {path}: {exc}") from exc

    if not raw or "domains" not in raw:
        return {}

    domains_list = raw["domains"]
    if not isinstance(domains_list, list):
        raise ValueError(f"Expected 'domains' to be a list in {path}")

    result: dict[str, DomainScenario] = {}
    for idx, entry in enumerate(domains_list):
        if not isinstance(entry, dict):
            raise ValueError(f"Domain entry {idx} in {path} is not a mapping")

        missing = _REQUIRED_FIELDS - entry.keys()
        if missing:
            raise ValueError(
                f"Domain entry {idx} in {path} missing required fields: {sorted(missing)}"
            )

        name = entry["name"]
        pattern = entry["pattern"]
        if pattern not in _VALID_PATTERNS:
            raise ValueError(
                f"Domain '{name}' in {path}: 'pattern' must be one of "
                f"{sorted(_VALID_PATTERNS)} (got '{pattern}')"
            )

        runs = entry["runs"]
        if not isinstance(runs, int) or runs <= 0:
            raise ValueError(f"Domain '{name}' in {path}: 'runs' must be a positive integer")

        noise = entry["noise"]
        if not isinstance(noise, int | float) or not (0.0 <= noise <= 1.0):
            raise ValueError(f"Domain '{name}' in {path}: 'noise' must be between 0.0 and 1.0")

        seed = entry["seed"]
        if not isinstance(seed, int):
            raise ValueError(f"Domain '{name}' in {path}: 'seed' must be an integer")

        result[name] = DomainScenario(
            name=name,
            description=entry["description"],
            pattern=pattern,
            runs=runs,
            noise=float(noise),
            seed=seed,
            user_queries=entry.get("user_queries", []),
            agent_personas=entry.get("agent_personas", {}),
            tool_descriptions=entry.get("tool_descriptions", {}),
        )

    return result


def load_domains_from_dir(dir_path: Path) -> dict[str, DomainScenario]:
    """Load all ``*.yaml`` and ``*.yml`` files from a directory.

    Files are processed in alphabetical order; later files override
    earlier ones for domains with the same name.
    """
    result: dict[str, DomainScenario] = {}
    yaml_files = sorted(p for p in dir_path.iterdir() if p.suffix in {".yaml", ".yml"})
    for f in yaml_files:
        result.update(load_domains_from_yaml(f))
    return result


def build_registry(config: Path | None = None) -> dict[str, DomainScenario]:
    """Build a merged scenario registry.

    Starts with the 10 built-in scenarios and merges in any domains from
    the given *config* path (a single YAML file or a directory of them).
    Custom domains with the same name as a built-in override the built-in.
    """
    from ocelgen.scenarios.registry import SCENARIO_REGISTRY

    registry = dict(SCENARIO_REGISTRY)

    if config is None:
        return registry

    if config.is_file():
        registry.update(load_domains_from_yaml(config))
    elif config.is_dir():
        registry.update(load_domains_from_dir(config))
    else:
        raise ValueError(f"Config path does not exist: {config}")

    return registry
