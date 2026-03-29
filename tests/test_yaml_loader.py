"""Tests for YAML-based domain scenario loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from ocelgen.scenarios.domain import DomainScenario
from ocelgen.scenarios.loader import build_registry, load_domains_from_dir, load_domains_from_yaml

_VALID_DOMAIN_YAML = """\
domains:
  - name: "test-domain"
    description: "A test domain"
    pattern: "sequential"
    runs: 10
    noise: 0.1
    seed: 42
    user_queries:
      - "Query one"
      - "Query two"
    agent_personas:
      researcher: "You are a test researcher"
    tool_descriptions:
      web_search: "Search the web"
"""


class TestLoadDomainsFromYaml:
    def test_load_single_domain(self, tmp_path: Path) -> None:
        f = tmp_path / "domains.yaml"
        f.write_text(_VALID_DOMAIN_YAML)

        result = load_domains_from_yaml(f)

        assert len(result) == 1
        d = result["test-domain"]
        assert isinstance(d, DomainScenario)
        assert d.name == "test-domain"
        assert d.pattern == "sequential"
        assert d.runs == 10
        assert d.noise == 0.1
        assert d.seed == 42
        assert d.user_queries == ["Query one", "Query two"]
        assert d.agent_personas == {"researcher": "You are a test researcher"}
        assert d.tool_descriptions == {"web_search": "Search the web"}

    def test_load_multiple_domains(self, tmp_path: Path) -> None:
        f = tmp_path / "domains.yaml"
        f.write_text(
            """\
domains:
  - name: "domain-a"
    description: "A"
    pattern: "sequential"
    runs: 10
    noise: 0.1
    seed: 1
  - name: "domain-b"
    description: "B"
    pattern: "parallel"
    runs: 20
    noise: 0.2
    seed: 2
  - name: "domain-c"
    description: "C"
    pattern: "supervisor"
    runs: 30
    noise: 0.3
    seed: 3
"""
        )
        result = load_domains_from_yaml(f)
        assert len(result) == 3
        assert set(result.keys()) == {"domain-a", "domain-b", "domain-c"}

    def test_optional_fields_default_to_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "domains.yaml"
        f.write_text(
            """\
domains:
  - name: "minimal"
    description: "Minimal domain"
    pattern: "sequential"
    runs: 5
    noise: 0.0
    seed: 1
"""
        )
        result = load_domains_from_yaml(f)
        d = result["minimal"]
        assert d.user_queries == []
        assert d.agent_personas == {}
        assert d.tool_descriptions == {}

    def test_invalid_pattern_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "domains.yaml"
        f.write_text(
            """\
domains:
  - name: "bad"
    description: "Bad"
    pattern: "unknown"
    runs: 10
    noise: 0.1
    seed: 1
"""
        )
        with pytest.raises(ValueError, match="pattern"):
            load_domains_from_yaml(f)

    def test_missing_required_field_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "domains.yaml"
        f.write_text(
            """\
domains:
  - description: "No name"
    pattern: "sequential"
    runs: 10
    noise: 0.1
    seed: 1
"""
        )
        with pytest.raises(ValueError, match="missing required fields"):
            load_domains_from_yaml(f)

    def test_invalid_noise_range_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "domains.yaml"
        f.write_text(
            """\
domains:
  - name: "bad-noise"
    description: "Bad"
    pattern: "sequential"
    runs: 10
    noise: 1.5
    seed: 1
"""
        )
        with pytest.raises(ValueError, match="noise"):
            load_domains_from_yaml(f)

    def test_malformed_yaml_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "domains.yaml"
        f.write_text("domains:\n  - name: [invalid yaml structure\n")
        with pytest.raises(ValueError, match="Failed to parse YAML"):
            load_domains_from_yaml(f)

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "domains.yaml"
        f.write_text("")
        result = load_domains_from_yaml(f)
        assert result == {}

    def test_no_domains_key_returns_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "domains.yaml"
        f.write_text("other_key: value\n")
        result = load_domains_from_yaml(f)
        assert result == {}

    def test_negative_runs_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "domains.yaml"
        f.write_text(
            """\
domains:
  - name: "bad-runs"
    description: "Bad"
    pattern: "sequential"
    runs: -5
    noise: 0.1
    seed: 1
"""
        )
        with pytest.raises(ValueError, match="runs"):
            load_domains_from_yaml(f)


class TestLoadDomainsFromDir:
    def test_load_from_directory(self, tmp_path: Path) -> None:
        (tmp_path / "a.yaml").write_text(
            """\
domains:
  - name: "from-a"
    description: "A"
    pattern: "sequential"
    runs: 10
    noise: 0.1
    seed: 1
"""
        )
        (tmp_path / "b.yml").write_text(
            """\
domains:
  - name: "from-b"
    description: "B"
    pattern: "parallel"
    runs: 20
    noise: 0.2
    seed: 2
"""
        )
        result = load_domains_from_dir(tmp_path)
        assert len(result) == 2
        assert "from-a" in result
        assert "from-b" in result

    def test_later_file_overrides_earlier(self, tmp_path: Path) -> None:
        (tmp_path / "01.yaml").write_text(
            """\
domains:
  - name: "shared"
    description: "First"
    pattern: "sequential"
    runs: 10
    noise: 0.1
    seed: 1
"""
        )
        (tmp_path / "02.yaml").write_text(
            """\
domains:
  - name: "shared"
    description: "Second"
    pattern: "parallel"
    runs: 99
    noise: 0.5
    seed: 2
"""
        )
        result = load_domains_from_dir(tmp_path)
        assert len(result) == 1
        assert result["shared"].description == "Second"
        assert result["shared"].runs == 99


class TestBuildRegistry:
    def test_no_config_returns_builtins(self) -> None:
        result = build_registry(None)
        assert len(result) == 10
        assert "customer-support-triage" in result

    def test_merge_adds_new_domain(self, tmp_path: Path) -> None:
        f = tmp_path / "custom.yaml"
        f.write_text(_VALID_DOMAIN_YAML)
        result = build_registry(f)
        assert len(result) == 11
        assert "test-domain" in result
        assert "customer-support-triage" in result

    def test_override_builtin(self, tmp_path: Path) -> None:
        f = tmp_path / "override.yaml"
        f.write_text(
            """\
domains:
  - name: "customer-support-triage"
    description: "Custom override"
    pattern: "sequential"
    runs: 999
    noise: 0.05
    seed: 1001
"""
        )
        result = build_registry(f)
        assert len(result) == 10
        assert result["customer-support-triage"].runs == 999
        assert result["customer-support-triage"].description == "Custom override"

    def test_config_directory(self, tmp_path: Path) -> None:
        (tmp_path / "extra.yaml").write_text(_VALID_DOMAIN_YAML)
        result = build_registry(tmp_path)
        assert len(result) == 11

    def test_nonexistent_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            build_registry(tmp_path / "nonexistent")
