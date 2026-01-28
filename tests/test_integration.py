"""End-to-end integration tests: engine → export → validate → manifest."""


from ocelgen.export.manifest import build_manifest
from ocelgen.export.normative import template_to_dict
from ocelgen.export.ocel_json import ocel_log_to_dict, write_ocel_json
from ocelgen.generation.engine import PATTERN_REGISTRY, generate
from ocelgen.validation.schema import validate_ocel_dict, validate_ocel_file


class TestEngine:
    def test_generate_sequential(self) -> None:
        result = generate("sequential", num_runs=5, noise_rate=0.3, seed=42)
        assert result.total_runs == 5
        assert result.conformant_runs + result.deviant_runs == 5
        assert len(result.log.events) > 0
        assert len(result.log.objects) > 0

    def test_generate_supervisor(self) -> None:
        result = generate("supervisor", num_runs=5, noise_rate=0.2, seed=42)
        assert result.total_runs == 5
        errors = validate_ocel_dict(ocel_log_to_dict(result.log))
        assert errors == []

    def test_generate_parallel(self) -> None:
        result = generate("parallel", num_runs=5, noise_rate=0.2, seed=42)
        assert result.total_runs == 5
        errors = validate_ocel_dict(ocel_log_to_dict(result.log))
        assert errors == []

    def test_unknown_pattern_raises(self) -> None:
        import pytest
        with pytest.raises(ValueError, match="Unknown pattern"):
            generate("nonexistent", num_runs=1)

    def test_deterministic_output(self) -> None:
        r1 = generate("sequential", num_runs=10, noise_rate=0.3, seed=42)
        r2 = generate("sequential", num_runs=10, noise_rate=0.3, seed=42)
        d1 = ocel_log_to_dict(r1.log)
        d2 = ocel_log_to_dict(r2.log)
        assert d1 == d2

    def test_different_seeds_differ(self) -> None:
        r1 = generate("sequential", num_runs=10, noise_rate=0.3, seed=1)
        r2 = generate("sequential", num_runs=10, noise_rate=0.3, seed=2)
        d1 = ocel_log_to_dict(r1.log)
        d2 = ocel_log_to_dict(r2.log)
        assert d1 != d2

    def test_zero_noise_all_conformant(self) -> None:
        result = generate("sequential", num_runs=20, noise_rate=0.0, seed=42)
        assert result.conformant_runs == 20
        assert result.deviant_runs == 0
        assert len(result.deviations) == 0

    def test_full_noise_all_deviant(self) -> None:
        result = generate("sequential", num_runs=20, noise_rate=1.0, seed=42)
        assert result.deviant_runs == 20
        assert result.conformant_runs == 0

    def test_schema_validation_at_scale(self) -> None:
        """Generate 50 runs with deviations and validate the merged output."""
        result = generate("sequential", num_runs=50, noise_rate=0.3, seed=42)
        errors = validate_ocel_dict(ocel_log_to_dict(result.log))
        assert errors == [], f"Schema errors: {errors}"


class TestFileRoundtrip:
    def test_generate_write_validate(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Full pipeline: generate → write → validate from file."""
        result = generate("sequential", num_runs=10, noise_rate=0.3, seed=42)

        ocel_path = tmp_path / "output.jsonocel"
        write_ocel_json(result.log, ocel_path)

        errors = validate_ocel_file(ocel_path)
        assert errors == [], f"File validation errors: {errors}"


class TestManifest:
    def test_manifest_structure(self) -> None:
        result = generate("sequential", num_runs=10, noise_rate=0.5, seed=42)
        manifest = build_manifest(result, seed=42)

        assert manifest["generator"] == "ocelgen"
        assert manifest["pattern"] == "sequential"
        assert manifest["seed"] == 42
        assert manifest["total_runs"] == 10
        assert manifest["conformant_runs"] + manifest["deviant_runs"] == 10
        assert len(manifest["runs"]) == 10

    def test_manifest_ground_truth_consistency(self) -> None:
        """Manifest deviation records should match is_deviation attrs in the log."""
        result = generate("sequential", num_runs=20, noise_rate=0.5, seed=42)
        manifest = build_manifest(result)

        # Runs marked deviant in manifest
        manifest_deviant_ids = {
            r["run_id"] for r in manifest["runs"] if not r["is_conformant"]
        }

        # Runs with is_deviation=true events in the log
        log_deviant_ids = set()
        for event in result.log.events:
            for attr in event.attributes:
                if attr.name == "is_deviation" and attr.value == "true":
                    for a2 in event.attributes:
                        if a2.name == "run_id":
                            log_deviant_ids.add(a2.value)

        # Every manifest deviant should have deviation events in the log
        assert manifest_deviant_ids.issubset(log_deviant_ids), (
            f"Manifest says deviant but no deviation events: "
            f"{manifest_deviant_ids - log_deviant_ids}"
        )

    def test_deviation_summary_counts(self) -> None:
        result = generate("sequential", num_runs=50, noise_rate=0.5, seed=42)
        manifest = build_manifest(result)
        summary = manifest["deviation_summary"]
        total = sum(summary.values())
        assert total == len(result.deviations)


class TestNormativeModel:
    def test_normative_model_structure(self) -> None:
        result = generate("sequential", num_runs=1, seed=42)
        model = template_to_dict(result.template)
        assert model["name"] == "sequential"
        assert len(model["steps"]) == 3
        assert len(model["edges"]) == 2

    def test_normative_model_for_each_pattern(self) -> None:
        for name in PATTERN_REGISTRY:
            result = generate(name, num_runs=1, seed=42)
            model = template_to_dict(result.template)
            assert model["name"] == name
            assert len(model["steps"]) > 0
