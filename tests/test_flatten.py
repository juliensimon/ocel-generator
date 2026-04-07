"""Tests for OCEL-to-tabular flattening."""

from ocelgen.generation.engine import generate
from ocelgen.upload.flatten import flatten_log


class TestFlatten:
    def test_flatten_returns_list_of_dicts(self) -> None:
        result = generate("sequential", num_runs=2, noise_rate=0.0, seed=42)
        rows = flatten_log(result.log, domain="test-domain")
        assert isinstance(rows, list)
        assert len(rows) > 0
        assert isinstance(rows[0], dict)

    def test_flatten_has_required_columns(self) -> None:
        result = generate("sequential", num_runs=1, noise_rate=0.0, seed=42)
        rows = flatten_log(result.log, domain="test-domain")
        required = {
            "event_id",
            "event_type",
            "timestamp",
            "run_id",
            "sequence_number",
            "is_deviation",
            "deviation_type",
            "domain",
            "is_conformant",
            "pattern",
            "user_query",
        }
        for row in rows:
            assert required.issubset(row.keys()), f"Missing columns: {required - row.keys()}"

    def test_flatten_event_count_matches_log(self) -> None:
        result = generate("sequential", num_runs=3, noise_rate=0.0, seed=42)
        rows = flatten_log(result.log, domain="test-domain")
        assert len(rows) == len(result.log.events)

    def test_flatten_domain_column_set(self) -> None:
        result = generate("sequential", num_runs=1, noise_rate=0.0, seed=42)
        rows = flatten_log(result.log, domain="my-domain")
        for row in rows:
            assert row["domain"] == "my-domain"

    def test_flatten_resolves_agent_role(self) -> None:
        result = generate("sequential", num_runs=1, noise_rate=0.0, seed=42)
        rows = flatten_log(result.log, domain="test")
        agent_invoked_rows = [r for r in rows if r["event_type"] == "agent_invoked"]
        assert len(agent_invoked_rows) > 0
        for row in agent_invoked_rows:
            assert row["agent_role"] != "", f"agent_role not resolved for {row['event_id']}"

    def test_flatten_with_deviations(self) -> None:
        result = generate("sequential", num_runs=10, noise_rate=0.5, seed=42)
        rows = flatten_log(result.log, domain="test")
        deviant_rows = [r for r in rows if r["is_deviation"]]
        assert len(deviant_rows) > 0

    def test_flatten_to_parquet(self, tmp_path) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        result = generate("sequential", num_runs=2, noise_rate=0.0, seed=42)
        rows = flatten_log(result.log, domain="test")

        table = pa.Table.from_pylist(rows)
        path = tmp_path / "test.parquet"
        pq.write_table(table, path)

        read_back = pq.read_table(path)
        assert read_back.num_rows == len(rows)
