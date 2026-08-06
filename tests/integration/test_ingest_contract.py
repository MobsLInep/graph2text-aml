"""Phase 1 ingest contract.

Runs the real entrypoint over the 500-row fixture, so the whole chain — verify, load,
canonicalise, compute statistics, write Parquet, write a manifest — is exercised without
needing the 475 MB release. Fixture files are unpinned in the registry, so verification
reports UNVERIFIED and proceeds, which is the intended behaviour for a first ingest.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from g2t_aml.data.canonical import CANONICAL_SCHEMA_VERSION, CanonicalGraph

pytestmark = [pytest.mark.integration, pytest.mark.slow]

FIXTURE_ROWS = 500


@pytest.fixture(scope="module")
def ingest_run(repo_root, tmp_path_factory):
    """Run scripts/01_ingest.py against the fixture tree, into a temp output tree."""
    out = tmp_path_factory.mktemp("ingest")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/01_ingest.py",
            "data=amlworld_fixture",
            f"paths.raw_dir={repo_root / 'tests' / 'fixtures'}",
            f"paths.interim_dir={out / 'interim'}",
            f"paths.runs_dir={out / 'runs'}",
            "ingest.verify_checksums=false",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"ingest failed:\n{result.stdout}\n{result.stderr}")
    return out / "interim" / "amlworld_hi_small"


def test_writes_the_canonical_artifacts(ingest_run):
    for name in ("nodes.parquet", "edges.parquet", "canonical.json", "manifest.json"):
        assert (ingest_run / name).exists(), name


def test_written_graph_reloads(ingest_run):
    graph = CanonicalGraph.load(ingest_run)
    assert graph.num_edges == FIXTURE_ROWS
    assert graph.dataset == "amlworld_hi_small"


def test_manifest_records_content_hashes_for_every_artifact(ingest_run):
    manifest = json.loads((ingest_run / "manifest.json").read_text())
    names = {a["name"] for a in manifest["artifacts"]}
    assert {"nodes.parquet", "edges.parquet", "canonical.json"} <= names
    for artifact in manifest["artifacts"]:
        assert len(artifact["sha256"]) == 64
        assert artifact["size_bytes"] > 0


def test_manifest_pins_the_schema_version(ingest_run):
    manifest = json.loads((ingest_run / "manifest.json").read_text())
    assert manifest["canonical_schema_version"] == CANONICAL_SCHEMA_VERSION


def test_manifest_records_that_the_run_was_not_subsetted(ingest_run):
    """A subsetted ingest must never be mistaken for a complete one."""
    manifest = json.loads((ingest_run / "manifest.json").read_text())
    assert manifest["is_complete_dataset"] is True
    assert manifest["subsetted_n_rows"] is None


def test_manifest_carries_the_availability_mask(ingest_run):
    manifest = json.loads((ingest_run / "manifest.json").read_text())
    mask = manifest["graph"]["availability"]
    assert mask["entity_types"] is False
    assert mask["monetary_amounts"] is True


def test_statistics_report_is_written(ingest_run):
    statistics = json.loads((ingest_run / "statistics.json").read_text())
    assert statistics["counts"]["num_edges"] == FIXTURE_ROWS
    assert "typology_distribution" in statistics
    assert statistics["extra"]["num_pattern_streams"] == 3


def test_subsetted_run_is_recorded_as_subsetted(repo_root, tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/01_ingest.py",
            "data=amlworld_fixture",
            f"paths.raw_dir={repo_root / 'tests' / 'fixtures'}",
            f"paths.interim_dir={tmp_path / 'interim'}",
            f"paths.runs_dir={tmp_path / 'runs'}",
            "ingest.verify_checksums=false",
            "ingest.n_rows=100",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads(
        (tmp_path / "interim" / "amlworld_hi_small" / "manifest.json").read_text()
    )
    assert manifest["subsetted_n_rows"] == 100
    assert manifest["is_complete_dataset"] is False


def test_elliptic2_skips_cleanly_when_access_is_not_granted(repo_root, tmp_path):
    """Phase 1 does not block on an access-gated substrate."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/01_ingest.py",
            "data=elliptic2",
            f"paths.raw_dir={tmp_path / 'empty'}",
            f"paths.interim_dir={tmp_path / 'interim'}",
            f"paths.runs_dir={tmp_path / 'runs'}",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "skipped" in (result.stdout + result.stderr)


def test_checksum_mismatch_aborts_the_ingest(repo_root, tmp_path):
    """Never proceed past a checksum mismatch."""
    raw = tmp_path / "raw" / "amlworld"
    raw.mkdir(parents=True)
    source = repo_root / "tests" / "fixtures" / "amlworld"
    (raw / "HI-Small_Trans.csv").write_text((source / "HI-Small_Trans.csv").read_text())
    (raw / "HI-Small_Patterns.txt").write_text((source / "HI-Small_Patterns.txt").read_text())
    result = subprocess.run(
        [
            sys.executable,
            "scripts/01_ingest.py",
            f"paths.raw_dir={tmp_path / 'raw'}",
            f"paths.interim_dir={tmp_path / 'interim'}",
            f"paths.runs_dir={tmp_path / 'runs'}",
            "ingest.verify_checksums=true",  # real digests are pinned; the fixture differs
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "mismatch" in (result.stdout + result.stderr).lower()
