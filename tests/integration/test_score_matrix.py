"""The step between a completed matrix run and the aggregator (D-112).

The runner writes ``generations.jsonl`` and a completion marker; the aggregator reads
``metrics.json`` and ``per_case.jsonl``. ``scripts/11c_score_matrix.py`` is what turns the
first pair into the second, and until it existed the aggregator found no metrics on any
real run. These tests drive the script on the committed quickstart corpus.
"""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("scipy")

from g2t_aml.experiments.aggregate import collect_rows  # noqa: E402
from g2t_aml.experiments.registry import get_system  # noqa: E402
from g2t_aml.experiments.runner import write_completion_marker  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "corpus" / "bronze_quickstart.jsonl.gz"


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Unpack the committed 220-record Bronze fixture."""
    target = tmp_path / "bronze.jsonl"
    with gzip.open(FIXTURE, "rb") as src, target.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return target


def _completed_b1_run(root: Path, corpus: Path, *, n: int = 30) -> Path:
    """Materialise a completed B1 run whose generations are the first ``n`` Bronze texts."""
    directory = root / "B1" / "seed42" / "hash0000"
    directory.mkdir(parents=True)
    rows = []
    with corpus.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            rows.append(
                {
                    "case_id": record["case_id"],
                    "system": "B1",
                    "seed": 42,
                    "stream": "balanced",
                    "texts": [record["target_narrative"]],
                }
            )
            if len(rows) == n:
                break
    (directory / "generations.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    write_completion_marker(
        directory, run_id="B1_seed42", system_id="B1", seed=42, config_hash="hash0000"
    )
    return directory


def _score(root: Path, corpus: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "11c_score_matrix.py"),
            "--root",
            str(root),
            "--corpus",
            str(corpus),
            "--systems",
            "B1",
            "--resamples",
            "200",
            *extra,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_a_scored_run_is_what_the_aggregator_reads(tmp_path: Path, corpus: Path) -> None:
    """Before this step the aggregator reported 'run completed but wrote no metrics.json'."""
    root = tmp_path / "matrix"
    directory = _completed_b1_run(root, corpus)

    rows, missing, _ = collect_rows(root, specs=[get_system("B1")])
    assert [m.reason for m in missing] == ["run completed but wrote no metrics.json"]

    result = _score(root, corpus)
    assert result.returncode == 0, result.stderr[-2000:]
    assert (directory / "metrics.json").is_file()

    rows, missing, _ = collect_rows(root, specs=[get_system("B1")])
    assert not missing
    headline = [r for r in rows if r.metric == "zero_hallucination_rate" and r.typology == "all"]
    # Bronze texts scored against their own fact records: faithful by construction.
    assert headline and headline[0].value == pytest.approx(1.0)
    assert headline[0].n_cases == 30


def test_per_case_values_carry_the_aggregator_metric_names(tmp_path: Path, corpus: Path) -> None:
    """The paired tests pair on these rows; a renamed key silently disables them."""
    root = tmp_path / "matrix"
    directory = _completed_b1_run(root, corpus, n=12)
    assert _score(root, corpus).returncode == 0

    rows = [json.loads(line) for line in (directory / "per_case.jsonl").read_text().splitlines()]
    assert len(rows) == 12
    for key in (
        "zero_hallucination_rate",
        "fact_precision",
        "hallucination_rate",
        "unverifiable_rate",
        "fact_coverage",
        "fact_f1",
        "critical_error_rate",
    ):
        assert all(isinstance(r[key], float) for r in rows), key
    assert all(r["stream"] == "balanced" for r in rows)


def test_a_scored_run_is_not_rescored_without_force(tmp_path: Path, corpus: Path) -> None:
    """Scoring is derived and deterministic; a second pass is skipped, not repeated."""
    root = tmp_path / "matrix"
    directory = _completed_b1_run(root, corpus, n=5)
    assert _score(root, corpus).returncode == 0
    first = (directory / "metrics.json").stat().st_mtime_ns
    assert _score(root, corpus).returncode == 0
    assert (directory / "metrics.json").stat().st_mtime_ns == first
