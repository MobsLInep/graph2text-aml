"""The encoder's per-case scores reach B2 through one path, written and read by one module.

``scripts/07b_score_cases.py`` and the B2 executor each used to spell the file's location
themselves, and only the reader's spelling existed: ``make score-cases`` succeeded, B2 asked
for a file nothing wrote, and the classify-then-template baseline could not run (D-109).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from g2t_aml.experiments.executors import (
    encoder_scores_path,
    load_encoder_scores,
    write_encoder_scores,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _rows() -> list[dict[str, object]]:
    return [
        {"case_id": "synthetic-0001", "gnn_risk_score": 0.91, "score_percentile": 99.0},
        {"case_id": "synthetic-0002", "gnn_risk_score": 0.04, "score_percentile": 12.5},
    ]


def test_what_score_cases_writes_is_what_b2_reads(tmp_path) -> None:
    """The writer's file is at the reader's path and carries the reader's keys."""
    written = write_encoder_scores(tmp_path, "gatv2", _rows())
    assert written == encoder_scores_path(tmp_path, "gatv2")
    assert load_encoder_scores(written) == {"synthetic-0001": 0.91, "synthetic-0002": 0.04}


def test_each_arm_has_its_own_file(tmp_path) -> None:
    """B2 names its encoder arm; another arm's scores must not answer for it."""
    write_encoder_scores(tmp_path, "mlp", _rows())
    assert not encoder_scores_path(tmp_path, "gatv2").exists()


def test_a_row_without_a_score_is_an_error_at_write_time(tmp_path) -> None:
    """Found when the scores are produced, not when a baseline is missing cases."""
    with pytest.raises(KeyError):
        write_encoder_scores(tmp_path, "gatv2", [{"case_id": "synthetic-0001"}])


def test_the_scoring_script_uses_the_shared_writer() -> None:
    """The script cannot be executed here (it needs a trained checkpoint), so pin the call."""
    source = (REPO_ROOT / "scripts" / "07b_score_cases.py").read_text(encoding="utf-8")
    assert "write_encoder_scores(" in source
