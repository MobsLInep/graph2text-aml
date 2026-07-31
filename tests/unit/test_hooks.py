from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "hooks"))

from check_no_data_staged import main, offending


def test_blocks_data_and_artifacts():
    staged = [
        "src/g2t_aml/facts/extract.py",
        "data/raw/amlworld/HI-Small_Trans.csv",
        "artifacts/metrics/run.json",
        "README.md",
    ]
    assert offending(staged) == [
        "data/raw/amlworld/HI-Small_Trans.csv",
        "artifacts/metrics/run.json",
    ]


def test_allows_gitkeep_markers():
    assert offending(["data/raw/.gitkeep", "artifacts/runs/.gitkeep"]) == []


def test_allows_similarly_named_paths_elsewhere():
    assert offending(["src/g2t_aml/data/loader.py", "docs/data_cards/amlworld.md"]) == []


def test_main_exit_codes():
    assert main(["README.md"]) == 0
    assert main(["data/raw/x.csv"]) == 1
