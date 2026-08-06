from __future__ import annotations

from pathlib import Path

import pytest

GOLDEN_DIR = Path(__file__).resolve().parent / "case_facts"


@pytest.fixture(scope="session")
def golden_dir() -> Path:
    return GOLDEN_DIR
