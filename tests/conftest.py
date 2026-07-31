from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def configs_dir() -> Path:
    return REPO_ROOT / "configs"


@pytest.fixture
def synthetic_ids() -> list[str]:
    """Synthetic account IDs only -- invariant 8 forbids real identifiers in fixtures."""
    return [f"ACCT-{i:06d}" for i in range(10)]
