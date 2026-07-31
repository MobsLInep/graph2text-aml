from __future__ import annotations

import os
import random

import numpy as np
import pytest

from g2t_aml.utils.seeding import CUBLAS_WORKSPACE_CONFIG, seed_everything


def test_returns_populated_record():
    record = seed_everything(42)
    assert record["seed"] == 42
    assert record["python_random"] is True
    assert record["numpy"] is True
    assert record["env"]["PYTHONHASHSEED"] == "42"
    assert record["env"]["CUBLAS_WORKSPACE_CONFIG"] == CUBLAS_WORKSPACE_CONFIG
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == CUBLAS_WORKSPACE_CONFIG


def test_record_is_json_serialisable():
    import json

    json.dumps(seed_everything(7))


def test_reproducible_across_calls():
    seed_everything(1234)
    first = (random.random(), np.random.rand(3).tolist())
    seed_everything(1234)
    second = (random.random(), np.random.rand(3).tolist())
    assert first == second


def test_different_seeds_diverge():
    seed_everything(1)
    a = np.random.rand(5).tolist()
    seed_everything(2)
    b = np.random.rand(5).tolist()
    assert a != b


@pytest.mark.parametrize("bad", [-1, 2**32])
def test_rejects_out_of_range_seed(bad):
    with pytest.raises(ValueError, match="seed must be in"):
        seed_everything(bad)


def test_non_deterministic_mode_still_seeds():
    record = seed_everything(3, deterministic=False)
    assert record["deterministic"] is False
    assert record["python_random"] is True
