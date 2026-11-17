"""What the GPU executors hand the entrypoints (D-114, D-116, D-117).

The entrypoints themselves need a GPU and an 8B model; these tests capture the command an
executor would run and check it: which checkpoint a trained arm decodes from, that the
matrix's regime overrides reach every trained arm, that a decode-only matrix refuses to
train, that A5 reads S1's weights with the guard on, and that B6 runs its own script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from g2t_aml.experiments import executors
from g2t_aml.experiments.registry import get_system
from g2t_aml.utils.io import write_json

REGIME = ["training.per_device_batch_size=1", "training.gradient_accumulation=32"]


@pytest.fixture
def captured(monkeypatch) -> list[dict[str, Any]]:
    """Replace the subprocess call with a recorder that writes a one-row generations file."""
    calls: list[dict[str, Any]] = []

    def fake(repo_root: Path, script: str, *, overrides, system_id: str) -> int:
        calls.append({"script": script, "overrides": list(overrides), "system": system_id})
        out = next(
            (Path(o.split("=", 1)[1]) for o in overrides if o.startswith("hydra.run.dir=")),
            None,
        )
        if out is None and "--out-dir" in overrides:
            out = Path(overrides[overrides.index("--out-dir") + 1])
        assert out is not None
        out.mkdir(parents=True, exist_ok=True)
        (out / "generations.jsonl").write_text('{"case_id": "c1", "texts": ["x"]}\n')
        return 0

    monkeypatch.setattr(executors, "_invoke_entrypoint", fake)
    return calls


def _run_dir(root: Path, system: str, seed: int, **context: Any) -> Path:
    directory = root / "matrix" / system / f"seed{seed}" / "hash"
    directory.mkdir(parents=True)
    base = {"repo_root": str(root), "bronze_path": str(root / "bronze.jsonl")}
    write_json(directory / "resolved_config.json", {"context": {**base, **context}})
    return directory


def _trained(root: Path, system: str, seed: int) -> Path:
    run = root / "train" / f"{system}_seed{seed}" / "2026-10-15_11-25-16"
    run.mkdir(parents=True)
    (run / f"{system}_final.pt").write_bytes(b"weights")
    (run / "TRAINED").write_text("2026-10-16")
    return run / f"{system}_final.pt"


def _value(overrides: list[str], key: str) -> str | None:
    return next((o.split("=", 1)[1] for o in overrides if o.startswith(f"{key}=")), None)


def test_a_queue_trained_arm_is_decoded_not_retrained(tmp_path, captured) -> None:
    final = _trained(tmp_path, "B7", 42)
    directory = _run_dir(
        tmp_path, "B7", 42, trained_root=str(tmp_path / "train"), generator_overrides=REGIME
    )
    payload = executors.run_trained_generator(get_system("B7"), 42, directory)
    overrides = captured[0]["overrides"]
    assert _value(overrides, "generator.resume_from") == final.as_posix()
    assert _value(overrides, "training.resume_from") is None
    assert all(o in overrides for o in REGIME)
    assert payload["trained_from"] == str(final)


def test_an_unfinished_training_run_is_not_decoded(tmp_path, captured) -> None:
    """A final checkpoint without the queue's TRAINED marker is not trusted."""
    final = _trained(tmp_path, "S2", 42)
    (final.parent / "TRAINED").unlink()
    directory = _run_dir(
        tmp_path, "S2", 42, trained_root=str(tmp_path / "train"), require_trained=True
    )
    with pytest.raises(executors.ExecutorError, match="no trained checkpoint"):
        executors.run_trained_generator(get_system("S2"), 42, directory)
    assert captured == []


def test_without_a_trained_root_the_arm_trains_here(tmp_path, captured) -> None:
    directory = _run_dir(tmp_path, "S1", 42)
    executors.run_trained_generator(get_system("S1"), 42, directory)
    overrides = captured[0]["overrides"]
    assert _value(overrides, "generator.resume_from") is None
    assert _value(overrides, "training.guard.enabled") == "false"


def test_a_partly_trained_arm_in_its_own_directory_continues(tmp_path, captured) -> None:
    directory = _run_dir(tmp_path, "A1", 1337)
    (directory / "A1_resume.pt").write_bytes(b"state")
    executors.run_trained_generator(get_system("A1"), 1337, directory)
    assert (
        _value(captured[0]["overrides"], "training.resume_from")
        == (directory / "A1_resume.pt").as_posix()
    )


def test_a5_decodes_s1s_weights_with_the_guard_on(tmp_path, captured) -> None:
    final = _trained(tmp_path, "S1", 42)
    directory = _run_dir(tmp_path, "A5", 42, trained_root=str(tmp_path / "train"))
    executors.run_trained_generator(get_system("A5"), 42, directory)
    overrides = captured[0]["overrides"]
    assert _value(overrides, "generator.resume_from") == final.as_posix()
    assert _value(overrides, "training.guard.enabled") == "true"
    assert _value(overrides, "experiment") == "matrix_a5"


def test_b6_runs_the_zero_shot_script_not_the_finetuning_entrypoint(tmp_path, captured) -> None:
    directory = _run_dir(tmp_path, "B6", 42)
    executors.run_local_zero_shot(get_system("B6"), 42, directory)
    assert captured[0]["script"] == "09c_local_zero_shot.py"
    assert captured[0]["overrides"][captured[0]["overrides"].index("--model") + 1] == (
        get_system("B6").base_model
    )
