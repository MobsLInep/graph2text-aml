"""Phase 11: the runner resumes, isolates failures, and refuses a stale completion marker.

Three properties carry real risk and are asserted here rather than assumed:

- **Resumption is marker AND hash.** A marker alone would skip a run whose config has
  since changed, reporting an old number under a new configuration. A hash alone would
  re-run everything on a GPU-week matrix.
- **Failure isolation.** One arm failing must not lose the other twenty-four.
- **Dependency blocking.** A5 reads S1's checkpoint, so if S1 fails, A5 must be blocked
  rather than attempted against a checkpoint that is not there.

Every executor here is a plain callable, which is the point of the runner taking them as
an argument: the whole scheduler is exercised on CPU in milliseconds, the same discipline
as ScriptedTeacher in Phase 5 and the stub backbone in Phase 9.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from g2t_aml.experiments.registry import (
    Executor,
    Resource,
    SystemSpec,
    get_system,
)
from g2t_aml.experiments.runner import (
    COMPLETION_MARKER,
    RunStatus,
    is_complete,
    plan_matrix,
    run_directory,
    run_matrix,
    write_completion_marker,
)
from g2t_aml.utils.io import read_json


def _spec(system_id: str, **kwargs: Any) -> SystemSpec:
    """Build a minimal spec for the scheduler tests."""
    defaults = {
        "role": "test",
        "description": "test",
        "executor": Executor.TEMPLATE,
        "resource": Resource.CPU,
    }
    return SystemSpec(system_id=system_id, **{**defaults, **kwargs})


def _ok(_spec: SystemSpec, _seed: int, directory: Path) -> dict[str, Any]:
    (directory / "generations.jsonl").write_text("{}\n", encoding="utf-8")
    return {"n_cases": 1}


def _boom(spec: SystemSpec, _seed: int, _directory: Path) -> dict[str, Any]:
    raise RuntimeError(f"{spec.system_id} exploded")


@pytest.fixture
def executors():
    return {str(Executor.TEMPLATE): _ok}


def test_plan_expands_seeds_and_orders_dependencies(tmp_path):
    plan = plan_matrix(root=tmp_path)
    order = [o.run_id for o in plan.outcomes]
    assert order.index("S1_seed42") < order.index("A5_seed42")
    assert "S1_seed1337" in order and "S1_seed2024" in order
    assert plan.problems == ()


def test_plan_splits_gpu_from_parallelisable_work(tmp_path):
    plan = plan_matrix(root=tmp_path)
    gpu = {o.spec.system_id for o in plan.gpu_runs}
    parallel = {o.spec.system_id for o in plan.parallel_runs}
    assert {"S1", "A1", "B7"} <= gpu
    assert {"B1", "B3"} <= parallel
    assert not (gpu & parallel)


def test_run_directory_is_keyed_on_the_config_hash(tmp_path):
    """Invariant 6 mechanically: a changed config cannot land on top of an old result."""
    a = run_directory(tmp_path, "S1", 42, "aaaaaaaaaaaaaaaa")
    b = run_directory(tmp_path, "S1", 42, "bbbbbbbbbbbbbbbb")
    assert a != b
    assert a.parent == b.parent


def test_completion_needs_both_the_marker_and_a_matching_hash(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    assert not is_complete(directory, "hash-a")
    write_completion_marker(directory, run_id="r", system_id="S1", seed=42, config_hash="hash-a")
    assert is_complete(directory, "hash-a")
    assert not is_complete(directory, "hash-b")


def test_a_truncated_marker_is_not_complete(tmp_path):
    """A killed job leaves a half-written file; it must not read as a finished run."""
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / COMPLETION_MARKER).write_text('{"config_hash": "ha', encoding="utf-8")
    assert not is_complete(directory, "hash-a")


def test_completed_runs_are_skipped_on_a_second_pass(tmp_path, executors):
    specs = [_spec("B1")]
    plan = plan_matrix(specs, root=tmp_path)
    first = run_matrix(plan, executors)
    assert [r.status for r in first.records] == [RunStatus.COMPLETED]

    calls: list[str] = []

    def _counting(spec: SystemSpec, seed: int, directory: Path) -> dict[str, Any]:
        calls.append(spec.system_id)
        return _ok(spec, seed, directory)

    second = run_matrix(plan, {str(Executor.TEMPLATE): _counting})
    assert [r.status for r in second.records] == [RunStatus.SKIPPED]
    assert calls == [], "a completed run must not be re-executed"


def test_a_changed_config_invalidates_the_completion_marker(tmp_path, executors):
    """The half of resumption that stops an old number being reported under a new config."""
    specs = [_spec("B1")]
    run_matrix(plan_matrix(specs, root=tmp_path, overrides={"corpus": "v1"}), executors)

    changed = plan_matrix(specs, root=tmp_path, overrides={"corpus": "v2"})
    result = run_matrix(changed, executors)
    assert [r.status for r in result.records] == [RunStatus.COMPLETED]
    # The old run's directory is untouched -- invariant 6: never delete or overwrite.
    seed_dirs = list((tmp_path / "B1" / "seed42").iterdir())
    assert len(seed_dirs) == 2


def test_descriptive_fields_do_not_invalidate_a_marker(tmp_path):
    """Editing a `role` string must not invalidate a GPU-week of completed runs."""
    a = plan_matrix([_spec("B1", role="ceiling")], root=tmp_path)
    b = plan_matrix([_spec("B1", role="the faithfulness ceiling")], root=tmp_path)
    assert a.outcomes[0].config_hash == b.outcomes[0].config_hash


def test_a_real_config_change_does_invalidate_a_marker(tmp_path):
    a = plan_matrix([_spec("B1", guard=True)], root=tmp_path)
    b = plan_matrix([_spec("B1", guard=False)], root=tmp_path)
    assert a.outcomes[0].config_hash != b.outcomes[0].config_hash


def test_force_reruns_a_completed_run(tmp_path, executors):
    specs = [_spec("B1")]
    plan = plan_matrix(specs, root=tmp_path)
    run_matrix(plan, executors)
    result = run_matrix(plan, executors, force=True)
    assert [r.status for r in result.records] == [RunStatus.COMPLETED]


def test_one_failure_does_not_abort_the_matrix(tmp_path):
    specs = [_spec("B1"), _spec("B3", executor=Executor.API_ZERO_SHOT), _spec("B6")]
    plan = plan_matrix(specs, root=tmp_path)
    result = run_matrix(plan, {str(Executor.TEMPLATE): _ok, str(Executor.API_ZERO_SHOT): _boom})
    statuses = {r.system_id: r.status for r in result.records}
    assert statuses == {
        "B1": RunStatus.COMPLETED,
        "B3": RunStatus.FAILED,
        "B6": RunStatus.COMPLETED,
    }
    assert not result.ok


def test_a_failure_records_its_traceback_on_disk(tmp_path):
    specs = [_spec("B3", executor=Executor.API_ZERO_SHOT)]
    plan = plan_matrix(specs, root=tmp_path)
    result = run_matrix(plan, {str(Executor.API_ZERO_SHOT): _boom})
    record = result.records[0]
    assert "exploded" in (record.error or "")
    assert "RuntimeError" in (record.traceback or "")
    payload = read_json(Path(record.directory) / "FAILED.json")
    assert "exploded" in payload["error"]
    assert not (Path(record.directory) / COMPLETION_MARKER).exists()


def test_a_missing_executor_fails_that_run_and_no_other(tmp_path):
    specs = [_spec("B1"), _spec("B6", executor=Executor.LOCAL_ZERO_SHOT)]
    plan = plan_matrix(specs, root=tmp_path)
    result = run_matrix(plan, {str(Executor.TEMPLATE): _ok})
    statuses = {r.system_id: r.status for r in result.records}
    assert statuses["B1"] is RunStatus.COMPLETED
    assert statuses["B6"] is RunStatus.FAILED
    assert "no executor registered" in (
        next(r for r in result.records if r.system_id == "B6").error or ""
    )


def test_a_dependent_is_blocked_when_its_dependency_fails(tmp_path):
    """A5 reads S1's checkpoint; running it against a checkpoint that is not there is worse
    than not running it, because it would produce a number."""
    parent = _spec("P1", executor=Executor.API_ZERO_SHOT)
    child = _spec("D1", depends_on=("P1",))
    plan = plan_matrix([parent, child], root=tmp_path)
    result = run_matrix(plan, {str(Executor.API_ZERO_SHOT): _boom, str(Executor.TEMPLATE): _ok})
    statuses = {r.system_id: r.status for r in result.records}
    assert statuses["P1"] is RunStatus.FAILED
    assert statuses["D1"] is RunStatus.BLOCKED
    blocked = next(r for r in result.records if r.system_id == "D1")
    assert blocked.blocked_by == "P1"


def test_a_dependent_is_blocked_when_only_one_seed_of_its_dependency_failed(tmp_path):
    """A partly-failed dependency is not a dependency: the dependent would read whichever
    seed happened to survive, which is not the arm the table says it is."""
    calls: list[int] = []

    def _fail_second_seed(spec: SystemSpec, seed: int, directory: Path) -> dict[str, Any]:
        calls.append(seed)
        if seed == 1337:
            raise RuntimeError("seed 1337 diverged")
        return _ok(spec, seed, directory)

    # `S1` rather than a neutral id: the multi-seed policy is what makes this scenario
    # possible at all, and the registry refuses a non-central system at two seeds.
    parent = _spec("S1", seeds=(42, 1337, 2024))
    child = _spec("D1", depends_on=("S1",))
    plan = plan_matrix([parent, child], root=tmp_path)
    result = run_matrix(plan, {str(Executor.TEMPLATE): _fail_second_seed})
    statuses = {r.run_id: r.status for r in result.records}
    assert statuses["S1_seed42"] is RunStatus.COMPLETED
    assert statuses["S1_seed1337"] is RunStatus.FAILED
    assert statuses["S1_seed2024"] is RunStatus.COMPLETED
    assert statuses["D1_seed42"] is RunStatus.BLOCKED


def test_an_external_artifact_dependency_blocks_when_absent(tmp_path, executors):
    spec = _spec("B2", depends_on=("encoder:gatv2",))
    plan = plan_matrix([spec], root=tmp_path)
    blocked = run_matrix(plan, executors, external_artifacts=())
    assert blocked.records[0].status is RunStatus.BLOCKED
    assert blocked.records[0].blocked_by == "encoder:gatv2"

    satisfied = run_matrix(plan, executors, external_artifacts=("encoder:gatv2",))
    assert satisfied.records[0].status is RunStatus.COMPLETED


def test_dry_run_executes_nothing(tmp_path):
    calls: list[str] = []

    def _counting(spec: SystemSpec, _seed: int, _directory: Path) -> dict[str, Any]:
        calls.append(spec.system_id)
        return {}

    plan = plan_matrix([_spec("B1")], root=tmp_path)
    result = run_matrix(plan, {str(Executor.TEMPLATE): _counting}, dry_run=True)
    assert calls == []
    assert result.records[0].status is RunStatus.PENDING


def test_run_matrix_refuses_a_plan_with_registry_problems(tmp_path):
    bad = _spec("Q", trained=True, executor=Executor.TRAINED_GENERATOR)
    plan = plan_matrix([bad], root=tmp_path)
    assert plan.problems
    with pytest.raises(ValueError, match="registry problems"):
        run_matrix(plan, {})


def test_every_run_writes_its_resolved_config(tmp_path, executors):
    plan = plan_matrix([_spec("B1")], root=tmp_path)
    result = run_matrix(plan, executors)
    resolved = read_json(Path(result.records[0].directory) / "resolved_config.json")
    assert resolved["system_id"] == "B1"
    assert resolved["seed"] == 42


def test_summary_is_written_and_tallies_by_status(tmp_path):
    specs = [_spec("B1"), _spec("B3", executor=Executor.API_ZERO_SHOT)]
    plan = plan_matrix(specs, root=tmp_path)
    out = tmp_path / "summary.json"
    run_matrix(
        plan,
        {str(Executor.TEMPLATE): _ok, str(Executor.API_ZERO_SHOT): _boom},
        summary_path=out,
    )
    payload = read_json(out)
    assert payload["tally"] == {"completed": 1, "failed": 1}
    assert payload["ok"] is False


def test_the_real_registry_plans_without_problems(tmp_path):
    plan = plan_matrix(root=tmp_path)
    assert plan.problems == ()
    assert plan.summary()["n_runs"] == len(plan.outcomes)
    assert get_system("S1").system_id in {o.spec.system_id for o in plan.outcomes}


def test_interruption_midway_resumes_from_where_it_stopped(tmp_path):
    """The property the whole marker design exists for, exercised end to end."""
    specs = [_spec("B1"), _spec("B2"), _spec("B3")]
    plan = plan_matrix(specs, root=tmp_path)

    attempted: list[str] = []

    def _die_on_b2(spec: SystemSpec, seed: int, directory: Path) -> dict[str, Any]:
        attempted.append(spec.system_id)
        if spec.system_id == "B2":
            raise RuntimeError("interrupted")
        return _ok(spec, seed, directory)

    first = run_matrix(plan, {str(Executor.TEMPLATE): _die_on_b2})
    assert [r.status for r in first.records] == [
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.COMPLETED,
    ]

    attempted.clear()

    def _succeed_everywhere(spec: SystemSpec, seed: int, directory: Path) -> dict[str, Any]:
        attempted.append(spec.system_id)
        return _ok(spec, seed, directory)

    second = run_matrix(plan, {str(Executor.TEMPLATE): _succeed_everywhere})
    # Only the failed run is re-attempted; the two that completed are skipped.
    assert attempted == ["B2"]
    assert [r.status for r in second.records] == [
        RunStatus.SKIPPED,
        RunStatus.COMPLETED,
        RunStatus.SKIPPED,
    ]
    assert second.ok
