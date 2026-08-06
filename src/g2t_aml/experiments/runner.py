"""Phase 11: the matrix runner -- dependency resolution, resumption, failure isolation.

The matrix is 26 runs across three resource classes, several of them GPU-days apart, on
hardware that will be interrupted. Three properties follow from that and are the whole
reason this module exists:

**Resumable.** A run is complete when its directory holds a completion marker whose
recorded config hash matches the config it would be given now. Both halves matter. A marker
alone would skip a run whose config has since changed -- silently reporting an old number
under a new configuration, which is the failure mode that produces an irreproducible table.
A hash alone would re-run everything that completed, which on a GPU-week matrix is not a
resumption strategy.

**Failure-isolating.** One system failing does not abort the matrix. It is recorded with
its traceback in the run's own directory and in the plan's summary, and the runner
continues. The exception is a *dependency* failure: A5 reads S1's checkpoint, so if S1
fails, A5 is marked blocked rather than attempted and its blocker is named.

**Scheduled by resource.** GPU jobs serialise against each other because the card holds one
model. CPU-only and API work parallelises. That is expressed as a resource class per run
(:class:`~g2t_aml.experiments.registry.Resource`), not as a hand-maintained order.

Every run gets its own directory holding the resolved config, the run context, and
whatever the executor produced. Nothing is ever overwritten (invariant 6): a re-run with a
changed config writes a new directory, and the old one keeps its number.

The runner never imports torch. Executors are supplied as callables, which is what lets the
whole scheduler, the resumption logic and the failure isolation be tested on CPU in
milliseconds with no model anywhere near it -- the same discipline as ``ScriptedTeacher``
in Phase 5 and the stub backbone in Phase 9.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from g2t_aml.experiments.registry import (
    Resource,
    SystemSpec,
    resolution_order,
    validate_registry,
)
from g2t_aml.utils.hashing import hash_config
from g2t_aml.utils.io import read_json, write_json
from g2t_aml.utils.logging import get_logger

__all__ = [
    "COMPLETION_MARKER",
    "ExecutorFn",
    "MatrixResult",
    "RunOutcome",
    "RunPlan",
    "RunRecord",
    "RunStatus",
    "completion_marker_path",
    "is_complete",
    "plan_matrix",
    "run_directory",
    "run_matrix",
    "write_completion_marker",
]

log = get_logger(__name__)

#: The file whose presence, together with a matching config hash, means "done".
COMPLETION_MARKER = "COMPLETED.json"

#: Written beside the marker so a run directory is self-describing.
RESOLVED_CONFIG = "resolved_config.json"


class RunStatus(StrEnum):
    """What happened to one run.

    Attributes:
        PENDING: Planned, not yet attempted.
        SKIPPED: Already complete with a matching config hash.
        STALE: A marker exists but its config hash does not match, so the run is redone
            into a NEW directory and the old one is left intact.
        COMPLETED: Ran to completion this session.
        FAILED: Raised. The traceback is recorded and the matrix continued.
        BLOCKED: A dependency failed or was never run, so this was not attempted.
    """

    PENDING = "pending"
    SKIPPED = "skipped"
    STALE = "stale"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


#: What the runner needs from anything that produces a system's outputs. It receives the
#: spec, the seed and the run directory, and returns whatever it wants recorded in the
#: completion marker -- typically the generations path and a few counts. Raising is how it
#: reports failure; the runner catches, records and continues.
ExecutorFn = Callable[[SystemSpec, int, Path], Mapping[str, Any]]


@dataclass(frozen=True)
class RunOutcome:
    """One planned run and everything known about it before execution.

    Attributes:
        spec: The system.
        seed: The seed this run uses.
        run_id: ``<system_id>_seed<seed>``. Stable across sessions, which is what makes
            resumption possible without a database.
        config: The resolved config for this run, as a plain mapping.
        config_hash: Its canonical hash. The other half of the completion check.
        directory: Where this run writes.
    """

    spec: SystemSpec
    seed: int
    run_id: str
    config: Mapping[str, Any]
    config_hash: str
    directory: Path


@dataclass
class RunRecord:
    """What happened when a planned run was executed.

    Attributes:
        run_id: Matches the outcome's.
        system_id: The system.
        seed: The seed.
        status: The result.
        config_hash: The config this run was given.
        directory: Where it wrote.
        started_utc: ISO timestamp, or None if never started.
        duration_s: Wall time, or None.
        error: The exception's ``repr``, when it failed.
        traceback: The formatted traceback, when it failed. Kept in full: a matrix that
            ran overnight and lost three arms needs to be diagnosable in the morning
            without re-running anything.
        blocked_by: The dependency that prevented this run, when blocked.
        payload: Whatever the executor returned.
    """

    run_id: str
    system_id: str
    seed: int
    status: RunStatus
    config_hash: str
    directory: str
    started_utc: str | None = None
    duration_s: float | None = None
    error: str | None = None
    traceback: str | None = None
    blocked_by: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-serialisable mapping.

        Returns:
            Every field, with the status as its string value.
        """
        return {
            "run_id": self.run_id,
            "system_id": self.system_id,
            "seed": self.seed,
            "status": str(self.status),
            "config_hash": self.config_hash,
            "directory": self.directory,
            "started_utc": self.started_utc,
            "duration_s": self.duration_s,
            "error": self.error,
            "traceback": self.traceback,
            "blocked_by": self.blocked_by,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class RunPlan:
    """Everything the matrix will attempt, in execution order.

    Attributes:
        outcomes: The planned runs, dependency-ordered and grouped so that no GPU run is
            scheduled beside another.
        problems: Registry validation problems. A non-empty list means the plan must not
            be executed.
    """

    outcomes: tuple[RunOutcome, ...]
    problems: tuple[str, ...] = ()

    @property
    def gpu_runs(self) -> tuple[RunOutcome, ...]:
        """Return the runs that must serialise against each other.

        Returns:
            The GPU and GPU-inference runs, in plan order.
        """
        return tuple(
            o for o in self.outcomes if o.spec.resource in (Resource.GPU, Resource.GPU_INFERENCE)
        )

    @property
    def parallel_runs(self) -> tuple[RunOutcome, ...]:
        """Return the runs that may execute concurrently.

        Returns:
            The CPU and API runs, in plan order.
        """
        return tuple(o for o in self.outcomes if o.spec.resource in (Resource.CPU, Resource.API))

    def summary(self) -> dict[str, Any]:
        """Summarise the plan without executing it.

        Returns:
            Run counts by resource class and the ordered run ids, which is what
            ``--dry-run`` prints.
        """
        by_resource: dict[str, int] = {}
        for outcome in self.outcomes:
            key = str(outcome.spec.resource)
            by_resource[key] = by_resource.get(key, 0) + 1
        return {
            "n_runs": len(self.outcomes),
            "runs_by_resource": dict(sorted(by_resource.items())),
            "n_gpu_serialised": len(self.gpu_runs),
            "n_parallelisable": len(self.parallel_runs),
            "order": [o.run_id for o in self.outcomes],
            "problems": list(self.problems),
        }


@dataclass(frozen=True)
class MatrixResult:
    """The outcome of executing a plan.

    Attributes:
        records: One per planned run.
        started_utc: When the matrix started.
        duration_s: Total wall time.
    """

    records: tuple[RunRecord, ...]
    started_utc: str
    duration_s: float

    def by_status(self, status: RunStatus) -> tuple[RunRecord, ...]:
        """Return every record with one status.

        Args:
            status: The status to filter on.

        Returns:
            The matching records, in execution order.
        """
        return tuple(r for r in self.records if r.status is status)

    @property
    def ok(self) -> bool:
        """Report whether every run either completed or was skipped.

        Returns:
            True when nothing failed and nothing was blocked. **A matrix with failures is
            not a matrix to aggregate silently** -- the aggregator reports missing runs by
            name, and this is the flag the script's exit code is built on.
        """
        return not (self.by_status(RunStatus.FAILED) or self.by_status(RunStatus.BLOCKED))

    def to_dict(self) -> dict[str, Any]:
        """Return the whole result as a JSON-serialisable mapping.

        Returns:
            The records plus a status tally, so a summary is readable without counting.
        """
        tally: dict[str, int] = {}
        for record in self.records:
            tally[str(record.status)] = tally.get(str(record.status), 0) + 1
        return {
            "started_utc": self.started_utc,
            "duration_s": self.duration_s,
            "ok": self.ok,
            "tally": dict(sorted(tally.items())),
            "records": [r.to_dict() for r in self.records],
        }


# ------------------------------------------------------------------- directories ---


def run_directory(root: Path | str, system_id: str, seed: int, config_hash: str) -> Path:
    """Return the directory one run writes to.

    The config hash is in the path, which is what makes invariant 6 mechanical rather than
    remembered: a re-run under a changed config cannot land on top of the old one, because
    it resolves to a different directory. The old number stays on disk and stays readable.

    Args:
        root: The matrix root, from ``cfg.paths``.
        system_id: The system.
        seed: The seed.
        config_hash: The run's config hash.

    Returns:
        ``<root>/<system_id>/seed<seed>/<hash-prefix>``.
    """
    return Path(root) / system_id / f"seed{seed}" / config_hash[:12]


def completion_marker_path(directory: Path | str) -> Path:
    """Return where a run's completion marker lives.

    Args:
        directory: The run directory.

    Returns:
        The marker path.
    """
    return Path(directory) / COMPLETION_MARKER


def write_completion_marker(
    directory: Path | str,
    *,
    run_id: str,
    system_id: str,
    seed: int,
    config_hash: str,
    payload: Mapping[str, Any] | None = None,
    duration_s: float | None = None,
) -> Path:
    """Mark a run complete.

    Written last, atomically, and only after the executor has returned. A marker written
    before the work finishes is a marker that makes a half-finished run look done to the
    next resumption, which is exactly the state a killed job leaves behind.

    Args:
        directory: The run directory.
        run_id: The run identifier.
        system_id: The system.
        seed: The seed.
        config_hash: The config this run executed under.
        payload: Whatever the executor returned.
        duration_s: Wall time.

    Returns:
        The marker path.
    """
    return write_json(
        completion_marker_path(directory),
        {
            "run_id": run_id,
            "system_id": system_id,
            "seed": seed,
            "config_hash": config_hash,
            "completed_utc": datetime.now(UTC).isoformat(),
            "duration_s": duration_s,
            "payload": dict(payload or {}),
        },
    )


def is_complete(directory: Path | str, config_hash: str) -> bool:
    """Report whether a run is complete under a given config.

    **Both halves are load-bearing.** The marker says the work finished; the hash says it
    finished under the configuration being asked for now. A marker whose hash does not
    match is not "complete with a warning" -- it is a different experiment that happens to
    share a system id.

    Args:
        directory: The run directory.
        config_hash: The config hash being asked about.

    Returns:
        True when the marker exists and its recorded hash matches.
    """
    marker = completion_marker_path(directory)
    if not marker.is_file():
        return False
    try:
        payload = read_json(marker)
    except (ValueError, OSError):
        # A truncated marker is a killed job, not a completed one.
        return False
    if not isinstance(payload, dict):
        return False
    return payload.get("config_hash") == config_hash


# ------------------------------------------------------------------------ planning ---


def _resolve_config(spec: SystemSpec, seed: int, overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Build the fully-resolved config for one run.

    The spec is the source of truth for every axis the matrix varies; ``overrides`` carries
    the run-invariant context (paths, data substrate, corpus tier) that is the same for
    every system and would be noise repeated sixteen times in the registry.

    Args:
        spec: The system.
        seed: The seed.
        overrides: Run-invariant configuration.

    Returns:
        A plain, JSON-serialisable mapping. This is what gets hashed, so anything absent
        here is something a config change cannot invalidate a completion marker on.
    """
    resolved = dict(spec.to_dict())
    resolved["seed"] = seed
    resolved["context"] = dict(sorted(overrides.items()))
    # Fields that describe the row rather than determine the computation are excluded from
    # the hash input: editing a `role` string must not invalidate a GPU-week of runs.
    for descriptive in ("role", "description", "notes"):
        resolved.pop(descriptive, None)
    return resolved


def plan_matrix(
    specs: Sequence[SystemSpec] | None = None,
    *,
    root: Path | str,
    overrides: Mapping[str, Any] | None = None,
) -> RunPlan:
    """Resolve dependencies, expand seeds, and produce the execution order.

    Args:
        specs: Systems to plan; the whole registry when omitted.
        root: The matrix root directory.
        overrides: Run-invariant configuration folded into every run's config hash.

    Returns:
        The plan. Its ``problems`` field carries any registry inconsistency; a caller that
        executes a plan with problems is running an experiment nobody validated, so
        :func:`run_matrix` refuses.
    """
    ordered = resolution_order(specs)
    context = dict(overrides or {})
    outcomes: list[RunOutcome] = []
    for spec in ordered:
        for seed in sorted(spec.seeds):
            config = _resolve_config(spec, seed, context)
            digest = hash_config(config)
            outcomes.append(
                RunOutcome(
                    spec=spec,
                    seed=seed,
                    run_id=f"{spec.system_id}_seed{seed}",
                    config=config,
                    config_hash=digest,
                    directory=run_directory(root, spec.system_id, seed, digest),
                )
            )
    return RunPlan(outcomes=tuple(outcomes), problems=tuple(validate_registry(specs)))


# ----------------------------------------------------------------------- execution ---


def _blocking_dependency(
    spec: SystemSpec, done: Mapping[str, bool], external: Iterable[str]
) -> str | None:
    """Find the first dependency that prevents a system from running.

    Args:
        spec: The system.
        done: System id to whether it succeeded.
        external: Artifact keys known to exist, e.g. ``encoder:gatv2``.

    Returns:
        The blocking dependency, or None when everything is satisfied.
    """
    available = set(external)
    for dependency in spec.depends_on:
        if ":" in dependency:
            if dependency not in available:
                return dependency
        elif not done.get(dependency, False):
            return dependency
    return None


def run_matrix(  # noqa: PLR0915 -- one scheduling loop; the branches ARE the statuses
    # (blocked, dry-run, skipped, no-executor, failed, completed) and splitting them
    # separates each outcome from the condition that produces it.
    plan: RunPlan,
    executors: Mapping[str, ExecutorFn],
    *,
    external_artifacts: Iterable[str] = (),
    force: bool = False,
    dry_run: bool = False,
    summary_path: Path | str | None = None,
) -> MatrixResult:
    """Execute a plan, isolating failures and skipping completed runs.

    Args:
        plan: The plan from :func:`plan_matrix`.
        executors: Executor name (the spec's ``executor`` value) to the callable that runs
            it. A spec whose executor has no entry is recorded as failed rather than
            raising, because "one arm has no implementation yet" must not abort fifteen
            that do.
        external_artifacts: Artifact keys that already exist, e.g. ``encoder:gatv2``.
            Dependencies naming one of these are satisfied; a system depending on a
            missing one is blocked, not attempted.
        force: Ignore completion markers and re-run everything.
        dry_run: Plan and report without invoking any executor. Every run is recorded
            PENDING.
        summary_path: Where to write the machine-readable summary, if anywhere.

    Returns:
        The result. Note that it is returned rather than raised even when runs failed --
        the caller decides the exit code, and the aggregator reports missing runs by name.

    Raises:
        ValueError: If the plan carries registry validation problems.
    """
    if plan.problems:
        raise ValueError(
            "refusing to execute a plan with registry problems: " + "; ".join(plan.problems)
        )

    started = datetime.now(UTC)
    t0 = time.monotonic()
    records: list[RunRecord] = []
    # A system counts as done when EVERY one of its seeds succeeded or was skipped. A
    # dependent reading a checkpoint from a partly-failed system would read whichever seed
    # happened to survive, which is not the arm the table says it is.
    seed_ok: dict[str, list[bool]] = {}
    external = set(external_artifacts)

    for outcome in plan.outcomes:
        spec, seed = outcome.spec, outcome.seed
        done = {sid: all(flags) for sid, flags in seed_ok.items()}
        blocker = _blocking_dependency(spec, done, external)
        if blocker is not None:
            log.warning("%s blocked by %s", outcome.run_id, blocker)
            records.append(
                RunRecord(
                    run_id=outcome.run_id,
                    system_id=spec.system_id,
                    seed=seed,
                    status=RunStatus.BLOCKED,
                    config_hash=outcome.config_hash,
                    directory=str(outcome.directory),
                    blocked_by=blocker,
                )
            )
            seed_ok.setdefault(spec.system_id, []).append(False)
            continue

        if dry_run:
            records.append(
                RunRecord(
                    run_id=outcome.run_id,
                    system_id=spec.system_id,
                    seed=seed,
                    status=RunStatus.PENDING,
                    config_hash=outcome.config_hash,
                    directory=str(outcome.directory),
                )
            )
            seed_ok.setdefault(spec.system_id, []).append(True)
            continue

        if not force and is_complete(outcome.directory, outcome.config_hash):
            log.info("%s already complete, skipping", outcome.run_id)
            marker = read_json(completion_marker_path(outcome.directory))
            payload = marker.get("payload", {}) if isinstance(marker, dict) else {}
            records.append(
                RunRecord(
                    run_id=outcome.run_id,
                    system_id=spec.system_id,
                    seed=seed,
                    status=RunStatus.SKIPPED,
                    config_hash=outcome.config_hash,
                    directory=str(outcome.directory),
                    payload=payload if isinstance(payload, dict) else {},
                )
            )
            seed_ok.setdefault(spec.system_id, []).append(True)
            continue

        executor = executors.get(str(spec.executor))
        if executor is None:
            message = f"no executor registered for {spec.executor}"
            log.error("%s: %s", outcome.run_id, message)
            records.append(
                RunRecord(
                    run_id=outcome.run_id,
                    system_id=spec.system_id,
                    seed=seed,
                    status=RunStatus.FAILED,
                    config_hash=outcome.config_hash,
                    directory=str(outcome.directory),
                    error=message,
                )
            )
            seed_ok.setdefault(spec.system_id, []).append(False)
            continue

        outcome.directory.mkdir(parents=True, exist_ok=True)
        write_json(outcome.directory / RESOLVED_CONFIG, dict(outcome.config))

        log.info(
            "running %s (%s, %s) -> %s",
            outcome.run_id,
            spec.executor,
            spec.resource,
            outcome.directory,
        )
        run_started = datetime.now(UTC).isoformat()
        run_t0 = time.monotonic()
        try:
            payload = dict(executor(spec, seed, outcome.directory))
        except Exception as exc:
            elapsed = time.monotonic() - run_t0
            log.exception("%s FAILED after %.1fs", outcome.run_id, elapsed)
            records.append(
                RunRecord(
                    run_id=outcome.run_id,
                    system_id=spec.system_id,
                    seed=seed,
                    status=RunStatus.FAILED,
                    config_hash=outcome.config_hash,
                    directory=str(outcome.directory),
                    started_utc=run_started,
                    duration_s=elapsed,
                    error=repr(exc),
                    traceback=traceback.format_exc(),
                )
            )
            write_json(
                outcome.directory / "FAILED.json",
                {
                    "run_id": outcome.run_id,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "failed_utc": datetime.now(UTC).isoformat(),
                },
            )
            seed_ok.setdefault(spec.system_id, []).append(False)
            continue

        elapsed = time.monotonic() - run_t0
        write_completion_marker(
            outcome.directory,
            run_id=outcome.run_id,
            system_id=spec.system_id,
            seed=seed,
            config_hash=outcome.config_hash,
            payload=payload,
            duration_s=elapsed,
        )
        log.info("%s completed in %.1fs", outcome.run_id, elapsed)
        records.append(
            RunRecord(
                run_id=outcome.run_id,
                system_id=spec.system_id,
                seed=seed,
                status=RunStatus.COMPLETED,
                config_hash=outcome.config_hash,
                directory=str(outcome.directory),
                started_utc=run_started,
                duration_s=elapsed,
                payload=payload,
            )
        )
        seed_ok.setdefault(spec.system_id, []).append(True)

    result = MatrixResult(
        records=tuple(records),
        started_utc=started.isoformat(),
        duration_s=time.monotonic() - t0,
    )
    if summary_path is not None:
        write_json(summary_path, result.to_dict())
    return result
