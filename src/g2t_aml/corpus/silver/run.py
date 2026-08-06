"""Drive a whole Silver build: concurrency, checkpointing, resume, halting.

Separated from :mod:`g2t_aml.corpus.silver.generate` because the per-case loop and the
orchestration around it fail in different ways and are tested differently. ``generate_one``
is pure given a teacher; this module owns the parts that make a multi-hour job survivable —
a bounded worker pool, an append-only record stream, a checkpoint that lets a killed run
resume without gaps or duplicates, and a budget stop that ends the run cleanly instead of
turning every remaining case into a discard.

**Resume correctness is the subtle requirement.** A case is checkpointed when it reaches a
terminal state — written *or* discarded — and records are appended as they complete rather
than accumulated and written at the end. The two together are what make "kill it and start
it again" produce the same corpus as an uninterrupted run: nothing is recomputed, and
nothing is lost between the last append and the kill except at most one partially written
line, which the reader drops.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from g2t_aml.corpus.silver.api_client import (
    BudgetExceeded,
    CheckpointStore,
    CostTracker,
    Teacher,
)
from g2t_aml.corpus.silver.generate import (
    CaseInput,
    DiscardRecord,
    GenerationOutcome,
    SilverConfig,
    generate_one,
)
from g2t_aml.corpus.tokenization import TokenCounter
from g2t_aml.facts.vocab import ControlledVocabulary
from g2t_aml.utils.logging import get_logger

__all__ = ["RunResult", "JSONLAppender", "run_generation"]

log = get_logger(__name__)


class JSONLAppender:
    """Append-only JSONL writer, safe across worker threads.

    Appends rather than buffering because the whole point is that a killed job keeps what
    it has already produced. Flushed per record: a run that loses an hour of work to an
    unflushed buffer has defeated the checkpoint it was writing alongside.
    """

    def __init__(self, path: Path | None) -> None:
        """Open the file for appending.

        Args:
            path: Destination JSONL, or None to discard writes — which is what a dry run
                uses, so the dry-run path exercises the same code as a real one.
        """
        self.path = Path(path) if path is not None else None
        self.n_written = 0
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, payload: dict[str, Any]) -> None:
        """Write one record.

        Args:
            payload: A JSON-serialisable mapping.
        """
        line = json.dumps(payload, sort_keys=True)
        with self._lock:
            self.n_written += 1
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
                    handle.flush()

    @staticmethod
    def read(path: Path) -> list[dict[str, Any]]:
        """Read a JSONL file, tolerating a truncated final line.

        A job killed mid-write leaves one incomplete line. Refusing to read the file at
        all would make a crash unrecoverable; silently dropping every malformed line
        would hide corruption. Only a malformed *last* line is dropped, and anything else
        raises.

        Args:
            path: The file.

        Returns:
            The records.

        Raises:
            json.JSONDecodeError: If a line other than the last is malformed.
        """
        if not path.is_file():
            return []
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        records: list[dict[str, Any]] = []
        for index, line in enumerate(lines):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    log.warning("dropping truncated final line of %s (killed mid-write)", path)
                    break
                raise
        return records


@dataclass
class RunResult:
    """Everything a build produced.

    Attributes:
        records: Accepted training records, serialised.
        discards: Every discarded case.
        outcomes: Per-case outcomes, for the usage and attempt reports.
        n_attempted: Cases this run actually called a teacher for.
        n_resumed: Cases skipped because a previous run had finished them.
        halted: Whether the run stopped early on the budget cap.
        halt_reason: Why, when it did.
    """

    records: list[dict[str, Any]] = field(default_factory=list)
    discards: list[DiscardRecord] = field(default_factory=list)
    outcomes: list[GenerationOutcome] = field(default_factory=list)
    n_attempted: int = 0
    n_resumed: int = 0
    halted: bool = False
    halt_reason: str = ""

    def attempt_histogram(self) -> dict[str, int]:
        """Return how many cases needed zero, one or two repairs.

        Returns:
            Attempt count to number of *accepted* cases. The share needing repair at all
            is the number the budget estimate turned on, and it is worth reporting
            against the ~20% that was assumed.
        """
        histogram: dict[str, int] = {}
        for outcome in self.outcomes:
            if outcome.accepted:
                key = str(outcome.attempts)
                histogram[key] = histogram.get(key, 0) + 1
        return dict(sorted(histogram.items()))


def run_generation(
    cases: Sequence[CaseInput],
    assignment: dict[str, str],
    teachers: dict[str, Teacher],
    *,
    vocabulary: ControlledVocabulary,
    config: SilverConfig,
    token_counter: TokenCounter,
    graph_ref_for: Callable[[str], str],
    tracker: CostTracker,
    checkpoint: CheckpointStore | None = None,
    records_path: Path | None = None,
    discards_path: Path | None = None,
    concurrency: int = 8,
    progress_every: int = 250,
) -> RunResult:
    """Generate, verify and write every case.

    Args:
        cases: The cases to build.
        assignment: Case id to teacher key.
        teachers: Teacher key to client.
        vocabulary: The controlled vocabulary.
        config: Thresholds.
        token_counter: Counts tokens for the length block.
        graph_ref_for: Case id to its ``<case store>#<case_id>`` reference.
        tracker: Cost tracker, already wired to the teachers.
        checkpoint: Completed-case store. When given, finished cases are skipped and new
            ones marked.
        records_path: Where accepted records are appended. None discards them, for a dry
            run.
        discards_path: Where discards are appended.
        concurrency: Worker threads.
        progress_every: Log a progress line every this many completions.

    Returns:
        The run result.
    """
    pending = [c for c in cases if checkpoint is None or c.case_id not in checkpoint]
    resumed = len(cases) - len(pending)
    if resumed:
        log.info("resuming: %d of %d cases were already finished", resumed, len(cases))

    result = RunResult(n_resumed=resumed)
    records = JSONLAppender(records_path)
    discards = JSONLAppender(discards_path)
    lock = threading.Lock()
    stop = threading.Event()

    def work(case: CaseInput) -> GenerationOutcome | None:
        if stop.is_set():
            return None
        teacher = teachers[assignment[case.case_id]]
        return generate_one(
            case,
            teacher,
            vocabulary=vocabulary,
            config=config,
            graph_ref=graph_ref_for(case.case_id),
            token_counter=token_counter,
        )

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(work, case): case for case in pending}
        completed = 0
        for future in as_completed(futures):
            case = futures[future]
            try:
                outcome = future.result()
            except BudgetExceeded as exc:
                # Halt the whole run. Every case still in flight or queued is left
                # untouched and un-checkpointed, so a resumed run picks up exactly where
                # this one stopped -- and, critically, none of them lands in the discard
                # log, where it would be miscounted as a model failure.
                if not stop.is_set():
                    stop.set()
                    result.halted = True
                    result.halt_reason = str(exc)
                    log.error("budget cap reached, halting: %s", exc)
                continue
            if outcome is None:
                continue

            with lock:
                completed += 1
                result.outcomes.append(outcome)
                result.n_attempted += 1
                if outcome.record is not None:
                    records.append(outcome.record.to_dict())
                elif outcome.discard is not None:
                    result.discards.append(outcome.discard)
                    discards.append(outcome.discard.to_dict())
                if checkpoint is not None:
                    checkpoint.mark(case.case_id)
                if completed % progress_every == 0:
                    log.info(
                        "  %d / %d cases; %d written, %d discarded, $%.2f spent",
                        completed,
                        len(pending),
                        records.n_written,
                        len(result.discards),
                        tracker.total_usd,
                    )

    if records_path is not None:
        result.records = JSONLAppender.read(records_path)
    else:
        result.records = [o.record.to_dict() for o in result.outcomes if o.record is not None]
    return result
