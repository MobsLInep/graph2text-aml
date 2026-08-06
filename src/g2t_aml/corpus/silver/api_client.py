"""Teacher clients: caching, retry, concurrency, cost control, checkpointing.

Production-grade because this job runs for hours across two providers and roughly 24,000
calls, and an API failure at hour four is expensive in a way a unit test never shows. Five
things this module exists to guarantee:

- **A re-run never re-pays.** Responses are cached on disk, content-addressed on
  ``(prompt_hash, model, temperature, top_p, seed, case_id, kind, attempt)``. Change any
  component and the key changes; change none and the call is free.
- **A transient failure is retried and a permanent one is not.** A 429 or a 503 is backed
  off with jitter; a 400 or a content refusal is recorded against the case and the job
  moves on. Retrying a permanent failure burns the budget on a call that cannot succeed.
- **The budget is a hard stop.** A running total is kept across threads, and the cap halts
  the job rather than warning about it.
- **An interrupted run resumes.** Completed case ids are checkpointed, so a killed job
  restarts without regenerating what it already wrote.
- **Every failure is on the record.** The structured error log is an output of the run.

**Sampling parameters are a per-teacher capability, not a global constant.** Every current
frontier Anthropic model — Opus 5, Sonnet 5, Opus 4.8, 4.7 — *rejects* ``temperature`` and
``top_p`` with a 400 rather than ignoring them, so a spec whose model cannot accept them
carries ``supports_sampling=False`` and the fields are omitted from the request and
recorded as null. Sending them anyway would fail every call in the run. Surface diversity
for those teachers comes from the per-case style directive in
:mod:`g2t_aml.corpus.silver.prompts` instead, which is deterministic and recorded. See
DECISIONS.md D-045.

**Nothing here imports a provider SDK at module scope.** The core dependency set is
CPU-only and provider-free (CLAUDE.md §4); ``anthropic`` lives in the ``api`` extra and is
imported inside the backend that needs it, so the tests — which drive the whole pipeline
through :class:`ScriptedTeacher` — run in the base environment.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from g2t_aml.corpus.silver.prompts import RenderedPrompt
from g2t_aml.utils.hashing import canonical_json
from g2t_aml.utils.io import atomic_path
from g2t_aml.utils.logging import get_logger

__all__ = [
    "BudgetExceeded",
    "BudgetGuard",
    "CheckpointStore",
    "CostTracker",
    "ErrorLog",
    "ErrorRecord",
    "PermanentAPIError",
    "ResponseCache",
    "RetryPolicy",
    "ScriptedTeacher",
    "Teacher",
    "TeacherError",
    "TeacherResponse",
    "TeacherSpec",
    "TransientAPIError",
    "build_teacher",
    "cache_key_components",
    "preflight",
    "specs_from_config",
]

log = get_logger(__name__)

#: HTTP statuses worth retrying. 408 timeout, 409 conflict, 429 rate limit, and the 5xx
#: family including 529 overloaded. Everything else is the caller's fault and will fail
#: identically on the next attempt.
TRANSIENT_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

#: The two-teacher requirement, as a number the code can check. Named because it is a
#: methodological constraint rather than an arbitrary minimum: one teacher is
#: distillation from one model.
MIN_TEACHERS = 2


class TeacherError(RuntimeError):
    """Base class for a failed teacher call."""


class TransientAPIError(TeacherError):
    """A failure that may succeed on retry: rate limit, overload, timeout, 5xx."""


class PermanentAPIError(TeacherError):
    """A failure that will not succeed on retry: malformed request, auth, refusal.

    Recorded against the case and skipped. Retrying spends budget on a call that cannot
    succeed, and — for a content refusal — obscures a result worth reporting.
    """


class BudgetExceeded(RuntimeError):  # noqa: N818 -- reads as the condition at the call
    # site (`except BudgetExceeded`), and a "BudgetExceededError" suffix would suggest a
    # failure rather than the deliberate stop this is.
    """Raised when the running cost would exceed the configured cap.

    Deliberately not a :class:`TeacherError`: it is not a property of one call, and it
    must halt the job rather than be absorbed by a per-case error handler.
    """


@dataclass(frozen=True)
class TeacherSpec:
    """One teacher, frozen exactly as it will be recorded on every generated record.

    Attributes:
        key: Stable identifier used in provenance, discard logs and balance reports.
        family: ``"frontier"`` or ``"open_weights"``. The two-teacher requirement is that
            these differ, and the balance report keys on it.
        provider: ``"anthropic"`` or ``"openai_compatible"``.
        model: The exact model string sent to the API.
        max_output_tokens: Response cap. On a model with thinking enabled this bounds
            thinking *and* text together, so it is set well above the target narrative
            length rather than snugly around it.
        supports_sampling: Whether the model accepts ``temperature`` / ``top_p``. False
            for every current frontier Anthropic model, which reject them with a 400.
        temperature: Sampling temperature, or None. Omitted from the request when
            ``supports_sampling`` is False, and recorded as null.
        top_p: Nucleus sampling parameter, on the same terms.
        seed: Provider seed, where supported. Recorded either way, because "the provider
            does not support a seed" is itself a reproducibility fact.
        effort: Anthropic effort level, or None. The depth control that replaces sampling
            parameters on models that reject them.
        thinking: Anthropic thinking mode. ``"adaptive"`` by default rather than
            ``"disabled"`` — disabling it on the current Opus family risks internal tags
            leaking into the visible response, and a leaked tag lands directly in a corpus
            narrative.
        api_key_env: Environment variable holding the credential. Never the key itself:
            a spec is serialised into every record and into the run config.
        base_url: Override for the provider endpoint. Required for an open-weights server.
        price_in_per_mtok: USD per million input tokens, for the cost report.
        price_out_per_mtok: USD per million output tokens.
        price_cache_read_per_mtok: USD per million cached input tokens, when the provider
            prices them separately.
        request_timeout_s: Per-request timeout.
    """

    key: str
    family: str
    provider: str
    model: str
    max_output_tokens: int = 4096
    supports_sampling: bool = False
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    effort: str | None = None
    thinking: str | None = "adaptive"
    api_key_env: str = "ANTHROPIC_API_KEY"
    base_url: str | None = None
    price_in_per_mtok: float = 0.0
    price_out_per_mtok: float = 0.0
    price_cache_read_per_mtok: float = 0.0
    request_timeout_s: float = 300.0

    def sampling_provenance(self) -> dict[str, Any]:
        """Return the decoding settings exactly as they should be recorded.

        The provider seed is ``provider_seed``, never ``seed``. ``seed`` on a training
        record is the run's global seed (invariant 5), which every tier records and which
        is an integer by schema; the provider seed is a different fact and is legitimately
        null on a provider that offers none. Collapsing the two would either lie about the
        run seed or claim a decoding seed that was never sent.

        Returns:
            Temperature, top_p, provider seed, effort and thinking, with the sampling
            fields null when the model does not accept them, plus the reason they are
            null. A record that simply omitted them would be indistinguishable from one
            generated before the fields existed.
        """
        return {
            "temperature": self.temperature if self.supports_sampling else None,
            "top_p": self.top_p if self.supports_sampling else None,
            "provider_seed": self.seed,
            "effort": self.effort,
            "thinking": self.thinking,
            "sampling_parameters_supported": self.supports_sampling,
            "sampling_omitted_reason": (
                None if self.supports_sampling else "model rejects temperature/top_p"
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the spec as recorded provenance.

        Returns:
            Identity, model, decoding settings and prices. ``api_key_env`` is included —
            it names a variable, never a secret — so a run can be attributed to a
            credential source.
        """
        return {
            "teacher": self.key,
            "family": self.family,
            "provider": self.provider,
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "api_key_env": self.api_key_env,
            "base_url": self.base_url,
            **self.sampling_provenance(),
        }


@dataclass(frozen=True)
class TeacherResponse:
    """One completion, with everything the cost report and the record need.

    Attributes:
        text: The generated narrative, stripped.
        model_served: The model that actually produced the text. Distinct from the
            requested model because a provider may substitute one; a record must say what
            wrote it, not what was asked for.
        input_tokens: Uncached prompt tokens.
        output_tokens: Generated tokens.
        cache_read_tokens: Prompt tokens served from the provider's prompt cache.
        cache_write_tokens: Prompt tokens written to the provider's prompt cache.
        cost_usd: Cost of this call under the spec's prices. Zero on a local cache hit.
        from_cache: Whether this came from the on-disk response cache.
        latency_s: Wall-clock seconds. Zero on a cache hit.
        stop_reason: Provider stop reason.
        request_id: Provider request id, for support escalation.
        attempts: How many HTTP attempts this call took, including the successful one.
    """

    text: str
    model_served: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    from_cache: bool = False
    latency_s: float = 0.0
    stop_reason: str = ""
    request_id: str = ""
    attempts: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Return the response as a JSON-serialisable mapping.

        Returns:
            Every field, for the cache file and the usage block on a record.
        """
        return {
            "text": self.text,
            "model_served": self.model_served,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cost_usd": round(self.cost_usd, 8),
            "stop_reason": self.stop_reason,
            "request_id": self.request_id,
            "attempts": self.attempts,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, from_cache: bool) -> TeacherResponse:
        """Rebuild a response from its cached form.

        Args:
            payload: The mapping written by :meth:`to_dict`.
            from_cache: Marks the rebuilt response as a cache hit.

        Returns:
            The response, with ``cost_usd`` zeroed when it came from the cache — a cached
            call costs nothing, and carrying the original cost forward would make the cost
            report count the same spend on every re-run.
        """
        return cls(
            text=str(payload["text"]),
            model_served=str(payload.get("model_served", "")),
            input_tokens=int(payload.get("input_tokens", 0)),
            output_tokens=int(payload.get("output_tokens", 0)),
            cache_read_tokens=int(payload.get("cache_read_tokens", 0)),
            cache_write_tokens=int(payload.get("cache_write_tokens", 0)),
            cost_usd=0.0 if from_cache else float(payload.get("cost_usd", 0.0)),
            from_cache=from_cache,
            stop_reason=str(payload.get("stop_reason", "")),
            request_id=str(payload.get("request_id", "")),
            attempts=int(payload.get("attempts", 1)),
        )


def cache_key_components(
    spec: TeacherSpec, prompt: RenderedPrompt, case_id: str, kind: str, attempt: int
) -> dict[str, Any]:
    """Return every component of a cache key, as a mapping.

    Exposed rather than inlined so a test can assert that changing any one component
    changes the key — the property that makes the cache safe to trust across a prompt
    edit or a model swap.

    Args:
        spec: The teacher.
        prompt: The rendered prompt.
        case_id: The case.
        kind: ``"rewrite"`` or ``"repair"``.
        attempt: Repair attempt index; 0 for the initial rewrite.

    Returns:
        The key components. ``rendered_prompt_hash`` covers the prompt text in full, so a
        changed fact record or Bronze draft misses the cache even when everything named
        here is identical.
    """
    return {
        "prompt_hash": prompt.prompt_hash,
        "rendered_prompt_hash": prompt.rendered_hash,
        "model": spec.model,
        "provider": spec.provider,
        "temperature": spec.temperature if spec.supports_sampling else None,
        "top_p": spec.top_p if spec.supports_sampling else None,
        "seed": spec.seed,
        "effort": spec.effort,
        "thinking": spec.thinking,
        "max_output_tokens": spec.max_output_tokens,
        "case_id": case_id,
        "kind": kind,
        "attempt": attempt,
    }


class ResponseCache:
    """Content-addressed on-disk cache of teacher responses.

    Keyed on the full request identity, so a re-run after a crash costs nothing and a
    prompt edit correctly invalidates everything it touched.
    """

    def __init__(self, root: Path, *, enabled: bool = True) -> None:
        """Open a cache rooted at a directory.

        Args:
            root: Directory to store entries under. Created on first write.
            enabled: When False every lookup misses and nothing is written. For a run
                that must genuinely re-call the API.
        """
        self.root = Path(root)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    @staticmethod
    def digest(components: dict[str, Any]) -> str:
        """Return the cache digest for a set of key components.

        Args:
            components: From :func:`cache_key_components`.

        Returns:
            A hex SHA-256 over the canonical JSON of the components, so key ordering
            cannot produce two digests for one request.
        """
        import hashlib

        return hashlib.sha256(canonical_json(components).encode("utf-8")).hexdigest()

    def path_for(self, digest: str) -> Path:
        """Return the file a digest maps to.

        Args:
            digest: The cache digest.

        Returns:
            A path fanned out by the first two hex characters, so no directory holds
            tens of thousands of entries.
        """
        return self.root / digest[:2] / f"{digest}.json"

    def get(self, digest: str) -> TeacherResponse | None:
        """Look a response up.

        Args:
            digest: The cache digest.

        Returns:
            The cached response, or None on a miss. A corrupt entry — a half-written file
            from a killed job — counts as a miss rather than raising.
        """
        if not self.enabled:
            return None
        path = self.path_for(digest)
        if not path.is_file():
            with self._lock:
                self.misses += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            log.warning("discarding corrupt cache entry %s", path)
            with self._lock:
                self.misses += 1
            return None
        with self._lock:
            self.hits += 1
        return TeacherResponse.from_dict(payload["response"], from_cache=True)

    def put(self, digest: str, components: dict[str, Any], response: TeacherResponse) -> None:
        """Store a response.

        The key components are written alongside the response so a cache directory can be
        audited — and a digest collision or a key-schema change diagnosed — without
        rerunning the job that produced it.

        Args:
            digest: The cache digest.
            components: The key components.
            response: The response to store.
        """
        if not self.enabled:
            return
        path = self.path_for(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "key": components,
            "digest": digest,
            "stored_at": datetime.now(UTC).isoformat(),
            "response": response.to_dict(),
        }
        with atomic_path(path) as tmp:
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def stats(self) -> dict[str, Any]:
        """Return hit and miss counts.

        Returns:
            Counts and the hit rate, for the cost report.
        """
        total = self.hits + self.misses
        return {
            "enabled": self.enabled,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 6) if total else 0.0,
        }


class CostTracker:
    """Running spend, per teacher and in total. Thread-safe."""

    def __init__(self) -> None:
        """Start an empty tracker."""
        self._lock = threading.Lock()
        self.total_usd = 0.0
        self.by_teacher: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self.tokens_in: dict[str, int] = {}
        self.tokens_out: dict[str, int] = {}
        self.cached_calls: dict[str, int] = {}

    def record(self, teacher: str, response: TeacherResponse) -> float:
        """Add one call to the running totals.

        Args:
            teacher: The teacher key.
            response: The completed call.

        Returns:
            The new grand total in USD.
        """
        with self._lock:
            self.total_usd += response.cost_usd
            self.by_teacher[teacher] = self.by_teacher.get(teacher, 0.0) + response.cost_usd
            self.calls[teacher] = self.calls.get(teacher, 0) + 1
            self.tokens_in[teacher] = self.tokens_in.get(teacher, 0) + response.input_tokens
            self.tokens_out[teacher] = self.tokens_out.get(teacher, 0) + response.output_tokens
            if response.from_cache:
                self.cached_calls[teacher] = self.cached_calls.get(teacher, 0) + 1
            return self.total_usd

    def to_dict(self) -> dict[str, Any]:
        """Return the cost report.

        Returns:
            Totals and per-teacher breakdowns, rounded for reporting.
        """
        with self._lock:
            return {
                "total_usd": round(self.total_usd, 4),
                "by_teacher_usd": {k: round(v, 4) for k, v in sorted(self.by_teacher.items())},
                "calls": dict(sorted(self.calls.items())),
                "calls_served_from_cache": dict(sorted(self.cached_calls.items())),
                "input_tokens": dict(sorted(self.tokens_in.items())),
                "output_tokens": dict(sorted(self.tokens_out.items())),
            }


class BudgetGuard:
    """A hard spending cap that halts the job.

    A warning is not a control. A run that quietly passes its cap has already spent the
    money by the time anyone reads the log, so this raises and the job stops.
    """

    def __init__(self, cap_usd: float, tracker: CostTracker) -> None:
        """Bind a cap to a tracker.

        Args:
            cap_usd: The ceiling. Zero or negative means no cap, for a run made entirely
                of cache hits.
            tracker: The running total to check against.
        """
        self.cap_usd = cap_usd
        self.tracker = tracker

    @property
    def enabled(self) -> bool:
        """Report whether a cap is in force.

        Returns:
            True when the cap is positive.
        """
        return self.cap_usd > 0

    def check(self) -> None:
        """Raise if the cap has already been reached.

        Called before each call is dispatched, so the halt happens before more money is
        committed rather than after.

        Raises:
            BudgetExceeded: When the running total has reached the cap.
        """
        if self.enabled and self.tracker.total_usd >= self.cap_usd:
            raise BudgetExceeded(
                f"spend ${self.tracker.total_usd:.2f} has reached the cap "
                f"${self.cap_usd:.2f}; halting. Raise corpus.budget.cap_usd deliberately "
                "or resume the run once the cap is reviewed."
            )


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with full jitter.

    Attributes:
        max_attempts: Total HTTP attempts per call, including the first.
        base_delay_s: Delay before the second attempt.
        max_delay_s: Ceiling on any single delay.
        jitter: When True the delay is drawn uniformly from ``[0, computed]`` — full
            jitter rather than a fixed backoff, because a rate-limited run whose workers
            all sleep the same duration retries in a thundering herd and gets rate-limited
            again in lockstep.
    """

    max_attempts: int = 5
    base_delay_s: float = 1.0
    max_delay_s: float = 60.0
    jitter: bool = True

    def delay_for(self, attempt: int, retry_after: float | None = None) -> float:
        """Return the delay before the next attempt.

        Args:
            attempt: 1-based index of the attempt that just failed.
            retry_after: The provider's ``retry-after`` in seconds, when it sent one. It
                wins over the computed backoff: the server knows when it will accept
                traffic again and guessing shorter just wastes an attempt.

        Returns:
            Seconds to sleep.
        """
        if retry_after is not None and retry_after > 0:
            return min(retry_after, self.max_delay_s)
        delay = min(self.base_delay_s * (2 ** (attempt - 1)), self.max_delay_s)
        return random.uniform(0, delay) if self.jitter else delay


@dataclass
class ErrorRecord:
    """One failed call, classified.

    Attributes:
        case_id: The case.
        teacher: Teacher key.
        kind: ``"rewrite"`` or ``"repair"``.
        attempt: Repair attempt index.
        classification: ``"transient"`` or ``"permanent"``.
        error_type: Exception class name.
        message: The message, truncated.
        timestamp: When it happened.
    """

    case_id: str
    teacher: str
    kind: str
    attempt: int
    classification: str
    error_type: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a JSONL row.

        Returns:
            The mapping written to the error log.
        """
        return {
            "case_id": self.case_id,
            "teacher": self.teacher,
            "kind": self.kind,
            "attempt": self.attempt,
            "classification": self.classification,
            "error_type": self.error_type,
            "message": self.message[:500],
            "timestamp": self.timestamp,
        }


class ErrorLog:
    """Structured, append-only log of failed calls.

    An output of the run, not a debug artifact: "the frontier model refused 41 cases on
    content grounds" is a reportable fact about generating AML narratives with a
    commercial model, and it is invisible if failures are only counted.
    """

    def __init__(self, path: Path | None = None) -> None:
        """Open a log.

        Args:
            path: Destination JSONL. When None the log is held in memory only, which is
                what the tests use.
        """
        self.path = Path(path) if path is not None else None
        self.records: list[ErrorRecord] = []
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, record: ErrorRecord) -> None:
        """Append one failure.

        Args:
            record: The failure.
        """
        with self._lock:
            self.records.append(record)
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")

    def summary(self) -> dict[str, Any]:
        """Return counts by classification, teacher and error type.

        Returns:
            A JSON-serialisable summary for the run report.
        """
        by_class: dict[str, int] = {}
        by_teacher: dict[str, int] = {}
        by_type: dict[str, int] = {}
        for record in self.records:
            by_class[record.classification] = by_class.get(record.classification, 0) + 1
            by_teacher[record.teacher] = by_teacher.get(record.teacher, 0) + 1
            by_type[record.error_type] = by_type.get(record.error_type, 0) + 1
        return {
            "n_errors": len(self.records),
            "by_classification": dict(sorted(by_class.items())),
            "by_teacher": dict(sorted(by_teacher.items())),
            "by_error_type": dict(sorted(by_type.items())),
        }


class CheckpointStore:
    """Which cases a run has already finished.

    Written as a line-per-case-id file rather than a rewritten JSON document: an append is
    atomic enough that a job killed mid-write loses at most the last line, whereas a
    truncated rewrite of a full manifest loses everything.
    """

    def __init__(self, path: Path | None) -> None:
        """Open or create a checkpoint file.

        Args:
            path: Destination. When None, checkpointing is disabled and nothing resumes.
        """
        self.path = Path(path) if path is not None else None
        self._done: set[str] = set()
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.is_file():
                self._done = {
                    line.strip()
                    for line in self.path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                }

    def __contains__(self, case_id: str) -> bool:
        """Report whether a case is already done.

        Args:
            case_id: The case.

        Returns:
            True when the case was completed by this or an earlier run.
        """
        return case_id in self._done

    def __len__(self) -> int:
        """Return how many cases are checkpointed.

        Returns:
            The count.
        """
        return len(self._done)

    def mark(self, case_id: str) -> None:
        """Record a case as finished.

        Args:
            case_id: The case, written or discarded — both are terminal outcomes and
                neither should be recomputed on resume.
        """
        with self._lock:
            if case_id in self._done:
                return
            self._done.add(case_id)
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(case_id + "\n")

    def done(self) -> frozenset[str]:
        """Return every completed case id.

        Returns:
            The set, frozen.
        """
        return frozenset(self._done)


class Teacher(Protocol):
    """What the generation loop needs from a teacher.

    Deliberately narrow: one method and a spec. :class:`ScriptedTeacher` and
    :class:`APITeacher` both satisfy it, which is what lets the whole
    generate→verify→repair→discard loop be tested without a network.
    """

    spec: TeacherSpec

    def complete(
        self, prompt: RenderedPrompt, *, case_id: str, kind: str, attempt: int = 0
    ) -> TeacherResponse:
        """Generate one completion.

        Args:
            prompt: The rendered prompt.
            case_id: The case, for cache keying and error attribution.
            kind: ``"rewrite"`` or ``"repair"``.
            attempt: Repair attempt index; 0 for the initial rewrite.

        Returns:
            The completion.

        Raises:
            TransientAPIError: On a retryable failure that outlived the retry policy.
            PermanentAPIError: On a failure that will not succeed on retry.
            BudgetExceeded: When the cap has been reached.
        """
        ...


class ScriptedTeacher:
    """A teacher driven by a callable, for tests and dry runs.

    Not a mock bolted on beside the real client but an implementation of the same
    protocol, so the loop under test is the loop that runs in production. The callable
    receives the same arguments the API backend would and returns narrative text, which
    is what lets a test script "clean pass", "repair once then pass" and "fail twice then
    discard" precisely.
    """

    def __init__(
        self,
        spec: TeacherSpec,
        responder: Callable[[RenderedPrompt, str, str, int], str],
        *,
        tracker: CostTracker | None = None,
    ) -> None:
        """Bind a spec to a responder.

        Args:
            spec: The teacher identity recorded on generated records.
            responder: ``(prompt, case_id, kind, attempt) -> narrative text``. Raising a
                :class:`TeacherError` from it exercises the error paths.
            tracker: Cost tracker to report into. Costs are computed from the spec's
                prices over a crude token estimate, so a dry run still produces a
                projection rather than zero.
        """
        self.spec = spec
        self._responder = responder
        self._tracker = tracker
        self.calls: list[tuple[str, str, int]] = []

    def complete(
        self, prompt: RenderedPrompt, *, case_id: str, kind: str, attempt: int = 0
    ) -> TeacherResponse:
        """Generate one scripted completion.

        Args:
            prompt: The rendered prompt.
            case_id: The case.
            kind: ``"rewrite"`` or ``"repair"``.
            attempt: Repair attempt index.

        Returns:
            The completion, with estimated token counts and cost.

        Raises:
            TeacherError: Whatever the responder raises.
        """
        self.calls.append((case_id, kind, attempt))
        text = self._responder(prompt, case_id, kind, attempt)
        n_in = _estimate_tokens(prompt.system) + _estimate_tokens(prompt.user)
        n_out = _estimate_tokens(text)
        response = TeacherResponse(
            text=text.strip(),
            model_served=self.spec.model,
            input_tokens=n_in,
            output_tokens=n_out,
            cost_usd=_price(self.spec, n_in, n_out, 0),
            stop_reason="end_turn",
        )
        if self._tracker is not None:
            self._tracker.record(self.spec.key, response)
        return response


def _estimate_tokens(text: str) -> int:
    """Estimate a token count without a tokenizer.

    Used only for cost *projection* in dry runs and scripted teachers; a real call reports
    the provider's own usage numbers and never consults this.

    Args:
        text: The text to measure.

    Returns:
        A four-characters-per-token approximation, floored at one for non-empty text.
    """
    return max(1, len(text) // 4) if text else 0


def _price(spec: TeacherSpec, n_in: int, n_out: int, n_cached: int) -> float:
    """Compute the USD cost of one call.

    Args:
        spec: The teacher, carrying the prices.
        n_in: Uncached input tokens.
        n_out: Output tokens.
        n_cached: Input tokens served from the provider's prompt cache.

    Returns:
        The cost in USD.
    """
    million = 1_000_000
    return (
        n_in * spec.price_in_per_mtok / million
        + n_out * spec.price_out_per_mtok / million
        + n_cached * spec.price_cache_read_per_mtok / million
    )


class APITeacher:
    """A teacher backed by a real provider, with cache, retry, budget and concurrency.

    The order of operations matters and is asserted by the tests: **cache first, then
    budget, then call.** Checking the budget before the cache would halt a resumed run
    that was going to spend nothing at all.
    """

    def __init__(
        self,
        spec: TeacherSpec,
        *,
        cache: ResponseCache,
        tracker: CostTracker,
        budget: BudgetGuard,
        errors: ErrorLog,
        retry: RetryPolicy | None = None,
        semaphore: threading.Semaphore | None = None,
        sleep: Callable[[float], None] = time.sleep,
        backend: Callable[[TeacherSpec, RenderedPrompt], TeacherResponse] | None = None,
    ) -> None:
        """Assemble a client.

        Args:
            spec: The teacher.
            cache: On-disk response cache.
            tracker: Running cost.
            budget: The hard cap.
            errors: Structured error log.
            retry: Backoff policy. Defaults to :class:`RetryPolicy`.
            semaphore: Concurrency limiter, shared across teachers when the limit is a
                property of the machine rather than of the provider.
            sleep: Injected so a retry test does not actually wait.
            backend: The transport. Defaults to the provider named by the spec; injected
                in tests to drive failure sequences.
        """
        self.spec = spec
        self.cache = cache
        self.tracker = tracker
        self.budget = budget
        self.errors = errors
        self.retry = retry if retry is not None else RetryPolicy()
        self.semaphore = semaphore
        self._sleep = sleep
        self._backend = backend if backend is not None else _backend_for(spec)

    def complete(
        self, prompt: RenderedPrompt, *, case_id: str, kind: str, attempt: int = 0
    ) -> TeacherResponse:
        """Generate one completion, serving from cache when possible.

        Args:
            prompt: The rendered prompt.
            case_id: The case.
            kind: ``"rewrite"`` or ``"repair"``.
            attempt: Repair attempt index.

        Returns:
            The completion.

        Raises:
            TransientAPIError: When every retry was exhausted.
            PermanentAPIError: On a non-retryable failure.
            BudgetExceeded: When the cap has been reached.
        """
        components = cache_key_components(self.spec, prompt, case_id, kind, attempt)
        digest = ResponseCache.digest(components)
        if (cached := self.cache.get(digest)) is not None:
            self.tracker.record(self.spec.key, cached)
            return cached

        self.budget.check()
        response = self._call_with_retry(prompt, case_id, kind, attempt)
        self.cache.put(digest, components, response)
        self.tracker.record(self.spec.key, response)
        return response

    def _call_with_retry(
        self, prompt: RenderedPrompt, case_id: str, kind: str, attempt: int
    ) -> TeacherResponse:
        """Call the backend, retrying transient failures.

        Args:
            prompt: The rendered prompt.
            case_id: The case.
            kind: The call kind.
            attempt: Repair attempt index.

        Returns:
            The completion.

        Raises:
            TransientAPIError: When the retry policy is exhausted.
            PermanentAPIError: Immediately, on a non-retryable failure.
        """
        last: Exception | None = None
        for n in range(1, self.retry.max_attempts + 1):
            started = time.monotonic()
            try:
                if self.semaphore is not None:
                    with self.semaphore:
                        response = self._backend(self.spec, prompt)
                else:
                    response = self._backend(self.spec, prompt)
            except PermanentAPIError as exc:
                self.errors.add(
                    ErrorRecord(
                        case_id=case_id,
                        teacher=self.spec.key,
                        kind=kind,
                        attempt=attempt,
                        classification="permanent",
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
                raise
            except TransientAPIError as exc:
                last = exc
                self.errors.add(
                    ErrorRecord(
                        case_id=case_id,
                        teacher=self.spec.key,
                        kind=kind,
                        attempt=attempt,
                        classification="transient",
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
                if n == self.retry.max_attempts:
                    break
                delay = self.retry.delay_for(n, getattr(exc, "retry_after", None))
                log.warning(
                    "%s attempt %d/%d for %s failed (%s); retrying in %.1fs",
                    self.spec.key,
                    n,
                    self.retry.max_attempts,
                    case_id,
                    exc,
                    delay,
                )
                self._sleep(delay)
                continue
            return TeacherResponse(
                text=response.text,
                model_served=response.model_served,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cache_read_tokens=response.cache_read_tokens,
                cache_write_tokens=response.cache_write_tokens,
                cost_usd=_price(
                    self.spec,
                    response.input_tokens,
                    response.output_tokens,
                    response.cache_read_tokens,
                ),
                latency_s=time.monotonic() - started,
                stop_reason=response.stop_reason,
                request_id=response.request_id,
                attempts=n,
            )
        raise TransientAPIError(
            f"{self.spec.key} failed {self.retry.max_attempts} times for {case_id}: {last}"
        )


def _backend_for(spec: TeacherSpec) -> Callable[[TeacherSpec, RenderedPrompt], TeacherResponse]:
    """Select the transport for a provider.

    Args:
        spec: The teacher.

    Returns:
        A callable performing one request.

    Raises:
        ValueError: If the provider is unknown. Never a silent default: a spec that names
            a provider this code does not implement must fail before the run starts, not
            produce records from whichever backend happened to be first.
    """
    if spec.provider == "anthropic":
        return _anthropic_call
    if spec.provider == "openai_compatible":
        return _openai_compatible_call
    raise ValueError(
        f"teacher {spec.key!r} names unknown provider {spec.provider!r}; expected "
        "'anthropic' or 'openai_compatible'"
    )


def _anthropic_call(spec: TeacherSpec, prompt: RenderedPrompt) -> TeacherResponse:
    """Perform one Anthropic Messages request through the official SDK.

    Three things here are model-behaviour decisions rather than plumbing:

    - ``temperature`` and ``top_p`` are sent **only** when the spec says the model accepts
      them. The current frontier models reject them with a 400.
    - Thinking is left on (adaptive) at a low effort rather than disabled. Disabling it on
      the current Opus family can leak internal reasoning tags into the visible response,
      and a leaked tag would land verbatim in a corpus narrative.
    - ``stop_reason == "refusal"`` is checked **before** the content is read, because a
      refused response carries no content block to index into. A refusal is permanent and
      is recorded rather than retried: an AML narrative task sits close enough to the
      safety classifiers that the refusal rate is itself worth reporting.

    Args:
        spec: The teacher.
        prompt: The rendered prompt.

    Returns:
        The completion.

    Raises:
        TransientAPIError: On a rate limit, overload, timeout or 5xx.
        PermanentAPIError: On a bad request, an auth failure or a content refusal.
        RuntimeError: If the ``api`` extra is not installed.
    """
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover -- exercised only without the extra
        raise RuntimeError(
            "the anthropic SDK is required for an 'anthropic' teacher; install the api "
            "extra with `uv sync --extra api`"
        ) from exc

    client = anthropic.Anthropic(
        api_key=os.environ.get(spec.api_key_env),
        base_url=spec.base_url,
        timeout=spec.request_timeout_s,
        max_retries=0,  # retries are this module's job, and are logged and budgeted
    )
    kwargs: dict[str, Any] = {
        "model": spec.model,
        "max_tokens": spec.max_output_tokens,
        "system": [
            {
                "type": "text",
                "text": prompt.system,
                # The system message is byte-identical for every case in the run, so this
                # turns ~900 tokens of instruction into a cache read on all but the first
                # call. prompts.assert_system_is_case_invariant enforces the precondition.
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": prompt.user}],
    }
    if spec.supports_sampling:
        if spec.temperature is not None:
            kwargs["temperature"] = spec.temperature
        if spec.top_p is not None:
            kwargs["top_p"] = spec.top_p
    if spec.thinking:
        kwargs["thinking"] = {"type": spec.thinking}
    if spec.effort:
        kwargs["output_config"] = {"effort": spec.effort}

    try:
        message = client.messages.create(**kwargs)
    except anthropic.APIStatusError as exc:
        retry_after = _retry_after(getattr(exc, "response", None))
        if exc.status_code in TRANSIENT_STATUSES:
            error = TransientAPIError(f"{exc.status_code} from {spec.model}: {exc}")
            error.retry_after = retry_after  # type: ignore[attr-defined]
            raise error from exc
        raise PermanentAPIError(f"{exc.status_code} from {spec.model}: {exc}") from exc
    except anthropic.APIConnectionError as exc:
        raise TransientAPIError(f"connection error to {spec.model}: {exc}") from exc

    if message.stop_reason == "refusal":
        raise PermanentAPIError(
            f"{spec.model} declined this case on content grounds "
            f"(stop_details={getattr(message, 'stop_details', None)})"
        )

    text = "".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    ).strip()
    if not text:
        raise PermanentAPIError(
            f"{spec.model} returned no text (stop_reason={message.stop_reason!r})"
        )

    usage = message.usage
    return TeacherResponse(
        text=text,
        model_served=str(message.model),
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        cache_read_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        cache_write_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        stop_reason=str(message.stop_reason or ""),
        request_id=str(getattr(message, "_request_id", "") or ""),
    )


def _openai_compatible_call(spec: TeacherSpec, prompt: RenderedPrompt) -> TeacherResponse:
    """Perform one chat-completions request against an OpenAI-compatible server.

    The open-weights teacher is served by vLLM, TGI or a hosted equivalent, all of which
    speak this one endpoint. Written against :mod:`urllib` rather than a client library
    because the request is a single JSON POST and the alternative is another dependency in
    an environment CLAUDE.md keeps deliberately thin.

    Args:
        spec: The teacher.
        prompt: The rendered prompt.

    Returns:
        The completion.

    Raises:
        TransientAPIError: On a rate limit, a 5xx or a network failure.
        PermanentAPIError: On a 4xx or a malformed response.
        ValueError: If the spec names no base URL.
    """
    import urllib.error
    import urllib.request

    if not spec.base_url:
        raise ValueError(f"teacher {spec.key!r} is openai_compatible but names no base_url")

    body: dict[str, Any] = {
        "model": spec.model,
        "max_tokens": spec.max_output_tokens,
        "messages": [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ],
    }
    if spec.supports_sampling:
        if spec.temperature is not None:
            body["temperature"] = spec.temperature
        if spec.top_p is not None:
            body["top_p"] = spec.top_p
    if spec.seed is not None:
        body["seed"] = spec.seed

    request = urllib.request.Request(
        url=spec.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get(spec.api_key_env, '')}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=spec.request_timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        if exc.code in TRANSIENT_STATUSES:
            error = TransientAPIError(f"{exc.code} from {spec.model}: {detail}")
            error.retry_after = _retry_after(exc)  # type: ignore[attr-defined]
            raise error from exc
        raise PermanentAPIError(f"{exc.code} from {spec.model}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise TransientAPIError(f"connection error to {spec.model}: {exc}") from exc

    try:
        choice = payload["choices"][0]
        text = str(choice["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise PermanentAPIError(f"malformed response from {spec.model}: {payload}") from exc
    if not text:
        raise PermanentAPIError(f"{spec.model} returned empty content")

    usage = payload.get("usage") or {}
    return TeacherResponse(
        text=text,
        model_served=str(payload.get("model", spec.model)),
        input_tokens=int(usage.get("prompt_tokens", 0) or 0),
        output_tokens=int(usage.get("completion_tokens", 0) or 0),
        stop_reason=str(choice.get("finish_reason", "") or ""),
        request_id=str(payload.get("id", "") or ""),
    )


def _retry_after(response: Any) -> float | None:
    """Read a ``retry-after`` header, when there is one.

    Args:
        response: An object exposing ``headers``, or None.

    Returns:
        The delay in seconds, or None when the header is absent or unparseable.
    """
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


def build_teacher(
    spec: TeacherSpec,
    *,
    cache: ResponseCache,
    tracker: CostTracker,
    budget: BudgetGuard,
    errors: ErrorLog,
    retry: RetryPolicy | None = None,
    semaphore: threading.Semaphore | None = None,
) -> Teacher:
    """Construct the client for a teacher spec.

    Args:
        spec: The teacher.
        cache: Response cache.
        tracker: Cost tracker.
        budget: Budget guard.
        errors: Error log.
        retry: Backoff policy.
        semaphore: Shared concurrency limiter.

    Returns:
        A client satisfying :class:`Teacher`.

    Raises:
        ValueError: If the spec names an unknown provider.
    """
    _backend_for(spec)  # fail now, not on the first call of a multi-hour run
    return APITeacher(
        spec,
        cache=cache,
        tracker=tracker,
        budget=budget,
        errors=errors,
        retry=retry,
        semaphore=semaphore,
    )


def specs_from_config(entries: Iterable[dict[str, Any]]) -> tuple[TeacherSpec, ...]:
    """Build teacher specs from configuration.

    Args:
        entries: One mapping per teacher, keyed as :class:`TeacherSpec` fields.

    Returns:
        The specs, in configuration order.

    Raises:
        ValueError: If fewer than two teachers are configured, if two share a key, or if
            every teacher is from the same family. **Single-teacher Silver is refused
            here rather than warned about**: it is the objection the whole tier exists to
            answer, and a run that produced it would have to be thrown away.
    """
    specs = tuple(TeacherSpec(**dict(entry)) for entry in entries)
    if len(specs) < MIN_TEACHERS:
        raise ValueError(
            f"Silver requires at least {MIN_TEACHERS} teachers, got {len(specs)}. Single-teacher "
            "Silver is distillation from one model and invites exactly the objection "
            "the tier exists to answer (DECISIONS.md D-044)."
        )
    if len({s.key for s in specs}) != len(specs):
        raise ValueError("teacher keys must be distinct; they key the balance report")
    if len({s.family for s in specs}) < MIN_TEACHERS:
        raise ValueError(
            "teachers must span at least two families (frontier and open_weights); "
            f"got {sorted({s.family for s in specs})}"
        )
    for spec in specs:
        _backend_for(spec)  # unknown provider fails here, not on the first call
    return specs


def preflight(specs: Iterable[TeacherSpec]) -> list[str]:
    """Report everything that would stop a run, before any work is done.

    Credentials are checked here rather than discovered on the first call because the
    first call happens inside a worker pool, several minutes of loading and rendering into
    a job — and on a long run, the second provider's missing key would not surface until
    the first case routed to it. Both problems, reported together, at second zero.

    Args:
        specs: The configured teachers.

    Returns:
        One human-readable problem per line, empty when the run can proceed. Returned
        rather than raised so the caller can print all of them at once: fixing one missing
        credential, rerunning, and discovering the next is the slow way to start a job.
    """
    problems: list[str] = []
    for spec in specs:
        if not os.environ.get(spec.api_key_env):
            problems.append(
                f"teacher {spec.key!r} ({spec.model}) needs ${spec.api_key_env}, which is "
                "unset in this environment"
            )
        if spec.provider == "openai_compatible" and not spec.base_url:
            problems.append(
                f"teacher {spec.key!r} is openai_compatible but names no base_url; set "
                "corpus.teachers[].base_url to the served endpoint"
            )
    return problems
