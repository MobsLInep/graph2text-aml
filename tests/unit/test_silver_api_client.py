"""Cache correctness, retry classification, the budget stop, and resume.

The plumbing that makes a multi-hour two-provider job survivable. Every test here drives
an injected backend rather than a network, so the failure sequences are exact.
"""

from __future__ import annotations

import threading

import pytest

from g2t_aml.corpus.silver.api_client import (
    APITeacher,
    BudgetExceeded,
    BudgetGuard,
    CheckpointStore,
    CostTracker,
    ErrorLog,
    PermanentAPIError,
    ResponseCache,
    RetryPolicy,
    TeacherResponse,
    TeacherSpec,
    TransientAPIError,
    cache_key_components,
    specs_from_config,
)
from g2t_aml.corpus.silver.prompts import RenderedPrompt

PROMPT = RenderedPrompt(
    template_name="silver_rewrite_v1",
    prompt_hash="a" * 64,
    system="system text",
    user="user text",
    rendered_hash="b" * 64,
    system_hash="c" * 64,
)

SPEC = TeacherSpec(
    key="frontier",
    family="frontier",
    provider="anthropic",
    model="claude-opus-5",
    price_in_per_mtok=5.0,
    price_out_per_mtok=25.0,
)


def make_teacher(tmp_path, backend, *, cap=0.0, retry=None, spec=SPEC):
    tracker = CostTracker()
    cache = ResponseCache(tmp_path / "cache")
    teacher = APITeacher(
        spec,
        cache=cache,
        tracker=tracker,
        budget=BudgetGuard(cap, tracker),
        errors=ErrorLog(),
        retry=retry if retry is not None else RetryPolicy(max_attempts=3, base_delay_s=0.0),
        sleep=lambda _: None,
        backend=backend,
    )
    return teacher, tracker, cache


def ok(text="a narrative", n_in=1000, n_out=200):
    def backend(spec, prompt):
        return TeacherResponse(
            text=text, model_served=spec.model, input_tokens=n_in, output_tokens=n_out
        )

    return backend


class TestCache:
    def test_identical_request_is_served_from_cache(self, tmp_path):
        calls = []

        def backend(spec, prompt):
            calls.append(1)
            return TeacherResponse(text="narrative", model_served=spec.model, input_tokens=100)

        teacher, tracker, cache = make_teacher(tmp_path, backend)
        first = teacher.complete(PROMPT, case_id="c1", kind="rewrite")
        second = teacher.complete(PROMPT, case_id="c1", kind="rewrite")

        assert len(calls) == 1
        assert not first.from_cache and second.from_cache
        assert first.text == second.text
        assert cache.hits == 1

    def test_a_cached_call_costs_nothing(self, tmp_path):
        """Otherwise every re-run would re-report the original spend and the cost report
        would grow without any money being spent."""
        teacher, tracker, _ = make_teacher(tmp_path, ok())
        teacher.complete(PROMPT, case_id="c1", kind="rewrite")
        spent_after_first = tracker.total_usd
        teacher.complete(PROMPT, case_id="c1", kind="rewrite")
        assert spent_after_first > 0
        assert tracker.total_usd == pytest.approx(spent_after_first)

    @pytest.mark.parametrize(
        "mutation",
        [
            {"case_id": "other-case"},
            {"kind": "repair"},
            {"attempt": 1},
        ],
    )
    def test_cache_key_changes_with_the_request(self, mutation):
        base = cache_key_components(SPEC, PROMPT, "c1", "rewrite", 0)
        args = {"case_id": "c1", "kind": "rewrite", "attempt": 0, **mutation}
        other = cache_key_components(SPEC, PROMPT, args["case_id"], args["kind"], args["attempt"])
        assert ResponseCache.digest(base) != ResponseCache.digest(other)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("model", "claude-sonnet-5"),
            ("temperature", 0.9),
            ("top_p", 0.5),
            ("seed", 99),
            ("effort", "high"),
            ("max_output_tokens", 8192),
        ],
    )
    def test_cache_key_changes_with_the_teacher(self, field, value):
        import dataclasses

        base = cache_key_components(SPEC, PROMPT, "c1", "rewrite", 0)
        spec = dataclasses.replace(SPEC, supports_sampling=True, **{field: value})
        other = cache_key_components(spec, PROMPT, "c1", "rewrite", 0)
        assert ResponseCache.digest(base) != ResponseCache.digest(other)

    def test_cache_key_changes_when_the_prompt_changes(self):
        import dataclasses

        base = cache_key_components(SPEC, PROMPT, "c1", "rewrite", 0)
        edited = dataclasses.replace(PROMPT, prompt_hash="z" * 64, rendered_hash="y" * 64)
        assert ResponseCache.digest(base) != ResponseCache.digest(
            cache_key_components(SPEC, edited, "c1", "rewrite", 0)
        )

    def test_a_corrupt_entry_is_a_miss_not_a_crash(self, tmp_path):
        teacher, _, cache = make_teacher(tmp_path, ok())
        teacher.complete(PROMPT, case_id="c1", kind="rewrite")
        entry = next((tmp_path / "cache").rglob("*.json"))
        entry.write_text("{ truncated", encoding="utf-8")
        assert teacher.complete(PROMPT, case_id="c1", kind="rewrite").text == "a narrative"

    def test_a_disabled_cache_never_serves(self, tmp_path):
        calls = []

        def backend(spec, prompt):
            calls.append(1)
            return TeacherResponse(text="n", model_served=spec.model)

        tracker = CostTracker()
        teacher = APITeacher(
            SPEC,
            cache=ResponseCache(tmp_path / "c", enabled=False),
            tracker=tracker,
            budget=BudgetGuard(0.0, tracker),
            errors=ErrorLog(),
            sleep=lambda _: None,
            backend=backend,
        )
        teacher.complete(PROMPT, case_id="c1", kind="rewrite")
        teacher.complete(PROMPT, case_id="c1", kind="rewrite")
        assert len(calls) == 2


class TestRetry:
    def test_a_transient_failure_is_retried_and_then_succeeds(self, tmp_path):
        attempts = []

        def backend(spec, prompt):
            attempts.append(1)
            if len(attempts) < 3:
                raise TransientAPIError("429 rate limited")
            return TeacherResponse(text="narrative", model_served=spec.model)

        teacher, _, _ = make_teacher(tmp_path, backend)
        response = teacher.complete(PROMPT, case_id="c1", kind="rewrite")
        assert len(attempts) == 3
        assert response.attempts == 3

    def test_a_permanent_failure_is_not_retried(self, tmp_path):
        attempts = []

        def backend(spec, prompt):
            attempts.append(1)
            raise PermanentAPIError("400 bad request")

        teacher, _, _ = make_teacher(tmp_path, backend)
        with pytest.raises(PermanentAPIError):
            teacher.complete(PROMPT, case_id="c1", kind="rewrite")
        assert len(attempts) == 1

    def test_exhausted_retries_raise(self, tmp_path):
        def backend(spec, prompt):
            raise TransientAPIError("503")

        teacher, _, _ = make_teacher(tmp_path, backend)
        with pytest.raises(TransientAPIError):
            teacher.complete(PROMPT, case_id="c1", kind="rewrite")

    def test_failures_are_classified_in_the_error_log(self, tmp_path):
        errors = ErrorLog()
        tracker = CostTracker()
        teacher = APITeacher(
            SPEC,
            cache=ResponseCache(tmp_path / "c"),
            tracker=tracker,
            budget=BudgetGuard(0.0, tracker),
            errors=errors,
            retry=RetryPolicy(max_attempts=2, base_delay_s=0.0),
            sleep=lambda _: None,
            backend=lambda s, p: (_ for _ in ()).throw(TransientAPIError("429")),
        )
        with pytest.raises(TransientAPIError):
            teacher.complete(PROMPT, case_id="c1", kind="rewrite")
        assert errors.summary()["by_classification"] == {"transient": 2}

    def test_retry_after_wins_over_the_computed_backoff(self):
        policy = RetryPolicy(base_delay_s=1.0, max_delay_s=60.0)
        assert policy.delay_for(1, retry_after=30.0) == 30.0
        assert policy.delay_for(1, retry_after=999.0) == 60.0

    def test_jitter_keeps_the_delay_inside_the_envelope(self):
        policy = RetryPolicy(base_delay_s=2.0, max_delay_s=60.0)
        assert all(0.0 <= policy.delay_for(3) <= 8.0 for _ in range(50))


class TestBudget:
    def test_the_cap_halts_the_job(self, tmp_path):
        teacher, tracker, _ = make_teacher(tmp_path, ok(n_in=1_000_000, n_out=1_000_000), cap=1.0)
        teacher.complete(PROMPT, case_id="c1", kind="rewrite")
        assert tracker.total_usd >= 1.0
        with pytest.raises(BudgetExceeded):
            teacher.complete(PROMPT, case_id="c2", kind="rewrite")

    def test_the_cache_is_checked_before_the_budget(self, tmp_path):
        """A resumed run that will spend nothing must not be halted by a cap it already
        reached in the run before it."""
        teacher, tracker, _ = make_teacher(tmp_path, ok(n_in=1_000_000, n_out=1_000_000), cap=1.0)
        teacher.complete(PROMPT, case_id="c1", kind="rewrite")
        assert teacher.complete(PROMPT, case_id="c1", kind="rewrite").from_cache

    def test_no_cap_means_no_halt(self, tmp_path):
        teacher, _, _ = make_teacher(tmp_path, ok(n_in=10_000_000), cap=0.0)
        for i in range(3):
            teacher.complete(PROMPT, case_id=f"c{i}", kind="rewrite")


class TestCheckpoint:
    def test_marks_survive_reopening(self, tmp_path):
        path = tmp_path / "ckpt.txt"
        store = CheckpointStore(path)
        store.mark("case-a")
        store.mark("case-b")
        store.mark("case-a")  # idempotent
        assert len(CheckpointStore(path)) == 2
        assert "case-a" in CheckpointStore(path)

    def test_disabled_checkpoint_remembers_nothing(self):
        store = CheckpointStore(None)
        store.mark("case-a")
        assert "case-a" in store  # in-process only
        assert len(store) == 1


class TestSpecs:
    def test_sampling_parameters_are_omitted_when_the_model_rejects_them(self):
        provenance = SPEC.sampling_provenance()
        assert provenance["temperature"] is None
        assert provenance["sampling_parameters_supported"] is False
        assert provenance["sampling_omitted_reason"]

    def test_sampling_parameters_are_recorded_when_supported(self):
        spec = TeacherSpec(
            key="open",
            family="open_weights",
            provider="openai_compatible",
            model="llama",
            supports_sampling=True,
            temperature=0.7,
            top_p=0.95,
        )
        assert spec.sampling_provenance()["temperature"] == 0.7
        assert spec.sampling_provenance()["top_p"] == 0.95

    def test_a_single_teacher_run_is_refused(self):
        with pytest.raises(ValueError, match="at least 2 teachers"):
            specs_from_config(
                [{"key": "a", "family": "frontier", "provider": "anthropic", "model": "m"}]
            )

    def test_two_teachers_from_one_family_are_refused(self):
        with pytest.raises(ValueError, match="two families"):
            specs_from_config(
                [
                    {"key": "a", "family": "frontier", "provider": "anthropic", "model": "m1"},
                    {"key": "b", "family": "frontier", "provider": "anthropic", "model": "m2"},
                ]
            )

    def test_an_unknown_provider_fails_before_the_run_starts(self, tmp_path):
        from g2t_aml.corpus.silver.api_client import build_teacher

        tracker = CostTracker()
        with pytest.raises(ValueError, match="unknown provider"):
            build_teacher(
                TeacherSpec(key="x", family="frontier", provider="carrier_pigeon", model="m"),
                cache=ResponseCache(tmp_path),
                tracker=tracker,
                budget=BudgetGuard(0.0, tracker),
                errors=ErrorLog(),
            )


class TestConcurrency:
    def test_the_semaphore_bounds_in_flight_calls(self, tmp_path):
        limit = 3
        semaphore = threading.Semaphore(limit)
        in_flight = []
        peak = []
        lock = threading.Lock()

        def backend(spec, prompt):
            with lock:
                in_flight.append(1)
                peak.append(len(in_flight))
            import time

            time.sleep(0.01)
            with lock:
                in_flight.pop()
            return TeacherResponse(text="n", model_served=spec.model)

        tracker = CostTracker()
        teacher = APITeacher(
            SPEC,
            cache=ResponseCache(tmp_path / "c", enabled=False),
            tracker=tracker,
            budget=BudgetGuard(0.0, tracker),
            errors=ErrorLog(),
            semaphore=semaphore,
            sleep=lambda _: None,
            backend=backend,
        )
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=12) as pool:
            list(
                pool.map(
                    lambda i: teacher.complete(PROMPT, case_id=f"c{i}", kind="rewrite"), range(24)
                )
            )
        assert max(peak) <= limit
