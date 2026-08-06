"""The Silver pipeline end to end: generate, resume, halt, filter, and gate.

Everything here runs against :class:`ScriptedTeacher`, so the loop under test is exactly
the loop a real run executes and no network is involved. The two properties that only
show up at this level are **resume correctness** — kill a run, restart it, and get one
corpus with no gaps and no duplicates — and the fact that Silver records pass the same
ten-point harness that gates Bronze.
"""

from __future__ import annotations

import json

import pytest
from tests import factories

from g2t_aml.corpus.bronze.renderer import render_bronze
from g2t_aml.corpus.record import validate_training_record
from g2t_aml.corpus.silver.api_client import (
    BudgetGuard,
    CheckpointStore,
    CostTracker,
    ErrorLog,
    ResponseCache,
    ScriptedTeacher,
    TeacherSpec,
)
from g2t_aml.corpus.silver.generate import (
    CaseInput,
    SilverConfig,
    assign_teachers,
    discard_report,
    teacher_balance_report,
)
from g2t_aml.corpus.silver.quality import filter_records
from g2t_aml.corpus.silver.run import JSONLAppender, run_generation
from g2t_aml.corpus.tokenization import get_token_counter
from g2t_aml.facts.extractor import extract_facts
from g2t_aml.facts.vocab import load_vocabulary

pytestmark = pytest.mark.integration

FRONTIER = TeacherSpec(
    key="frontier",
    family="frontier",
    provider="anthropic",
    model="claude-opus-5",
    price_in_per_mtok=5.0,
    price_out_per_mtok=25.0,
)
OPEN = TeacherSpec(
    key="open_weights",
    family="open_weights",
    provider="openai_compatible",
    model="llama-3.3-70b",
    supports_sampling=True,
    temperature=0.7,
    top_p=0.95,
    price_in_per_mtok=0.6,
    price_out_per_mtok=0.6,
)

POISON = " A further 88,412.00 US Dollar reached 999999|FAKE0001, some 41% of inflow."


@pytest.fixture(scope="module")
def vocab():
    return load_vocabulary()


@pytest.fixture(scope="module")
def cases(vocab):
    """Twelve synthetic cases over six strata of two. Synthetic ids only.

    Deliberately *not* twelve singleton strata: the assignment guarantee is balance
    within a ``(typology, split)`` stratum, so a fixture where every stratum holds one
    case tests hash luck rather than the property.
    """
    builders = [
        ("fan_out", lambda: factories.fan_out_case(width=6)),
        ("fan_in", lambda: factories.fan_in_case(width=6)),
    ]
    splits = ("train", "val", "test")
    made = []
    for index in range(12):
        typology, builder = builders[index % len(builders)]
        facts = extract_facts(factories.as_laundering_stream(builder(), typology))
        bronze = render_bronze(facts, vocabulary=vocab)
        made.append(
            CaseInput(
                case_id=f"case-{index:03d}",
                split=splits[(index // 2) % len(splits)],
                facts=facts,
                bronze=bronze,
            )
        )
    return made


def teachers_for(responder, tmp_path, *, cap=0.0, tracker=None):
    tracker = tracker if tracker is not None else CostTracker()
    return (
        {
            FRONTIER.key: ScriptedTeacher(FRONTIER, responder, tracker=tracker),
            OPEN.key: ScriptedTeacher(OPEN, responder, tracker=tracker),
        },
        tracker,
    )


def clean(cases):
    by_id = {c.case_id: c.bronze.text for c in cases}

    def responder(prompt, case_id, kind, attempt):
        return by_id[case_id]

    return responder


def run(cases, teachers, tracker, tmp_path, *, checkpoint=None, records=None, discards=None):
    return run_generation(
        cases,
        assign_teachers(cases, [FRONTIER, OPEN]),
        teachers,
        vocabulary=load_vocabulary(),
        config=SilverConfig(),
        token_counter=get_token_counter(),
        graph_ref_for=lambda case_id: f"data/store#{case_id}",
        tracker=tracker,
        checkpoint=checkpoint,
        records_path=records,
        discards_path=discards,
        concurrency=4,
    )


class TestEndToEnd:
    def test_a_clean_run_writes_every_case(self, cases, tmp_path):
        teachers, tracker = teachers_for(clean(cases), tmp_path)
        result = run(cases, teachers, tracker, tmp_path, records=tmp_path / "silver.jsonl")
        assert len(result.records) == len(cases)
        assert result.discards == []
        assert {r["case_id"] for r in result.records} == {c.case_id for c in cases}

    def test_records_validate_against_the_frozen_training_record_schema(self, cases, tmp_path):
        teachers, tracker = teachers_for(clean(cases), tmp_path)
        result = run(cases, teachers, tracker, tmp_path, records=tmp_path / "silver.jsonl")
        for payload in result.records:
            validate_training_record(payload)
            assert payload["tier"] == "silver"

    def test_every_written_record_carries_its_provenance(self, cases, tmp_path):
        teachers, tracker = teachers_for(clean(cases), tmp_path)
        result = run(cases, teachers, tracker, tmp_path, records=tmp_path / "silver.jsonl")
        for payload in result.records:
            generator = payload["generator"]
            assert generator["teacher"] in {"frontier", "open_weights"}
            assert generator["model"]
            assert len(generator["prompt_hash"]) == 64
            assert "temperature" in generator
            assert "generated_at" in generator

    def test_both_teachers_are_used_and_balanced(self, cases, tmp_path):
        teachers, tracker = teachers_for(clean(cases), tmp_path)
        result = run(cases, teachers, tracker, tmp_path, records=tmp_path / "silver.jsonl")
        assignment = assign_teachers(cases, [FRONTIER, OPEN])
        balance = teacher_balance_report(
            assignment, cases, kept={r["case_id"] for r in result.records}
        )
        counts = balance["assigned"]["by_teacher"]
        assert set(counts) == {"frontier", "open_weights"}
        assert abs(counts["frontier"] - counts["open_weights"]) <= 1
        for split_counts in balance["assigned"]["by_split"].values():
            assert abs(split_counts.get("frontier", 0) - split_counts.get("open_weights", 0)) <= 1

    def test_a_corpus_of_unrepairable_rewrites_is_entirely_discarded(self, cases, tmp_path):
        by_id = {c.case_id: c.bronze.text for c in cases}
        teachers, tracker = teachers_for(lambda p, c, k, a: by_id[c] + POISON, tmp_path)
        result = run(
            cases,
            teachers,
            tracker,
            tmp_path,
            records=tmp_path / "silver.jsonl",
            discards=tmp_path / "discards.jsonl",
        )
        assert result.records == []
        assert len(result.discards) == len(cases)
        report = discard_report(result.discards, result.n_attempted)
        assert report["discard_rate"] == 1.0
        assert report["by_hallucination_class"]
        assert all(d["attempts"] == 2 for d in JSONLAppender.read(tmp_path / "discards.jsonl"))

    def test_the_discard_log_is_written_as_it_goes(self, cases, tmp_path):
        by_id = {c.case_id: c.bronze.text for c in cases}
        teachers, tracker = teachers_for(lambda p, c, k, a: by_id[c] + POISON, tmp_path)
        path = tmp_path / "discards.jsonl"
        run(cases, teachers, tracker, tmp_path, discards=path)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == len(cases)
        for row in rows:
            assert row["reason"]
            assert row["teacher"] in {"frontier", "open_weights"}
            assert row["typology"]
            assert row["split"] in {"train", "val", "test"}


class TestResume:
    def test_resume_produces_no_gaps_and_no_duplicates(self, cases, tmp_path):
        """Kill a run halfway, restart it, and the corpus must be the corpus an
        uninterrupted run would have produced -- every case once, none missing."""
        records = tmp_path / "silver.jsonl"
        checkpoint = CheckpointStore(tmp_path / "checkpoint.txt")
        half = cases[:5]
        rest = cases[5:]

        teachers, tracker = teachers_for(clean(cases), tmp_path)
        run(half, teachers, tracker, tmp_path, checkpoint=checkpoint, records=records)
        assert len(checkpoint) == 5

        # A fresh process: new checkpoint object reading the same file, new teachers.
        resumed_checkpoint = CheckpointStore(tmp_path / "checkpoint.txt")
        teachers2, tracker2 = teachers_for(clean(cases), tmp_path)
        result = run(
            cases, teachers2, tracker2, tmp_path, checkpoint=resumed_checkpoint, records=records
        )

        assert result.n_resumed == 5
        assert result.n_attempted == len(rest)
        written = [r["case_id"] for r in JSONLAppender.read(records)]
        assert len(written) == len(cases)
        assert len(set(written)) == len(cases)
        assert set(written) == {c.case_id for c in cases}

    def test_a_resumed_case_is_never_regenerated(self, cases, tmp_path):
        checkpoint = CheckpointStore(tmp_path / "checkpoint.txt")
        for case in cases:
            checkpoint.mark(case.case_id)
        calls: list[str] = []

        def responder(prompt, case_id, kind, attempt):
            calls.append(case_id)
            return "should never be called"

        teachers, tracker = teachers_for(responder, tmp_path)
        result = run(cases, teachers, tracker, tmp_path, checkpoint=checkpoint)
        assert calls == []
        assert result.n_attempted == 0
        assert result.n_resumed == len(cases)

    def test_discarded_cases_are_checkpointed_too(self, cases, tmp_path):
        """A case that was tried and failed is finished. Leaving it un-checkpointed would
        make every resume re-pay for the cases least likely to ever succeed."""
        by_id = {c.case_id: c.bronze.text for c in cases}
        checkpoint = CheckpointStore(tmp_path / "checkpoint.txt")
        teachers, tracker = teachers_for(lambda p, c, k, a: by_id[c] + POISON, tmp_path)
        run(cases, teachers, tracker, tmp_path, checkpoint=checkpoint)
        assert len(checkpoint) == len(cases)

    def test_a_truncated_final_line_is_dropped_not_fatal(self, tmp_path):
        path = tmp_path / "silver.jsonl"
        path.write_text('{"case_id": "a"}\n{"case_id": "b"}\n{"case_id": "trunc', encoding="utf-8")
        assert [r["case_id"] for r in JSONLAppender.read(path)] == ["a", "b"]

    def test_a_malformed_middle_line_is_fatal(self, tmp_path):
        path = tmp_path / "silver.jsonl"
        path.write_text('{"case_id": "a"}\nnot json\n{"case_id": "b"}\n', encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            JSONLAppender.read(path)


class TestBudgetHalt:
    def test_the_cap_halts_the_run_without_writing_discards(self, cases, tmp_path):
        """A budget stop is not a data finding. Every case left un-run must stay out of
        the discard log, where it would be counted as a model failure in the paper."""
        tracker = CostTracker()
        cache = ResponseCache(tmp_path / "cache", enabled=False)
        errors = ErrorLog()
        budget = BudgetGuard(0.000_01, tracker)  # trips almost immediately
        by_id = {c.case_id: c.bronze.text for c in cases}

        from g2t_aml.corpus.silver.api_client import APITeacher, TeacherResponse

        def backend(spec, prompt):
            return TeacherResponse(
                text=by_id[next(iter(by_id))],
                model_served=spec.model,
                input_tokens=100_000,
                output_tokens=100_000,
            )

        teachers = {
            spec.key: APITeacher(
                spec,
                cache=cache,
                tracker=tracker,
                budget=budget,
                errors=errors,
                sleep=lambda _: None,
                backend=backend,
            )
            for spec in (FRONTIER, OPEN)
        }
        result = run(
            cases,
            teachers,
            tracker,
            tmp_path,
            records=tmp_path / "silver.jsonl",
            discards=tmp_path / "discards.jsonl",
        )
        assert result.halted
        assert result.halt_reason
        assert not any(d.stage == "verification" for d in result.discards)
        assert result.n_attempted < len(cases)

    def test_a_halted_run_resumes_from_where_it_stopped(self, cases, tmp_path):
        checkpoint = CheckpointStore(tmp_path / "checkpoint.txt")
        checkpoint.mark(cases[0].case_id)
        teachers, tracker = teachers_for(clean(cases), tmp_path)
        result = run(cases, teachers, tracker, tmp_path, checkpoint=checkpoint)
        assert result.n_resumed == 1
        assert result.n_attempted == len(cases) - 1


class TestFilteringAndGate:
    def test_a_verbatim_bronze_rewrite_is_verified_but_filtered_out(self, cases, tmp_path):
        """The pipeline's sharpest self-check: copying Bronze passes verification
        perfectly and still must not reach the corpus."""
        teachers, tracker = teachers_for(clean(cases), tmp_path)
        result = run(cases, teachers, tracker, tmp_path, records=tmp_path / "silver.jsonl")
        assert len(result.records) == len(cases)  # all verified

        kept, report = filter_records(result.records, {c.case_id: c.bronze.text for c in cases})
        assert kept == []
        assert report.by_reason["bronze_verbatim"] == len(cases)

    def test_the_cost_report_covers_both_teachers(self, cases, tmp_path):
        teachers, tracker = teachers_for(clean(cases), tmp_path)
        run(cases, teachers, tracker, tmp_path)
        report = tracker.to_dict()
        assert set(report["by_teacher_usd"]) == {"frontier", "open_weights"}
        assert report["total_usd"] > 0
        assert sum(report["calls"].values()) == len(cases)
