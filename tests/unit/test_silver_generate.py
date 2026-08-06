"""The loop: clean pass, repair-then-pass, repair-twice-then-discard, and the balance.

The three loop paths the phase brief names are tested exactly, because a discard rate is
only a finding if the machinery that produces it is known to work in all three directions.
A verifier that never accepts is as useless as one that never rejects, and both look like
a plausible number in a report.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pytest
from tests import factories

from g2t_aml.corpus.bronze.renderer import render_bronze
from g2t_aml.corpus.silver.api_client import (
    BudgetExceeded,
    ScriptedTeacher,
    TeacherSpec,
    TransientAPIError,
)
from g2t_aml.corpus.silver.generate import (
    MAX_UNVERIFIABLE_RATE,
    CaseInput,
    SilverConfig,
    assign_teachers,
    discard_report,
    generate_one,
    teacher_balance_report,
    verify_rewrite,
)
from g2t_aml.corpus.validate import MAX_UNVERIFIABLE_RATE as HARNESS_RATE
from g2t_aml.facts.extractor import extract_facts
from g2t_aml.facts.vocab import load_vocabulary

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
)

#: A sentence that adds a fabricated account, a fabricated amount and a fabricated share.
POISON = (
    " A further 88,412.00 US Dollar moved through 999999|FAKE0001, "
    "some 41% of the inflow observed."
)


@pytest.fixture(scope="module")
def vocab():
    return load_vocabulary()


@pytest.fixture
def case(vocab):
    facts = extract_facts(
        factories.as_laundering_stream(factories.fan_out_case(width=6), "fan_out")
    )
    return CaseInput(
        case_id=facts.case_id,
        split="train",
        facts=facts,
        bronze=render_bronze(facts, vocabulary=vocab),
    )


class TestThresholds:
    def test_the_budget_agrees_with_the_harness(self):
        """Two thresholds that can drift are one bug away from a corpus that passes
        generation and then fails its own gate."""
        assert MAX_UNVERIFIABLE_RATE == HARNESS_RATE


class TestVerification:
    def test_bronze_verifies_against_itself(self, case, vocab):
        verdict, report = verify_rewrite(case.bronze.text, case, vocabulary=vocab)
        assert verdict.accepted
        assert verdict.contradicted == 0
        assert verdict.unverifiable == 0
        assert report.n_added == 0
        assert verdict.salience_coverage == pytest.approx(1.0)

    def test_an_invented_quantity_is_caught(self, case, vocab):
        verdict, report = verify_rewrite(case.bronze.text + POISON, case, vocabulary=vocab)
        assert not verdict.accepted
        assert report.n_added >= 2
        assert verdict.unverifiable > 0

    def test_a_fabricated_account_is_h1(self, case, vocab):
        verdict, _ = verify_rewrite(case.bronze.text + POISON, case, vocabulary=vocab)
        assert "H1" in verdict.by_hallucination_class

    def test_a_guilt_assertion_is_caught(self, case, vocab):
        text = case.bronze.text + " The account holder is guilty of money laundering."
        verdict, _ = verify_rewrite(text, case, vocabulary=vocab)
        assert verdict.contradicted > 0
        assert "H7" in verdict.by_hallucination_class
        assert verdict.critical_error_rate > 0

    def test_dropping_the_findings_is_caught_by_salience_not_by_the_checker(self, case, vocab):
        """The failure neither the contradiction count nor the unverifiable rate can see:
        a fluent, faithful narrative that says almost nothing."""
        verdict, _ = verify_rewrite(
            "The subject account was reviewed. The activity warrants further review.",
            case,
            vocabulary=vocab,
        )
        assert verdict.contradicted == 0
        assert verdict.salience_coverage < 1.0
        assert "salient_facts_dropped" in verdict.failures()
        assert not verdict.accepted

    def test_the_verdict_carries_the_thresholds_it_was_judged_under(self, case, vocab):
        config = SilverConfig(max_unverifiable_rate=0.5, min_salience_coverage=0.1)
        verdict, _ = verify_rewrite(
            case.bronze.text + POISON, case, vocabulary=vocab, config=config
        )
        assert verdict.max_unverifiable_rate == 0.5
        assert "unverifiable_rate_exceeded" not in verdict.failures()


class TestLoop:
    def test_clean_pass_writes_a_record_with_no_repairs(self, case, vocab):
        teacher = ScriptedTeacher(FRONTIER, lambda p, c, k, a: case.bronze.text)
        outcome = generate_one(case, teacher, vocabulary=vocab, graph_ref="store#x")
        assert outcome.accepted
        assert outcome.attempts == 0
        assert outcome.discard is None
        assert len(teacher.calls) == 1

    def test_repair_then_pass(self, case, vocab):
        def responder(prompt, case_id, kind, attempt):
            return case.bronze.text if kind == "repair" else case.bronze.text + POISON

        teacher = ScriptedTeacher(FRONTIER, responder)
        outcome = generate_one(case, teacher, vocabulary=vocab, graph_ref="store#x")
        assert outcome.accepted
        assert outcome.attempts == 1
        assert [k for _, k, _ in teacher.calls] == ["rewrite", "repair"]
        assert outcome.record.generator["repair_attempts"] == 1

    def test_repair_twice_then_discard(self, case, vocab):
        teacher = ScriptedTeacher(FRONTIER, lambda p, c, k, a: case.bronze.text + POISON)
        outcome = generate_one(case, teacher, vocabulary=vocab, graph_ref="store#x")
        assert not outcome.accepted
        assert outcome.attempts == 2
        assert len(teacher.calls) == 3  # one rewrite, two repairs, then stop
        assert outcome.discard.stage == "verification"
        assert outcome.discard.by_hallucination_class

    def test_the_repair_limit_is_not_exceeded(self, case, vocab):
        teacher = ScriptedTeacher(FRONTIER, lambda p, c, k, a: case.bronze.text + POISON)
        outcome = generate_one(
            case, teacher, vocabulary=vocab, config=SilverConfig(max_repair_attempts=1)
        )
        assert outcome.attempts == 1
        assert len(teacher.calls) == 2

    def test_a_discard_records_what_the_model_invented(self, case, vocab):
        teacher = ScriptedTeacher(FRONTIER, lambda p, c, k, a: case.bronze.text + POISON)
        outcome = generate_one(case, teacher, vocabulary=vocab)
        assert any("88,412.00" in q for q in outcome.discard.quoted_additions)

    def test_an_api_failure_is_a_separate_discard_stage(self, case, vocab):
        def responder(prompt, case_id, kind, attempt):
            raise TransientAPIError("503 after every retry")

        outcome = generate_one(case, ScriptedTeacher(FRONTIER, responder), vocabulary=vocab)
        assert not outcome.accepted
        assert outcome.discard.stage == "api"
        assert outcome.discard.reason == ("api_error",)

    def test_a_budget_stop_propagates_and_is_never_a_discard(self, case, vocab):
        """A halted run must not fill the discard log with cases nobody looked at -- they
        would be counted as model failures in the paper's table."""

        def responder(prompt, case_id, kind, attempt):
            raise BudgetExceeded("cap reached")

        with pytest.raises(BudgetExceeded):
            generate_one(case, ScriptedTeacher(FRONTIER, responder), vocabulary=vocab)


class TestRecordProvenance:
    def test_every_record_carries_full_provenance(self, case, vocab):
        teacher = ScriptedTeacher(FRONTIER, lambda p, c, k, a: case.bronze.text)
        record = generate_one(case, teacher, vocabulary=vocab, graph_ref="store#x").record
        generator = record.generator
        for field in (
            "teacher",
            "family",
            "provider",
            "model",
            "prompt_hash",
            "rendered_prompt_hash",
            "temperature",
            "top_p",
            "seed",
            "repair_attempts",
            "generated_at",
        ):
            assert field in generator, field
        assert generator["model"] == "claude-opus-5"
        assert record.tier == "silver"

    def test_a_model_that_rejects_sampling_records_null_and_the_reason(self, case, vocab):
        teacher = ScriptedTeacher(FRONTIER, lambda p, c, k, a: case.bronze.text)
        generator = generate_one(case, teacher, vocabulary=vocab).record.generator
        assert generator["temperature"] is None
        assert generator["sampling_parameters_supported"] is False
        assert generator["sampling_omitted_reason"]

    def test_an_open_weights_record_carries_its_sampling_settings(self, case, vocab):
        teacher = ScriptedTeacher(OPEN, lambda p, c, k, a: case.bronze.text)
        generator = generate_one(case, teacher, vocabulary=vocab).record.generator
        assert generator["temperature"] == 0.7
        assert generator["top_p"] == 0.95

    def test_surviving_slots_index_the_rewrite_not_the_bronze(self, case, vocab):
        """The harness asserts narrative[span] == rendered_value per slot. A carried-over
        Bronze span would put a lie in the record and mis-align Phase 10's evaluation."""
        preamble = "Preliminary review note. "

        teacher = ScriptedTeacher(FRONTIER, lambda p, c, k, a: preamble + case.bronze.text)
        record = generate_one(case, teacher, vocabulary=vocab).record
        assert record.target_slots
        for slot in record.target_slots:
            start, end = slot.span
            assert record.target_narrative[start:end] == slot.rendered_value

    def test_a_dropped_slot_is_not_carried_over(self, case, vocab):
        teacher = ScriptedTeacher(
            FRONTIER,
            lambda p, c, k, a: case.bronze.text.replace(str(case.facts.structure.n_edges), "", 1),
        )
        outcome = generate_one(case, teacher, vocabulary=vocab)
        record = outcome.record if outcome.accepted else None
        if record is not None:
            for slot in record.target_slots:
                start, end = slot.span
                assert record.target_narrative[start:end] == slot.rendered_value


class TestTeacherAssignment:
    def test_assignment_is_deterministic(self, case):
        cases = [
            CaseInput(case_id=f"c{i}", split="train", facts=case.facts, bronze=case.bronze)
            for i in range(200)
        ]
        first = assign_teachers(cases, [FRONTIER, OPEN])
        second = assign_teachers(list(reversed(cases)), [FRONTIER, OPEN])
        assert first == second

    def test_assignment_is_balanced_within_every_stratum(self, case, vocab):
        """A bare hash mod two balances over the corpus and not within a stratum, and this
        corpus has strata of 60 cases where a coin flip lands 40/20 often enough to
        matter."""
        cases = []
        for split in ("train", "val", "test"):
            for i in range(61):  # deliberately odd, so exact halves are impossible
                cases.append(
                    CaseInput(
                        case_id=f"{split}-{i:04d}",
                        split=split,
                        facts=case.facts,
                        bronze=case.bronze,
                    )
                )
        assignment = assign_teachers(cases, [FRONTIER, OPEN])
        report = teacher_balance_report(assignment, cases)
        for split, counts in report["assigned"]["by_split"].items():
            assert abs(counts["frontier"] - counts["open_weights"]) <= 1, split
        assert report["assigned"]["max_deviation_from_even"] <= 1.5

    def test_singleton_strata_do_not_all_go_to_one_teacher(self, case):
        """Round-robin from position zero in every stratum balances the large strata and
        hands EVERY one-case stratum to teacher zero -- reproducing, on a corpus of rare
        typology/split combinations, the exact skew stratification was added to remove.
        Caught by the integration fixture, pinned here.
        """

        # assign_teachers reads only case_id, split and typology, so a stub exercises the
        # stratification directly and lets every stratum hold exactly one case.
        @dataclass(frozen=True)
        class Stub:
            case_id: str
            split: str
            typology: str

        cases = [
            Stub(case_id=f"t{typology}-{split}", split=split, typology=f"typology-{typology}")
            for typology in range(40)
            for split in ("train", "val", "test")
        ]
        assignment = assign_teachers(cases, [FRONTIER, OPEN])  # type: ignore[arg-type]
        counts = Counter(assignment.values())
        assert set(counts) == {"frontier", "open_weights"}
        # Hash-spread rather than exactly even, since each stratum contributes one case.
        assert min(counts.values()) / sum(counts.values()) > 0.3

    def test_balance_report_covers_the_surviving_corpus(self, case):
        cases = [
            CaseInput(case_id=f"c{i:03d}", split="train", facts=case.facts, bronze=case.bronze)
            for i in range(100)
        ]
        assignment = assign_teachers(cases, [FRONTIER, OPEN])
        # Drop every frontier record, the asymmetry the report has to surface.
        kept = {c for c, t in assignment.items() if t != "frontier"}
        report = teacher_balance_report(assignment, cases, kept=kept)
        rates = report["retention_by_teacher"]["retention_rate"]
        assert rates["frontier"] == 0.0
        assert rates["open_weights"] == 1.0
        assert report["retention_by_teacher"]["retention_spread"] == 1.0

    def test_no_teachers_is_refused(self, case):
        with pytest.raises(ValueError, match="no teachers"):
            assign_teachers([case], [])


class TestDiscardReport:
    def test_api_and_verification_discards_are_reported_separately(self, case, vocab):
        """One is a result about models, the other an operational fact about a run.
        Adding them together inflates the number the paper reports."""
        verification = ScriptedTeacher(FRONTIER, lambda p, c, k, a: case.bronze.text + POISON)
        api = ScriptedTeacher(
            FRONTIER, lambda p, c, k, a: (_ for _ in ()).throw(TransientAPIError("503"))
        )
        discards = [
            generate_one(case, verification, vocabulary=vocab).discard,
            generate_one(case, api, vocabulary=vocab).discard,
        ]
        report = discard_report(discards, n_attempted=10)
        assert report["n_discarded"] == 2
        assert report["discard_rate"] == 0.2
        assert report["verification_discard_rate"] == 0.1
        assert report["api_discard_rate"] == 0.1
        assert report["by_hallucination_class"]
