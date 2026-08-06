"""Layer 2 faithfulness, against hand-computed counts.

Every expected value here is worked out from the claim list in the test, not read back
from the module. The metric definitions in the brief are arithmetic over four counts, so
the arithmetic is what gets pinned; the interesting tests are the ones about *which*
claims land in which count, because that is where a definition can be quietly wrong and
still produce plausible numbers.
"""

from __future__ import annotations

import pytest
from tests.factories import as_laundering_stream, fan_out_case

from g2t_aml.eval.claim_extraction.deterministic import extract_claims
from g2t_aml.eval.layer2_faithfulness import (
    CaseFaithfulness,
    aggregate_faithfulness,
    score_case,
    score_cases,
)
from g2t_aml.eval.types import ScoredCase, SystemOutput
from g2t_aml.facts.checkers import Claim, ClaimType
from g2t_aml.facts.extractor import extract_facts


@pytest.fixture(scope="module")
def facts():
    return extract_facts(as_laundering_stream(fan_out_case(width=9), "fan_out"))


def scored(facts, narrative: str, **kwargs) -> ScoredCase:
    return ScoredCase(
        output=SystemOutput(
            system=kwargs.pop("system", "test"),
            case_id=facts.case_id,
            narrative=narrative,
            **kwargs,
        ),
        facts=facts,
    )


def numeric_claim(field_path: str, value: object, span=(0, 1)) -> Claim:
    return Claim(
        text_span=span,
        field_path=field_path,
        claim_type=ClaimType.NUMERIC,
        value=value,
        raw_text=str(value),
    )


def case_with(n_claims: int, n_contradicted: int, **kwargs) -> CaseFaithfulness:
    """Build a per-case result directly, for testing the aggregation arithmetic."""
    defaults = {
        "case_id": kwargs.pop("case_id", "c1"),
        "system": kwargs.pop("system", "test"),
        "typology": kwargs.pop("typology", "fan_out"),
        "dataset": kwargs.pop("dataset", "amlworld_hi_small"),
        "seed": kwargs.pop("seed", None),
        "stream": kwargs.pop("stream", "balanced"),
        "n_claims": n_claims,
        "n_supported": n_claims - n_contradicted,
        "n_contradicted": n_contradicted,
        "n_unverifiable": 0,
        "n_salient_required": 4,
        "n_salient_covered": 4,
        "n_numeric": n_claims,
        "n_numeric_correct": n_claims - n_contradicted,
        "typology_correct": True,
        "ordering_correct": None,
    }
    defaults.update(kwargs)
    return CaseFaithfulness(**defaults)


# ------------------------------------------------------- the arithmetic ---


def test_the_rates_are_the_definitions_in_the_brief(facts):
    # Eight claims: five right, one wrong, two the record cannot speak to.
    claims = [
        numeric_claim("structure.n_nodes", 10),
        numeric_claim("structure.n_edges", 9),
        numeric_claim("focal_entity.out_degree", 9),
        numeric_claim("focal_entity.in_degree", 0),
        numeric_claim("motifs.fan_out.width", 9),
        numeric_claim("labels.n_illicit_counterparties", 999),  # wrong
        numeric_claim("flow.cross_border", 1),  # never available
        Claim((0, 1), None, ClaimType.NUMERIC, 73, "73"),  # unbacked
    ]

    result = score_case(scored(facts, "irrelevant to claim-level scoring"), claims)

    assert result.n_claims == 8
    assert (result.n_supported, result.n_contradicted, result.n_unverifiable) == (5, 1, 2)
    assert result.fact_precision == pytest.approx(5 / 8)
    assert result.hallucination_rate == pytest.approx(1 / 8)
    assert result.unverifiable_rate == pytest.approx(2 / 8)
    assert not result.zero_hallucination


def test_fact_f1_is_the_harmonic_mean_of_precision_and_coverage():
    case = case_with(n_claims=4, n_contradicted=1, n_salient_required=4, n_salient_covered=2)
    assert case.fact_precision == 0.75
    assert case.fact_coverage == 0.5
    assert case.fact_f1 == pytest.approx(2 * 0.75 * 0.5 / 1.25)


def test_numeric_accuracy_is_none_rather_than_perfect_when_no_number_was_stated(facts):
    # A system that avoids numbers must not accumulate perfect scores it never earned.
    result = score_case(scored(facts, "The account is the subject of this referral."), [])
    assert result.numeric_accuracy is None


def test_a_claim_naming_no_field_is_excluded_from_numeric_accuracy(facts):
    # An unbacked quantity is already counted in the unverifiable rate. Counting it here
    # too would make Numeric Accuracy report invention where it is meant to report
    # arithmetic.
    claims = [
        numeric_claim("structure.n_nodes", 10),
        Claim((0, 1), None, ClaimType.NUMERIC, 73, "73"),
    ]
    result = score_case(scored(facts, "x"), claims)
    assert result.n_numeric == 1
    assert result.numeric_accuracy == 1.0


def test_a_contradicted_mention_does_not_count_as_coverage(facts):
    # The property that stops Fact F1 being maximisable by asserting every salient field
    # wrongly.
    right = score_case(scored(facts, "x"), [numeric_claim("structure.n_nodes", 10)])
    wrong = score_case(scored(facts, "x"), [numeric_claim("structure.n_nodes", 999)])

    assert wrong.n_salient_covered < right.n_salient_covered


def test_an_empty_narrative_is_perfectly_precise_and_uncovered(facts):
    # Reported plainly rather than special-cased: a narrative that asserts nothing is
    # perfectly precise and useless, and Fact F1 is what refuses to reward it.
    result = score_case(scored(facts, "Nothing whatsoever."), [])
    assert result.fact_precision == 1.0
    assert result.zero_hallucination
    assert result.fact_coverage < 1.0
    assert result.fact_f1 < 1.0


def test_text_level_findings_are_counted_as_claims(facts):
    # A narrative can be arithmetically perfect and still assert guilt. Leaving that out
    # of both numerator and denominator would let a system with a guilt overclaim in
    # every narrative report 100% Zero-Hallucination.
    clean = score_case(scored(facts, "The subject account dispersed funds."), [])
    guilty = score_case(
        scored(facts, "The account holder is money laundering through these transfers."), []
    )

    assert clean.zero_hallucination
    assert not guilty.zero_hallucination
    assert guilty.n_critical == 1


# ------------------------------------------------- the headline metric ---


def test_zero_hallucination_is_per_narrative_not_averaged_precision():
    # The distinction the headline exists for. Both slices average 90% precision; one is
    # a system that puts a single error into every narrative, the other is perfect on
    # nine cases and badly wrong on one. Averaged precision cannot tell them apart.
    spread = [case_with(n_claims=10, n_contradicted=1, case_id=f"c{i}") for i in range(10)]
    concentrated = [case_with(n_claims=10, n_contradicted=0, case_id=f"c{i}") for i in range(9)]
    concentrated.append(case_with(n_claims=10, n_contradicted=10, case_id="c9"))

    a = aggregate_faithfulness(spread)
    b = aggregate_faithfulness(concentrated)

    assert a.fact_precision == pytest.approx(b.fact_precision)
    assert a.zero_hallucination_rate == 0.0
    assert b.zero_hallucination_rate == pytest.approx(0.9)


def test_the_macro_mean_and_the_pooled_ratio_are_both_reported():
    # They differ whenever narratives carry different claim counts, and the report shows
    # both so the choice is visible rather than taken on trust.
    cases = [
        case_with(n_claims=100, n_contradicted=50, case_id="long"),
        case_with(n_claims=2, n_contradicted=0, case_id="short"),
    ]
    aggregate = aggregate_faithfulness(cases)

    assert aggregate.fact_precision == pytest.approx((0.5 + 1.0) / 2)
    assert aggregate.pooled_fact_precision == pytest.approx(52 / 102)


def test_narratives_with_no_claims_are_counted_prominently():
    # A rising count here is the signature of an extractor failure masquerading as a
    # quality improvement: no claims means perfect precision and perfect
    # zero-hallucination.
    cases = [case_with(n_claims=0, n_contradicted=0, case_id="a"), case_with(4, 0, case_id="b")]
    assert aggregate_faithfulness(cases).n_narratives_with_no_claims == 1


def test_typology_accuracy_is_none_when_no_case_carries_a_ground_truth_typology():
    cases = [case_with(4, 0, typology_correct=None, case_id=f"c{i}") for i in range(3)]
    assert aggregate_faithfulness(cases).typology_accuracy is None


def test_typology_accuracy_penalises_naming_no_typology(facts):
    # "Correct typology named" means named. A narrative that names none has not named
    # the right one.
    silent = score_case(scored(facts, "The account dispersed funds widely."), [])
    assert silent.typology_correct is False


def test_ordering_accuracy_is_none_when_the_narrative_makes_no_ordering_claim(facts):
    assert score_case(scored(facts, "x"), []).ordering_correct is None


def test_critical_error_rate_is_per_narrative():
    cases = [
        case_with(4, 0, case_id="a", results=()),
        case_with(4, 0, case_id="b", results=()),
    ]
    assert aggregate_faithfulness(cases).critical_error_rate == 0.0


def test_aggregating_an_empty_slice_does_not_raise():
    aggregate = aggregate_faithfulness([], system="nobody")
    assert aggregate.n_cases == 0
    assert aggregate.zero_hallucination_rate == 0.0
    assert aggregate.system == "nobody"


def test_a_mixed_slice_is_labelled_mixed():
    cases = [case_with(4, 0, system="a", case_id="x"), case_with(4, 0, system="b", case_id="y")]
    assert aggregate_faithfulness(cases).system == "mixed"


# ------------------------------------------------------ through the API ---


def test_score_cases_runs_the_whole_pipeline(facts):
    narratives = [
        "The subject account sent funds to 9 distinct counterparties.",
        "The subject account sent funds to 40 distinct counterparties.",
    ]
    pairs = [(scored(facts, text), extract_claims(text, facts).claims) for text in narratives]

    results = score_cases(pairs)

    assert [r.zero_hallucination for r in results] == [True, False]
    assert aggregate_faithfulness(results).zero_hallucination_rate == 0.5


def test_a_mispaired_case_and_record_is_refused(facts):
    mismatched = ScoredCase(
        output=SystemOutput(system="test", case_id="a-different-case", narrative="x"),
        facts=facts,
    )
    with pytest.raises(Exception, match="the pairing is wrong"):
        _ = mismatched.case_id
