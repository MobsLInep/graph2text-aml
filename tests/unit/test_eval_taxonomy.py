"""The H1-H9 classifier, with one deliberate example of each of the nine classes.

Nine classes, nine narratives, each written to trigger exactly one. The per-class table
is the most quotable thing the harness produces, so every cell in it needs a test that
would fail if the class stopped being assignable at all — which is the failure mode a
per-class *rate* cannot show, because a class that never fires reads as a system that
never makes that error.
"""

from __future__ import annotations

import pytest
from tests.factories import as_laundering_stream, fan_out_case

from g2t_aml.eval.claim_extraction.deterministic import extract_claims
from g2t_aml.eval.layer2_faithfulness import score_case
from g2t_aml.eval.taxonomy_scorer import (
    CRITICAL_IDS,
    EXCULPATORY_FACTS,
    HAND_LABEL_SAMPLE_SIZE,
    ClassifiedError,
    classify,
    omissions,
    sample_for_hand_labelling,
    score_taxonomy,
    validate_against_hand_labels,
)
from g2t_aml.eval.types import ScoredCase, SystemOutput
from g2t_aml.facts.extractor import extract_facts
from g2t_aml.facts.taxonomy import HallucinationClass

#: One narrative per class. Each is minimal: it asserts the one thing its class is about
#: and as little else as possible, so a test failure names the class rather than the
#: narrative.
NARRATIVE_BY_CLASS: dict[str, str] = {
    # An account identifier that is not in the case inventory.
    "H1": "Funds were forwarded to account 777|NOTREAL for onward distribution.",
    # A count the record refutes: the fan-out width is 9.
    "H2": "The subject account sent funds to 40 distinct counterparties.",
    # A timestamp far outside the case window.
    "H3": "The review window runs from 1999-01-01 00:00 to 1999-01-02 00:00.",
    # An entity type. No substrate carries one, so this is unevidenced by construction.
    "H4": "The receiving address is operated as a mixer.",
    # A typology the record does not carry: this case is a fan-out.
    "H5": "The activity is consistent with a simple cycle.",
    # A citation outside the whitelist.
    "H6": "The transfers breach the USD 42,000 mandatory disclosure threshold.",
    # Guilt rather than suspicion. A SAR is a referral, not a finding of fact.
    "H7": "The account holder is money laundering through these transfers.",
    # Motive, which no transaction graph can establish.
    "H8": "The transfers were deliberately structured to stay below reporting limits.",
    # H9 is detected by absence and has no narrative of its own; see its own test.
}


@pytest.fixture(scope="module")
def facts():
    return extract_facts(as_laundering_stream(fan_out_case(width=9), "fan_out"))


@pytest.fixture(scope="module")
def exculpatory_facts():
    """A case that genuinely carries exculpatory facts, which H9 needs to fire at all.

    The laundering-stream fixture has every counterparty flagged illicit and the focal
    account flagged too, so nothing about it weakens the suspicion and H9 correctly never
    fires. Omitting a fact that does not exist is not an omission.
    """
    return extract_facts(fan_out_case(width=9))


def scored(facts, narrative: str, system: str = "test") -> ScoredCase:
    return ScoredCase(
        output=SystemOutput(system=system, case_id=facts.case_id, narrative=narrative),
        facts=facts,
    )


def classes_for(facts, narrative: str) -> set[str]:
    """Return every hallucination class the narrative produces."""
    case = scored(facts, narrative)
    faithfulness = score_case(case, extract_claims(narrative, facts).claims)
    found = {
        assignment[0]
        for result in faithfulness.results
        if (assignment := classify(result)) is not None
    }
    return found


# ------------------------------------------------- one example per class ---


@pytest.mark.parametrize("class_id", sorted(NARRATIVE_BY_CLASS))
def test_each_class_has_a_narrative_that_triggers_it(facts, class_id):
    assert class_id in classes_for(facts, NARRATIVE_BY_CLASS[class_id])


def test_h9_fires_when_an_exculpatory_fact_is_omitted(exculpatory_facts):
    # H9 is the only class detected by absence: the narrative's failure is the sentence
    # it did not write, so there is no claim to attach the finding to. This case has no
    # illicit counterparty at all, which materially weakens the suspicion.
    narrative = "The subject account dispersed funds across a number of accounts."
    found = omissions(scored(exculpatory_facts, narrative), ())

    assert found
    assert {e.hallucination_class for e in found} == {"H9"}
    assert all(e.source == "omission" for e in found)
    assert all(e.field_path in {f.field_path for f in EXCULPATORY_FACTS} for e in found)


def test_h9_does_not_fire_for_a_field_the_narrative_does_mention(exculpatory_facts):
    # Mentioning the exculpatory fact discharges the omission. Stated through the real
    # pipeline rather than a fabricated CheckResult, so what discharges H9 is exactly
    # what Layer 2 counts as a claim.
    silent = "The subject account dispersed funds across a number of accounts."
    speaks = (
        "The subject account dispersed funds across a number of accounts. Of 9 "
        "counterparties in scope, 0 are associated with transactions previously "
        "flagged as illicit."
    )

    def omitted(narrative: str) -> set[str | None]:
        case = scored(exculpatory_facts, narrative)
        faithfulness = score_case(case, extract_claims(narrative, exculpatory_facts).claims)
        return {e.field_path for e in omissions(case, faithfulness.results)}

    before = omitted(silent)
    after = omitted(speaks)

    assert "labels.n_illicit_counterparties" in before
    assert "labels.n_illicit_counterparties" not in after


def test_every_taxonomy_class_is_reachable_by_the_classifier():
    # The guard against a class that exists in the enum, appears as a zero row in the
    # paper's table, and is in fact unassignable.
    covered = set(NARRATIVE_BY_CLASS) | {"H9"}
    assert covered == {h.ident for h in HallucinationClass}


# ------------------------------------------------------ critical errors ---


def test_the_critical_ids_are_exactly_h4_h6_h7():
    assert {"H4", "H6", "H7"} == CRITICAL_IDS


@pytest.mark.parametrize("class_id", ["H4", "H6", "H7"])
def test_a_critical_class_raises_the_critical_error_rate(facts, class_id):
    narrative = NARRATIVE_BY_CLASS[class_id]
    case = scored(facts, narrative)
    faithfulness = score_case(case, extract_claims(narrative, facts).claims)

    report = score_taxonomy([(case, faithfulness)], detect_omissions=False)

    assert report.critical_error_rate == 1.0
    assert report.critical_by_class[class_id] == 1.0


def test_a_non_critical_class_leaves_the_critical_rate_at_zero(facts):
    narrative = NARRATIVE_BY_CLASS["H2"]
    case = scored(facts, narrative)
    faithfulness = score_case(case, extract_claims(narrative, facts).claims)

    report = score_taxonomy([(case, faithfulness)], detect_omissions=False)

    assert report.rate_by_class["H2"] == 1.0
    assert report.critical_error_rate == 0.0


# ----------------------------------------------------- the refinement rule ---


def test_an_unverifiable_quantity_is_not_promoted_to_a_numeric_error(facts):
    # The distinction the module leans on hardest. A number the record cannot speak to
    # has not disagreed with anything; classing it H2 would move the largest bucket of
    # unverifiable claims into the hallucination count and roughly double every reported
    # hallucination rate for no reason but a classification choice.
    narrative = "An internal risk score of 73 was assigned to the subject."
    assert "H2" not in classes_for(facts, narrative)
    assert "H8" in classes_for(facts, narrative)


def test_a_supported_result_is_not_classified(facts):
    narrative = "The subject account sent funds to 9 distinct counterparties."
    case = scored(facts, narrative)
    faithfulness = score_case(case, extract_claims(narrative, facts).claims)
    supported = [r for r in faithfulness.results if r.verdict.value == "supported"]

    assert supported
    assert all(classify(r) is None for r in supported)


def test_the_classifier_records_where_a_class_came_from(facts):
    narrative = NARRATIVE_BY_CLASS["H2"]
    case = scored(facts, narrative)
    faithfulness = score_case(case, extract_claims(narrative, facts).claims)

    report = score_taxonomy([(case, faithfulness)], detect_omissions=False)
    sources = {e.source for e in report.errors if e.hallucination_class == "H2"}

    # H2 comes from the checker itself, not from this module's refinement. Splitting the
    # validation by source is what stops the checker's work flattering the refinement.
    assert sources == {"checker"}


# --------------------------------------------------------- cross-tabulation ---


def test_the_cross_tab_keys_class_by_typology(facts):
    narrative = NARRATIVE_BY_CLASS["H2"]
    case = scored(facts, narrative)
    faithfulness = score_case(case, extract_claims(narrative, facts).claims)

    report = score_taxonomy([(case, faithfulness)], detect_omissions=False)

    assert report.cross_tab["H2"] == {"fan_out": 1}
    assert set(report.cross_tab) == {h.ident for h in HallucinationClass}


def test_rates_are_per_narrative_not_per_claim(facts):
    # Two H2 findings in one narrative is one narrative with an H2, not two.
    narrative = "The subgraph comprises 40 accounts connected by 90 transactions."
    case = scored(facts, narrative)
    faithfulness = score_case(case, extract_claims(narrative, facts).claims)

    report = score_taxonomy([(case, faithfulness)], detect_omissions=False)

    assert report.by_class["H2"] == 2
    assert report.rate_by_class["H2"] == 1.0


def test_an_empty_slice_reports_zeroes_rather_than_raising():
    report = score_taxonomy([])
    assert report.n_cases == 0
    assert report.critical_error_rate == 0.0
    assert set(report.by_class) == {h.ident for h in HallucinationClass}


# ------------------------------------------------- hand-label validation ---


def _error(case_id: str, class_id: str, start: int, source: str = "checker") -> ClassifiedError:
    return ClassifiedError(
        case_id=case_id,
        system="test",
        typology="fan_out",
        hallucination_class=class_id,
        verdict="contradicted",
        source=source,
        field_path=None,
        text="x",
        reason="",
        span=(start, start + 1),
    )


def test_hand_label_validation_reports_overall_and_per_class_agreement():
    sample = [
        _error("c1", "H2", 0),
        _error("c2", "H2", 0),
        _error("c3", "H6", 0),
        _error("c4", "H6", 0),
    ]
    labels = {
        "c1:0-1": "H2",
        "c2:0-1": "H2",
        "c3:0-1": "H6",
        "c4:0-1": "H2",  # a disagreement, in the class the paper leans on
    }

    report = validate_against_hand_labels(sample, labels)

    assert report.n_labelled == 4
    assert report.accuracy == 0.75
    assert report.per_class["H2"]["accuracy"] == 1.0
    assert report.per_class["H6"]["accuracy"] == 0.5
    assert report.kappa is not None
    assert report.kappa_band is not None


def test_hand_label_validation_splits_agreement_by_source():
    sample = [_error("c1", "H2", 0, "checker"), _error("c2", "H3", 0, "refined")]
    labels = {"c1:0-1": "H2", "c2:0-1": "H8"}

    report = validate_against_hand_labels(sample, labels)

    assert report.by_source["checker"] == 1.0
    assert report.by_source["refined"] == 0.0


def test_hand_label_validation_ignores_unlabelled_errors():
    report = validate_against_hand_labels([_error("c1", "H2", 0)], {})
    assert report.n_labelled == 0
    assert report.kappa is None


def test_a_hand_label_outside_the_nine_classes_is_refused():
    with pytest.raises(KeyError, match="unknown hallucination class"):
        validate_against_hand_labels([_error("c1", "H2", 0)], {"c1:0-1": "H12"})


def test_the_hand_label_sample_is_stratified_and_reproducible():
    errors = [_error(f"c{i}", "H2", i) for i in range(400)] + [
        _error(f"d{i}", "H6", i) for i in range(10)
    ]

    drawn = sample_for_hand_labelling(errors, n=100, seed=1)
    again = sample_for_hand_labelling(errors, n=100, seed=1)

    assert drawn == again
    assert len(drawn) == 100
    # Stratification is what makes the rare classes measurable at all: a uniform draw
    # from this pool would take ~2 of the 10 H6 errors.
    assert sum(1 for e in drawn if e.hallucination_class == "H6") == 10


def test_the_hand_label_sample_returns_everything_when_the_pool_is_small():
    errors = [_error(f"c{i}", "H2", i) for i in range(5)]
    assert len(sample_for_hand_labelling(errors, n=HAND_LABEL_SAMPLE_SIZE)) == 5
