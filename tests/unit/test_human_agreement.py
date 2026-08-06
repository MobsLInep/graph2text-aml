"""Agreement statistics, checked against hand-computed fixtures.

Every statistic here is verified against a value worked out by hand in the test, not
against another implementation. An agreement number computed by two versions of the same
misunderstanding agrees with itself.
"""

from __future__ import annotations

import pytest

from g2t_aml.human.agreement import (
    cohens_kappa,
    is_double_annotated,
    jaccard,
    krippendorff_alpha_nominal,
    measure_agreement,
    token_f1,
)
from g2t_aml.human.store import Annotation


def annotation(case_id, annotator, typology="fan_out", narrative="A narrative.", seconds=900.0):
    return Annotation(
        case_id=case_id,
        dataset="amlworld_hi_small",
        annotator_id=annotator,
        narrative=narrative,
        seconds_spent=seconds,
        revision_count=1,
        typology_assigned=typology,
    )


# ------------------------------------------------------- Cohen's kappa ---


def test_perfect_agreement_over_two_labels_is_one():
    a = ["fan_out", "fan_in", "fan_out", "fan_in"]
    assert cohens_kappa(a, list(a)) == pytest.approx(1.0)


def test_hand_computed_kappa():
    """po = 6/8 = 0.75.

    Rater A: 5 fan_out, 3 fan_in.  Rater B: 5 fan_out, 3 fan_in.
    pe = (5/8)(5/8) + (3/8)(3/8) = 0.390625 + 0.140625 = 0.53125
    kappa = (0.75 - 0.53125) / (1 - 0.53125) = 0.21875 / 0.46875 = 0.466666...
    """
    a = ["fan_out"] * 5 + ["fan_in"] * 3
    b = ["fan_out"] * 4 + ["fan_in"] + ["fan_out"] + ["fan_in"] * 2
    assert sorted(b) == sorted(a)
    assert sum(x == y for x, y in zip(a, b, strict=True)) == 6
    assert cohens_kappa(a, b) == pytest.approx(0.4666666, abs=1e-6)


def test_the_single_label_degenerate_case_returns_one_rather_than_a_nan():
    """Both raters answer 'unclassified' every time.

    Expected agreement is also 1.0, so the usual formula is 0/0. The documented
    resolution is 1.0 — they did agree on everything — and the point of the test is that
    a NaN never reaches a report.
    """
    a = ["unclassified"] * 10
    assert cohens_kappa(a, list(a)) == pytest.approx(1.0)


def test_agreeing_at_chance_on_a_skewed_distribution_scores_near_zero():
    """The reason chance correction matters here: `unclassified` is 88% of the split."""
    a = ["unclassified"] * 9 + ["fan_out"]
    b = ["unclassified"] * 8 + ["fan_out", "unclassified"]
    assert cohens_kappa(a, b) < 0.2


def test_systematic_disagreement_is_negative():
    a = ["fan_out", "fan_in"] * 4
    b = ["fan_in", "fan_out"] * 4
    assert cohens_kappa(a, b) < 0


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="differ in length"):
        cohens_kappa(["a"], ["a", "b"])


def test_empty_input_raises():
    with pytest.raises(ValueError, match="no items"):
        cohens_kappa([], [])


# ------------------------------------------------- Krippendorff's alpha ---


def test_alpha_is_one_when_every_doubled_unit_is_unanimous_across_two_labels():
    units = [["fan_out", "fan_out"], ["fan_in", "fan_in"]]
    assert krippendorff_alpha_nominal(units) == pytest.approx(1.0)


def test_hand_computed_alpha_on_a_single_disagreeing_pair():
    """Two units, each coded twice; one unit disagrees.

    n = 4 pairable values, values = {a: 3, b: 1}.
    n*Do = sum over units of (1/(m-1)) * (ordered disagreeing pairs)
         = unit1 (a,a): 0 ; unit2 (a,b): (1/1)*2 = 2  -> 2
    n*De = sum_c n_c(n - n_c) / (n - 1) = (3*1 + 1*3) / 3 = 2
    alpha = 1 - 2/2 = 0
    """
    units = [["a", "a"], ["a", "b"]]
    assert krippendorff_alpha_nominal(units) == pytest.approx(0.0)


def test_units_with_one_rater_are_skipped():
    units = [["a"], ["b"], ["a", "a"], ["b", "b"]]
    assert krippendorff_alpha_nominal(units) == pytest.approx(1.0)


def test_no_doubled_unit_gives_zero():
    assert krippendorff_alpha_nominal([["a"], ["b"]]) == 0.0


# ------------------------------------------------------------- Jaccard ---


def test_jaccard_hand_computed():
    assert jaccard({"a", "b", "c"}, {"b", "c", "d"}) == pytest.approx(2 / 4)


def test_two_empty_sets_agree_completely():
    """A case with no salient fields is not a disagreement."""
    assert jaccard(set(), set()) == 1.0


def test_disjoint_sets_score_zero():
    assert jaccard({"a"}, {"b"}) == 0.0


# ------------------------------------------------------------- token F1 ---


def test_identical_narratives_score_one():
    assert token_f1(
        "dispersed funds across nine accounts", "dispersed funds across nine accounts"
    ) == pytest.approx(1.0)


def test_two_faithful_but_differently_worded_narratives_score_low():
    a = "The subject dispersed 9,435 Canadian Dollar to 2 counterparties over 15.7 hours."
    b = "Across 15.7 hours, 2 recipients received 9,435 Canadian Dollar from the account."
    assert 0.0 < token_f1(a, b) < 0.8


def test_empty_input_scores_zero():
    assert token_f1("", "anything") == 0.0


# ------------------------------------------------- double-annotation set ---


def test_assignment_is_deterministic():
    assert is_double_annotated("case-x", 0.15) == is_double_annotated("case-x", 0.15)


def test_assignment_rate_is_approximately_the_requested_share():
    ids = [f"amlworld_hi_small-{i:08x}" for i in range(4000)]
    rate = sum(is_double_annotated(c, 0.15) for c in ids) / len(ids)
    assert 0.13 < rate < 0.17


def test_rate_zero_selects_nothing_and_rate_one_selects_everything():
    ids = [f"case-{i}" for i in range(200)]
    assert not any(is_double_annotated(c, 0.0) for c in ids)
    assert all(is_double_annotated(c, 1.0) for c in ids)


def test_an_out_of_range_rate_raises():
    with pytest.raises(ValueError, match="outside"):
        is_double_annotated("case", 1.5)


def test_adding_cases_does_not_change_existing_assignments():
    """The set is fixed before anyone starts and survives the sample being extended."""
    before = {c: is_double_annotated(c, 0.15) for c in (f"case-{i}" for i in range(50))}
    after = {c: is_double_annotated(c, 0.15) for c in (f"case-{i}" for i in range(500))}
    assert all(after[c] == v for c, v in before.items())


# ------------------------------------------------------------- the report ---


def test_pairs_are_found_and_singles_counted():
    annotations = [
        annotation("c1", "annotator-01"),
        annotation("c1", "annotator-02"),
        annotation("c2", "annotator-01"),
    ]
    report = measure_agreement(annotations)
    assert report.n_double_annotated == 1
    assert report.n_single_annotated == 1


def test_typology_confusion_is_broken_out_by_pair():
    annotations = [
        annotation("c1", "annotator-01", "gather_scatter"),
        annotation("c1", "annotator-02", "scatter_gather"),
        annotation("c2", "annotator-01", "gather_scatter"),
        annotation("c2", "annotator-02", "scatter_gather"),
    ]
    report = measure_agreement(annotations)
    assert report.typology_confusion == {"gather_scatter/scatter_gather": 2}


def test_content_overlap_uses_the_supplied_alignment():
    annotations = [annotation("c1", "annotator-01"), annotation("c1", "annotator-02")]
    report = measure_agreement(
        annotations,
        mentioned_by={
            ("c1", "annotator-01"): ("a", "b", "c"),
            ("c1", "annotator-02"): ("b", "c", "d"),
        },
    )
    assert report.mean_content_jaccard == pytest.approx(0.5)


def test_two_annotations_from_one_annotator_on_one_case_raise():
    """Pairing someone with themselves would inflate every statistic."""
    annotations = [annotation("c1", "annotator-01"), annotation("c1", "annotator-01")]
    with pytest.raises(ValueError, match="two annotations from one annotator"):
        measure_agreement(annotations)


def test_per_annotator_mean_time_is_reported():
    annotations = [
        annotation("c1", "annotator-01", seconds=600.0),
        annotation("c2", "annotator-01", seconds=1200.0),
    ]
    report = measure_agreement(annotations)
    assert report.per_annotator["annotator-01"]["mean_minutes"] == pytest.approx(15.0)


def test_report_serialises_and_summarises():
    import json

    report = measure_agreement([annotation("c1", "annotator-01"), annotation("c1", "annotator-02")])
    json.dumps(report.to_dict())
    assert "kappa" in report.summary()
    assert "legitimate variance" in report.summary()


def test_an_empty_set_reports_zeroes_rather_than_raising():
    report = measure_agreement([])
    assert report.n_double_annotated == 0
    assert report.kappa == 0.0
