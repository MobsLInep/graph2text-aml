"""Calibration: the gate, and why it is four separate gates rather than an average."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.factories import (
    as_laundering_stream,
    cycle_case,
    fan_out_case,
    gather_scatter_case,
    scatter_gather_case,
)

from g2t_aml.facts.extractor import extract_facts
from g2t_aml.human.calibration import (
    DIMENSIONS,
    CalibrationError,
    CalibrationItem,
    CalibrationSet,
    build_calibration_set,
    score_annotator,
)
from g2t_aml.human.store import Annotation

CLEAN = (
    "The subject account dispersed funds to several counterparties within the reviewed "
    "window. The activity appears consistent with the observed pattern and warrants "
    "further review."
)


def facts_for(typology):
    builders = {
        "fan_out": lambda: fan_out_case(width=5),
        "cycle": lambda: cycle_case(length=4),
        "gather_scatter": lambda: gather_scatter_case(),
        "scatter_gather": lambda: scatter_gather_case(),
    }
    return extract_facts(as_laundering_stream(builders[typology](), typology))


def calibration_set(n=4):
    typologies = ["fan_out", "cycle", "gather_scatter", "scatter_gather"][:n]
    return CalibrationSet(
        items=tuple(
            CalibrationItem(
                case_id=f"case-{i}",
                reference_typology=typology,
                reference_narrative=CLEAN,
                reference_mentioned=("focal_entity.id", "structure.n_nodes"),
                commentary=f"reference for {typology}",
            )
            for i, typology in enumerate(typologies)
        )
    )


def submissions(typologies, narratives=None, annotator="annotator-01", seconds=900.0):
    narratives = narratives or [CLEAN] * len(typologies)
    return [
        Annotation(
            case_id=f"case-{i}",
            dataset="amlworld_hi_small",
            annotator_id=annotator,
            narrative=narrative,
            seconds_spent=seconds,
            revision_count=2,
            typology_assigned=typology,
            is_calibration=True,
        )
        for i, (typology, narrative) in enumerate(zip(typologies, narratives, strict=True))
    ]


def facts_map(n=4):
    typologies = ["fan_out", "cycle", "gather_scatter", "scatter_gather"][:n]
    return {f"case-{i}": facts_for(t) for i, t in enumerate(typologies)}


# ------------------------------------------------------------ building a set ---


def test_the_set_spreads_across_typologies_rather_than_sampling_proportionally():
    """Ten items in proportion would be eight unclassified and calibrate nobody."""
    candidates = [(f"c{i}", "unclassified") for i in range(200)] + [
        (f"t{i}", t) for i, t in enumerate(["fan_out", "cycle", "stack", "bipartite"])
    ]
    built = build_calibration_set(candidates, n_cases=8, seed=1)
    typologies = {i.reference_typology for i in built.items}
    assert len(typologies) >= 4


def test_the_set_is_deterministic_under_a_seed():
    candidates = [(f"c{i}", "unclassified") for i in range(50)]
    a = build_calibration_set(candidates, n_cases=10, seed=3)
    b = build_calibration_set(list(reversed(candidates)), n_cases=10, seed=3)
    assert a.case_ids == b.case_ids


def test_too_few_candidates_is_refused():
    with pytest.raises(CalibrationError, match="cannot supply"):
        build_calibration_set([("c1", "fan_out")], n_cases=10)


def test_a_new_set_has_blank_references_for_the_lead_to_fill():
    built = build_calibration_set([(f"c{i}", "fan_out") for i in range(20)], n_cases=10)
    assert all(not i.reference_narrative for i in built.items)


def test_set_round_trips():
    built = calibration_set()
    assert CalibrationSet.from_dict(built.to_dict()).case_ids == built.case_ids


# --------------------------------------------------------------- scoring ---


def test_scoring_against_blank_references_is_refused():
    """It would pass every annotator against nothing."""
    blank = CalibrationSet(
        items=(
            CalibrationItem(case_id="case-0", reference_typology="fan_out", reference_narrative=""),
        )
    )
    with pytest.raises(CalibrationError, match="no reference narratives"):
        score_annotator("annotator-01", submissions(["fan_out"]), blank, facts_map(1))


def test_a_perfect_annotator_passes():
    result = score_annotator(
        "annotator-01",
        submissions(["fan_out", "cycle", "gather_scatter", "scatter_gather"]),
        calibration_set(),
        facts_map(),
        mentioned_by_case=dict.fromkeys(
            [f"case-{i}" for i in range(4)], ("focal_entity.id", "structure.n_nodes")
        ),
    )
    assert result.passed
    assert all(d.passed for d in result.dimensions)


def test_an_annotator_who_submitted_nothing_fails_every_dimension():
    result = score_annotator("annotator-02", [], calibration_set(), facts_map())
    assert not result.passed
    assert result.n_items == 0
    assert all(not d.passed for d in result.dimensions)


def test_an_incomplete_set_does_not_pass_however_good_the_items_are():
    """Eight of ten items is not a calibration."""
    result = score_annotator(
        "annotator-01",
        submissions(["fan_out", "cycle"]),
        calibration_set(),
        facts_map(2),
        mentioned_by_case={"case-0": ("focal_entity.id",), "case-1": ("focal_entity.id",)},
    )
    assert result.n_items == 2
    assert result.n_expected == 4
    assert not result.passed


def test_a_guilt_overclaim_fails_hedging_at_a_threshold_of_one():
    """One overclaim in a ten-item set is one too many."""
    guilty = CLEAN + " The account holder is guilty of laundering."
    result = score_annotator(
        "annotator-01",
        submissions(
            ["fan_out", "cycle", "gather_scatter", "scatter_gather"], [guilty] + [CLEAN] * 3
        ),
        calibration_set(),
        facts_map(),
    )
    hedging = next(d for d in result.dimensions if d.name == "hedging_compliance")
    assert hedging.score < 1.0
    assert not hedging.passed
    assert DIMENSIONS["hedging_compliance"] == 1.0


def test_a_systematic_typology_confusion_is_named_as_systematic():
    """Confusing the pair twice is a pattern, not a slip, and the guidance says so."""
    result = score_annotator(
        "annotator-01",
        submissions(["fan_out", "cycle", "scatter_gather", "gather_scatter"]),
        calibration_set(),
        facts_map(),
    )
    assert any("systematic" in g for g in result.guidance)


def test_a_single_typology_slip_is_not_called_systematic():
    result = score_annotator(
        "annotator-01",
        submissions(["cycle", "cycle", "gather_scatter", "scatter_gather"]),
        calibration_set(),
        facts_map(),
    )
    assert not any("systematic" in g for g in result.guidance)


def test_guidance_quotes_the_specific_item_and_the_reference():
    result = score_annotator(
        "annotator-01",
        submissions(["cycle", "cycle", "gather_scatter", "scatter_gather"]),
        calibration_set(),
        facts_map(),
    )
    typology = next(d for d in result.dimensions if d.name == "typology_agreement")
    assert any("case-0" in detail for detail in typology.detail)
    assert any("fan_out" in detail for detail in typology.detail)


def test_an_unmeasured_salience_dimension_scores_zero_rather_than_passing_by_default():
    result = score_annotator(
        "annotator-01",
        submissions(["fan_out", "cycle", "gather_scatter", "scatter_gather"]),
        calibration_set(),
        facts_map(),
        mentioned_by_case=None,
    )
    salience = next(d for d in result.dimensions if d.name == "salience_coverage")
    assert salience.score == 0.0
    assert not salience.passed
    assert "not measured" in salience.detail[0]


def test_dimensions_are_thresholded_separately_not_averaged():
    """A perfect score on three dimensions must not carry a failure on the fourth."""
    guilty = CLEAN + " This proves laundering."
    result = score_annotator(
        "annotator-01",
        submissions(
            ["fan_out", "cycle", "gather_scatter", "scatter_gather"], [guilty] + [CLEAN] * 3
        ),
        calibration_set(),
        facts_map(),
        mentioned_by_case=dict.fromkeys(
            [f"case-{i}" for i in range(4)], ("focal_entity.id", "structure.n_nodes")
        ),
    )
    typology = next(d for d in result.dimensions if d.name == "typology_agreement")
    assert typology.passed
    assert not result.passed


def test_mean_time_per_item_is_reported():
    result = score_annotator(
        "annotator-01",
        submissions(["fan_out", "cycle", "gather_scatter", "scatter_gather"], seconds=600.0),
        calibration_set(),
        facts_map(),
    )
    assert result.mean_minutes == pytest.approx(10.0)


def test_custom_thresholds_are_honoured():
    result = score_annotator(
        "annotator-01",
        submissions(["cycle", "cycle", "gather_scatter", "scatter_gather"]),
        calibration_set(),
        facts_map(),
        thresholds={"typology_agreement": 0.4},
    )
    typology = next(d for d in result.dimensions if d.name == "typology_agreement")
    assert typology.score == 0.75  # three of four match; case-0 says cycle, reference fan_out
    assert typology.passed


def test_result_serialises_and_summarises():
    import json

    result = score_annotator(
        "annotator-01",
        submissions(["fan_out", "cycle", "gather_scatter", "scatter_gather"]),
        calibration_set(),
        facts_map(),
    )
    json.dumps(result.to_dict())
    assert "Calibration" in result.summary()
