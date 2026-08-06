"""The rating interface: the clock, the capture, and the blinding boundary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.factories import (
    as_laundering_stream,
    elliptic2_case,
    fan_out_case,
    view_of,
)

from g2t_aml.experiments.registry import system_ids
from g2t_aml.facts.extractor import extract_facts
from g2t_aml.facts.salience import salience_report
from g2t_aml.facts.vocab import load_vocabulary
from g2t_aml.human.caseloader import AnnotationCase
from g2t_aml.human.study_design import StudyItem, build_design
from g2t_aml.human.study_ui import (
    LIKERT_DIMENSIONS,
    BlurAwareTimer,
    RatingResponse,
    ResponseStore,
    ResponseStoreError,
    assert_no_system_identity,
    load_narratives,
    render_item,
)

NARRATIVE = "The subject account received funds from six counterparties within 22 hours."


def _case(graph) -> AnnotationCase:
    facts = extract_facts(graph)
    return AnnotationCase(
        case_id=facts.case_id,
        facts=facts,
        view=view_of(graph),
        salience=salience_report(facts, load_vocabulary()),
    )


@pytest.fixture
def amlworld_case() -> AnnotationCase:
    return _case(as_laundering_stream(fan_out_case(width=6), "fan_out"))


@pytest.fixture
def elliptic2_case_fixture() -> AnnotationCase:
    return _case(elliptic2_case())


def _item(case: AnnotationCase, dataset: str = "amlworld_hi_small") -> StudyItem:
    return StudyItem(
        item_id="abc123def4567890",
        rater_id="rater-01",
        case_id=case.case_id,
        dataset=dataset,
        position=0,
        round_index=0,
    )


# ------------------------------------------------------------------- timer ---


def test_timer_is_zero_before_it_starts():
    assert BlurAwareTimer().active_seconds(now=10.0) == 0.0


def test_timer_measures_elapsed_time():
    timer = BlurAwareTimer()
    timer.start(100.0)
    timer.stop(160.0)
    assert timer.active_seconds() == pytest.approx(60.0)


def test_timer_subtracts_a_hidden_period():
    """The whole point: a rater who switches tab is not reading."""
    timer = BlurAwareTimer()
    timer.start(0.0)
    timer.blur(10.0)
    timer.focus(310.0)  # five minutes away
    timer.stop(320.0)
    assert timer.active_seconds() == pytest.approx(20.0)
    assert timer.hidden_seconds == pytest.approx(300.0)
    assert timer.n_blurs == 1


def test_timer_subtracts_several_hidden_periods():
    timer = BlurAwareTimer()
    timer.start(0.0)
    for start, end in ((5.0, 15.0), (20.0, 50.0), (60.0, 61.0)):
        timer.blur(start)
        timer.focus(end)
    timer.stop(100.0)
    assert timer.active_seconds() == pytest.approx(100.0 - 41.0)
    assert timer.n_blurs == 3


def test_timer_reads_correctly_while_still_hidden():
    timer = BlurAwareTimer()
    timer.start(0.0)
    timer.blur(30.0)
    assert timer.active_seconds(now=200.0) == pytest.approx(30.0)


def test_stopping_while_hidden_closes_the_hidden_period():
    timer = BlurAwareTimer()
    timer.start(0.0)
    timer.blur(10.0)
    timer.stop(400.0)
    assert timer.active_seconds() == pytest.approx(10.0)


def test_start_is_idempotent_across_reruns():
    """Streamlit re-executes the script on every interaction."""
    timer = BlurAwareTimer()
    timer.start(0.0)
    timer.start(50.0)
    timer.stop(60.0)
    assert timer.active_seconds() == pytest.approx(60.0)


def test_a_double_blur_counts_once():
    timer = BlurAwareTimer()
    timer.start(0.0)
    timer.blur(10.0)
    timer.blur(12.0)
    timer.focus(20.0)
    timer.stop(30.0)
    assert timer.n_blurs == 1
    assert timer.active_seconds() == pytest.approx(20.0)


def test_focus_without_blur_changes_nothing():
    timer = BlurAwareTimer()
    timer.start(0.0)
    timer.focus(10.0)
    timer.stop(20.0)
    assert timer.active_seconds() == pytest.approx(20.0)


def test_reading_a_running_clock_without_now_is_refused():
    timer = BlurAwareTimer()
    timer.start(0.0)
    with pytest.raises(ValueError, match="needs `now`"):
        timer.active_seconds()


def test_stop_is_final():
    timer = BlurAwareTimer()
    timer.start(0.0)
    timer.stop(10.0)
    timer.stop(900.0)
    assert timer.active_seconds() == pytest.approx(10.0)


# --------------------------------------------------------------- rendering ---


def test_renders_an_amlworld_case(amlworld_case):
    rendered = render_item(
        _item(amlworld_case), amlworld_case, {"abc123def4567890": NARRATIVE}, total=10
    )
    assert rendered.narrative == NARRATIVE
    assert rendered.panel.sections
    assert rendered.graph.to_dict()["n_nodes_displayed"] > 0


def test_renders_an_elliptic2_case_without_inventing_monetary_facts(elliptic2_case_fixture):
    """Invariant 4, in pixels rather than in text."""
    rendered = render_item(
        _item(elliptic2_case_fixture, dataset="elliptic2"),
        elliptic2_case_fixture,
        {"abc123def4567890": NARRATIVE},
        total=10,
    )
    assert rendered.panel.section("Value") is None
    blob = json.dumps(rendered.panel.to_dict())
    for word in ("Dollar", "Euro", "USD", "EUR"):
        assert word not in blob


def test_rendering_refuses_an_item_with_no_narrative(amlworld_case):
    with pytest.raises(ResponseStoreError, match="no text"):
        render_item(_item(amlworld_case), amlworld_case, {}, total=10)


def test_rendering_refuses_a_case_that_is_not_the_item_s(amlworld_case, elliptic2_case_fixture):
    with pytest.raises(ResponseStoreError, match="but was given case"):
        render_item(
            _item(amlworld_case),
            elliptic2_case_fixture,
            {"abc123def4567890": NARRATIVE},
            total=10,
        )


# ---------------------------------------------------------------- blinding ---


def test_a_rendered_item_names_no_system_in_the_matrix(amlworld_case):
    """The blinding assertion, over data rather than over pixels."""
    rendered = render_item(
        _item(amlworld_case), amlworld_case, {"abc123def4567890": NARRATIVE}, total=10
    )
    assert_no_system_identity(rendered.to_dict(), [*system_ids(), "Bronze", "Silver"])


def test_the_blinding_assertion_actually_fires():
    with pytest.raises(AssertionError, match="names system"):
        assert_no_system_identity({"narrative": "produced by S1 today"}, ["S1"])


def test_the_blinding_assertion_does_not_fire_on_a_substring_of_an_id():
    """`B1` occurs inside a hex digest constantly; matching must be on word boundaries."""
    assert_no_system_identity({"item_id": "ab1cdeb1f0"}, ["B1", "S1"])


def test_the_narrative_pool_refuses_a_system_field(tmp_path):
    path = tmp_path / "narratives.jsonl"
    path.write_text(json.dumps({"item_id": "x", "narrative": "n", "system": "S1"}) + "\n")
    with pytest.raises(ResponseStoreError, match="unblinds the study"):
        load_narratives(path)


def test_the_narrative_pool_refuses_a_duplicate_item(tmp_path):
    path = tmp_path / "narratives.jsonl"
    path.write_text(
        json.dumps({"item_id": "x", "narrative": "a"})
        + "\n"
        + json.dumps({"item_id": "x", "narrative": "b"})
        + "\n"
    )
    with pytest.raises(ResponseStoreError, match="duplicate"):
        load_narratives(path)


def test_a_stored_response_carries_no_system(tmp_path):
    design, key = build_design(
        [f"c{i:03d}" for i in range(100)],
        ["S1", "S2", "B7", "B3", "Bronze"],
        [f"rater-{i:02d}" for i in range(10)],
        dataset="d",
        items_per_rater=60,
    )
    item = design.items[0]
    store = ResponseStore(root=tmp_path)
    store.append(_response(item.item_id, item.rater_id, item.case_id))
    blob = store.path_for(item.rater_id).read_text()
    for system in ("S1", "S2", "B7", "B3", "Bronze"):
        assert f'"{system}"' not in blob


# ------------------------------------------------------- capture and resume ---


def _response(item_id: str, rater_id: str = "rater-01", case_id: str = "c000") -> RatingResponse:
    return RatingResponse(
        item_id=item_id,
        rater_id=rater_id,
        case_id=case_id,
        position=0,
        is_repeat=False,
        factual_correctness=6,
        completeness=5,
        actionability=4,
        readability=7,
        regulatory_tone=3,
        would_file=True,
        seconds_to_usable_draft=123.4,
        presented_narrative=NARRATIVE,
        corrected_narrative=NARRATIVE.replace("six", "seven"),
        timing_source="browser",
    )


def test_both_versions_of_the_text_are_captured(tmp_path):
    store = ResponseStore(root=tmp_path)
    store.append(_response("item-1"))
    (stored,) = store.read("rater-01")
    assert stored.presented_narrative == NARRATIVE
    assert stored.corrected_narrative != NARRATIVE
    assert "seven" in stored.corrected_narrative


def test_completed_items_drive_save_and_resume(tmp_path):
    store = ResponseStore(root=tmp_path)
    assert store.completed_item_ids("rater-01") == set()
    store.append(_response("item-1"))
    store.append(_response("item-2"))
    assert store.completed_item_ids("rater-01") == {"item-1", "item-2"}


def test_the_store_is_append_only_and_keeps_the_latest(tmp_path):
    store = ResponseStore(root=tmp_path)
    store.append(_response("item-1"))
    revised = RatingResponse(**{**_response("item-1").__dict__, "factual_correctness": 1})
    store.append(revised)
    assert len(store.read("rater-01")) == 2
    (latest,) = (r for r in store.read_all() if r.item_id == "item-1")
    assert latest.factual_correctness == 1


def test_a_response_rejects_an_out_of_range_rating():
    with pytest.raises(ResponseStoreError, match="the scale is the integers"):
        RatingResponse(**{**_response("x").__dict__, "factual_correctness": 8})


def test_a_response_rejects_a_real_name_as_a_rater_id():
    with pytest.raises(ResponseStoreError, match="pseudonym"):
        RatingResponse(**{**_response("x").__dict__, "rater_id": "Alex Smith"})


def test_a_response_rejects_an_empty_presented_narrative():
    with pytest.raises(ResponseStoreError, match="rating of nothing"):
        RatingResponse(**{**_response("x").__dict__, "presented_narrative": "   "})


def test_a_response_rejects_an_unknown_timing_source():
    with pytest.raises(ResponseStoreError, match="timing_source"):
        RatingResponse(**{**_response("x").__dict__, "timing_source": "guess"})


def test_a_response_round_trips_through_its_serialisation():
    original = _response("item-1")
    assert RatingResponse.from_dict(original.to_dict()).to_dict() == original.to_dict()


# ------------------------------------------------------------------ scales ---


def test_every_dimension_has_three_anchors():
    """An unanchored scale makes its agreement statistic meaningless."""
    for dimension in LIKERT_DIMENSIONS:
        assert dimension.anchor_low.startswith("1 -")
        assert dimension.anchor_mid.startswith("4 -")
        assert dimension.anchor_high.startswith("7 -")
        assert len(dimension.anchor_low) > 40


def test_factual_correctness_leads_the_scales():
    """It is the dimension the automatic metric is validated against."""
    assert LIKERT_DIMENSIONS[0].key == "factual_correctness"


# ------------------------------------------------ the anchors cannot drift ---


def test_the_training_pack_reproduces_every_anchor_verbatim():
    """A rater told one definition and shown another has been given two scales.

    The anchors live in code and are quoted in `docs/human_study/rater_training.md`. This
    asserts the quotation is exact, so an edit to one without the other fails CI rather
    than silently giving the panel a different instrument from the one the interface uses.
    """
    pack = (
        Path(__file__).resolve().parents[2] / "docs" / "human_study" / "rater_training.md"
    ).read_text(encoding="utf-8")
    for dimension in LIKERT_DIMENSIONS:
        assert dimension.question in pack, f"{dimension.key}: question missing"
        for anchor in (dimension.anchor_low, dimension.anchor_mid, dimension.anchor_high):
            # The pack renders anchors in a table without the leading "1 - " marker.
            body = anchor.split(" - ", 1)[1]
            assert body in pack, f"{dimension.key}: anchor text drifted:\n{body}"
