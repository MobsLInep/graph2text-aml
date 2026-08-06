"""The annotation store and the second-reviewer workflow."""

from __future__ import annotations

import json

import pytest

from g2t_aml.human.review import (
    Adjudication,
    Review,
    ReviewError,
    ReviewLog,
    ReviewVerdict,
)
from g2t_aml.human.store import (
    Annotation,
    AnnotationStore,
    AnnotationStoreError,
    FlagOutcome,
)
from g2t_aml.human.validation import LiveFlag, Severity


def annotation(case_id="c1", annotator="annotator-01", **kwargs):
    defaults = {
        "dataset": "amlworld_hi_small",
        "narrative": "A narrative about the case.",
        "seconds_spent": 900.0,
        "revision_count": 3,
    }
    return Annotation(case_id=case_id, annotator_id=annotator, **(defaults | kwargs))


def flag(rule="forbidden:guilt", hallucination_class="H7"):
    return LiveFlag(
        rule=rule,
        severity=Severity.CRITICAL,
        message="msg",
        span=(0, 5),
        hallucination_class=hallucination_class,
        excerpt="proves",
    )


# ---------------------------------------------------------- the annotation ---


def test_a_real_name_as_an_annotator_id_is_refused():
    """Invariant 8 includes who wrote which narrative."""
    with pytest.raises(AnnotationStoreError, match="valid pseudonym"):
        annotation(annotator="Jane Smith")


def test_an_email_as_an_annotator_id_is_refused():
    with pytest.raises(AnnotationStoreError, match="valid pseudonym"):
        annotation(annotator="jane@example.com")


def test_an_empty_narrative_is_refused():
    with pytest.raises(AnnotationStoreError, match="no narrative"):
        annotation(narrative="   ")


def test_negative_time_is_refused():
    with pytest.raises(AnnotationStoreError, match="cannot be negative"):
        annotation(seconds_spent=-1.0)


def test_submitted_at_is_stamped_automatically():
    assert annotation().submitted_at


def test_override_counts_are_derived_not_stored():
    record = annotation(
        flags=(
            FlagOutcome(flag=flag(), overridden=True),
            FlagOutcome(flag=flag("length:out_of_bounds", None), overridden=True),
            FlagOutcome(flag=flag("forbidden:motive", "H8"), overridden=False),
        )
    )
    assert record.n_overridden == 2
    assert record.n_critical_overridden == 1


def test_round_trips_through_its_serialised_form():
    record = annotation(flags=(FlagOutcome(flag=flag(), overridden=True, annotator_note="why"),))
    rebuilt = Annotation.from_dict(record.to_dict())
    assert rebuilt.narrative == record.narrative
    assert rebuilt.flags[0].overridden is True
    assert rebuilt.flags[0].annotator_note == "why"


# --------------------------------------------------------------- the store ---


def test_append_and_read(tmp_path):
    store = AnnotationStore(root=tmp_path)
    store.append(annotation("c1"))
    store.append(annotation("c2"))
    assert [a.case_id for a in store.read("annotator-01")] == ["c1", "c2"]


def test_reading_an_unknown_annotator_returns_empty(tmp_path):
    assert AnnotationStore(root=tmp_path).read("annotator-09") == []


def test_the_store_refuses_to_record_generated_text(tmp_path):
    """The mechanical half of 'never show an annotator model output'."""
    store = AnnotationStore(root=tmp_path)
    with pytest.raises(AnnotationStoreError, match="Gold is the independent human reference"):
        store.append(annotation(), bronze_narrative="Account X is the originating account...")


@pytest.mark.parametrize(
    "key", ["bronze_narrative", "silver_narrative", "model_narrative", "suggestion", "prefill"]
)
def test_every_forbidden_key_is_refused(tmp_path, key):
    store = AnnotationStore(root=tmp_path)
    with pytest.raises(AnnotationStoreError):
        store.append(annotation(), **{key: "text"})


def test_harmless_extra_keys_are_permitted(tmp_path):
    store = AnnotationStore(root=tmp_path)
    path = store.append(annotation(), session_id="abc")
    assert json.loads(path.read_text().splitlines()[0])["session_id"] == "abc"


def test_read_all_keeps_only_the_latest_submission_per_annotator_and_case(tmp_path):
    """A revisited item is history, not a second opinion."""
    store = AnnotationStore(root=tmp_path)
    store.append(annotation("c1", narrative="first draft"))
    store.append(annotation("c1", narrative="revised draft"))
    everything = store.read_all()
    assert len(everything) == 1
    assert everything[0].narrative == "revised draft"


def test_calibration_items_are_excluded_by_default(tmp_path):
    store = AnnotationStore(root=tmp_path)
    store.append(annotation("c1", is_calibration=True))
    store.append(annotation("c2"))
    assert [a.case_id for a in store.read_all()] == ["c2"]
    assert len(store.read_all(include_calibration=True)) == 2


def test_a_corrupt_line_is_reported_with_its_location(tmp_path):
    store = AnnotationStore(root=tmp_path)
    store.append(annotation())
    (tmp_path / "annotator-01.jsonl").write_text("not json\n")
    with pytest.raises(AnnotationStoreError, match=r":1:"):
        store.read("annotator-01")


def test_annotators_are_listed(tmp_path):
    store = AnnotationStore(root=tmp_path)
    store.append(annotation(annotator="annotator-02"))
    store.append(annotation(annotator="annotator-01"))
    assert store.annotators() == ("annotator-01", "annotator-02")


# -------------------------------------------------------------- the review ---


def test_a_disputed_review_without_an_adjudication_is_refused():
    with pytest.raises(ReviewError, match="no adjudication"):
        Review(
            case_id="c1",
            reviewer_id="reviewer-01",
            verdict=ReviewVerdict.ACCEPT,
            disagreements=("the width is wrong",),
        )


def test_a_non_accepting_review_without_an_adjudication_is_refused():
    with pytest.raises(ReviewError, match="no adjudication"):
        Review(case_id="c1", reviewer_id="reviewer-01", verdict=ReviewVerdict.REVISE)


def test_an_adjudication_without_a_rationale_is_refused():
    """The rationale is where this phase's qualitative findings come from."""
    with pytest.raises(ReviewError, match="decision and a rationale"):
        Adjudication(decided_by="lead-01", decision="keep", rationale="  ")


def test_a_clean_accept_needs_no_adjudication():
    review = Review(case_id="c1", reviewer_id="reviewer-01", verdict=ReviewVerdict.ACCEPT)
    assert review.accepted


def test_a_reviewer_who_annotated_the_case_is_refused():
    review = Review(case_id="c1", reviewer_id="annotator-01", verdict=ReviewVerdict.ACCEPT)
    with pytest.raises(ReviewError, match="both annotated and reviewed"):
        review.validate_against(("annotator-01",))


def test_a_double_annotated_case_must_name_a_chosen_annotator():
    review = Review(case_id="c1", reviewer_id="reviewer-01", verdict=ReviewVerdict.ACCEPT)
    with pytest.raises(ReviewError, match="names no chosen_annotator"):
        review.validate_against(("annotator-01", "annotator-02"))


def test_the_chosen_annotator_must_have_annotated_the_case():
    review = Review(
        case_id="c1",
        reviewer_id="reviewer-01",
        verdict=ReviewVerdict.ACCEPT,
        chosen_annotator="annotator-09",
    )
    with pytest.raises(ReviewError, match="did not annotate"):
        review.validate_against(("annotator-01", "annotator-02"))


def test_an_adjudicator_who_was_party_to_the_dispute_is_refused():
    review = Review(
        case_id="c1",
        reviewer_id="reviewer-01",
        verdict=ReviewVerdict.ACCEPT,
        disagreements=("x",),
        adjudication=Adjudication(decided_by="annotator-01", decision="keep", rationale="because"),
    )
    with pytest.raises(ReviewError, match="were party to"):
        review.validate_against(("annotator-01",))


def test_an_independent_review_of_a_double_annotated_case_validates():
    review = Review(
        case_id="c1",
        reviewer_id="reviewer-01",
        verdict=ReviewVerdict.ACCEPT,
        chosen_annotator="annotator-02",
        disagreements=("x",),
        adjudication=Adjudication(decided_by="lead-01", decision="keep", rationale="because"),
    )
    review.validate_against(("annotator-01", "annotator-02"))


def test_review_round_trips_through_its_serialised_form():
    review = Review(
        case_id="c1",
        reviewer_id="reviewer-01",
        verdict=ReviewVerdict.REJECT,
        adjudication=Adjudication(decided_by="lead-01", decision="drop", rationale="wrong case"),
    )
    rebuilt = Review.from_dict(review.to_dict())
    assert rebuilt.verdict is ReviewVerdict.REJECT
    assert rebuilt.adjudication is not None
    assert rebuilt.adjudication.rationale == "wrong case"


# ----------------------------------------------------------- the review log ---


def test_the_log_appends_and_reads_back(tmp_path):
    log = ReviewLog(path=tmp_path / "reviews.jsonl")
    log.append(Review(case_id="c1", reviewer_id="reviewer-01", verdict=ReviewVerdict.ACCEPT))
    log.append(Review(case_id="c2", reviewer_id="reviewer-01", verdict=ReviewVerdict.ACCEPT))
    assert [r.case_id for r in log.read()] == ["c1", "c2"]


def test_a_re_reviewed_case_takes_its_latest_review_and_keeps_the_earlier_one(tmp_path):
    log = ReviewLog(path=tmp_path / "reviews.jsonl")
    log.append(
        Review(
            case_id="c1",
            reviewer_id="reviewer-01",
            verdict=ReviewVerdict.REVISE,
            adjudication=Adjudication(decided_by="lead-01", decision="revise", rationale="r"),
        )
    )
    log.append(Review(case_id="c1", reviewer_id="reviewer-01", verdict=ReviewVerdict.ACCEPT))
    assert len(log.read()) == 2
    assert log.latest_by_case()["c1"].accepted


def test_an_absent_log_reads_as_empty(tmp_path):
    assert ReviewLog(path=tmp_path / "nothing.jsonl").read() == []
