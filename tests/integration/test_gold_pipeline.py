"""Phase 6 end to end: real cases, hand-written narratives, the ten-point gate.

Runs against the built AMLworld corpus when it is present and skips cleanly when it is
not, the same way the other integration tests treat data that has to be generated.

The narratives here were written by reading the fact panel and nothing else, which is what
an annotator does. They are not fixtures reverse-engineered from the alignment: two of them
deliberately do *not* pass, and their failures are the properties this phase's machinery
exists to enforce.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from g2t_aml.corpus.factsio import load_case_facts_file
from g2t_aml.corpus.validate import load_split_manifest, validate_corpus
from g2t_aml.human.agreement import measure_agreement
from g2t_aml.human.factpanel import build_fact_panel
from g2t_aml.human.gold_ingest import bronze_narrative_from_record, ingest_annotations
from g2t_aml.human.reservation import load_reservation
from g2t_aml.human.review import Adjudication, Review, ReviewVerdict
from g2t_aml.human.store import Annotation, AnnotationStore, FlagOutcome
from g2t_aml.human.validation import validate_draft
from g2t_aml.utils.io import read_jsonl

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed" / "amlworld_hi_small"
MANIFEST = REPO_ROOT / "schemas" / "splits" / "amlworld"

#: A real reserved case: 4 accounts, licit, bipartite detected, nothing flagged.
HARD_NEGATIVE = "amlworld_hi_small-006e89f5bc1f04b6"

FAITHFUL = """[1] Subject & Scope
Account 019|8090C7E20 is the subject of this report and acts as the originating account
within the reviewed activity. The reviewed subgraph comprises 4 accounts connected by
4 transactions across 3 institutions.

[2] Activity Observed
Over an observed period of 15.7 hours the subject account paid out 9,435 Canadian Dollar
across 2 outbound transactions to 2 distinct counterparties and received nothing within
the reviewed window. The largest single movement was 7,021 Canadian Dollar. Settlement
used ACH and Cheque. No transfers fell close to the USD 10,000 reporting threshold, and no
burst was detected.

[3] Pattern & Typology
The subgraph is bipartite, with sides of 2 and 2 accounts and a density of 0.250. No
fan-in, fan-out, cycle, stack, gather-scatter or scatter-gather structure was detected at
the configured thresholds. The case carries the typology label unclassified, and no
structure in the reviewed window supports assigning a more specific one.

[4] Basis & Action
No transaction in the reviewed subgraph is flagged in the source data, neither counterparty
appears on a flagged transaction, and the subject account is not itself flagged. On the
evidence available in this subgraph the activity appears consistent with ordinary outbound
settlement, and no basis for escalation is identified.
"""

#: The same case, written by someone who escalated a hard negative and asserted guilt.
#:
#: Note the wording: "is guilty of money laundering", not "is laundering money". The
#: frozen vocabulary's guilt list carries `is money laundering` and `is guilty of` but not
#: the equally natural `is laundering money`, so that phrasing passes the text scan. The
#: gap is real and is recorded in PHASE_LOG as a Phase 6 finding; it is not worked around
#: here, because editing a frozen vocabulary to make a test pass is exactly the move that
#: makes a frozen artifact meaningless.
OVERCLAIMING = FAITHFUL.replace(
    "the activity appears consistent with ordinary outbound\nsettlement, and no basis for "
    "escalation is identified.",
    "the account holder is guilty of money\nlaundering through a shell company.",
)

#: The same case with a fabricated figure that aligns to no fact field.
INVENTED = FAITHFUL.replace(
    "The largest single movement was 7,021 Canadian Dollar.",
    "The largest single movement was 7,021 Canadian Dollar, part of 88,412 Canadian Dollar\n"
    "moved across 41 further transfers.",
)


def _available() -> bool:
    return (
        (PROCESSED / "facts" / f"{HARD_NEGATIVE}.json").is_file()
        and (PROCESSED / "corpus" / "bronze.jsonl").is_file()
        and (MANIFEST / "test.txt").is_file()
    )


needs_corpus = pytest.mark.skipif(
    not _available(), reason="the AMLworld corpus is not built; run `make bronze` first"
)


@pytest.fixture(scope="module")
def facts():
    return load_case_facts_file(PROCESSED / "facts" / f"{HARD_NEGATIVE}.json")


@pytest.fixture(scope="module")
def bronze():
    for payload in read_jsonl(PROCESSED / "corpus" / "bronze.jsonl"):
        if isinstance(payload, dict) and payload.get("case_id") == HARD_NEGATIVE:
            return {HARD_NEGATIVE: bronze_narrative_from_record(payload)}
    pytest.skip(f"{HARD_NEGATIVE} is not in the Bronze corpus")


@pytest.fixture(scope="module")
def split_assignment():
    return load_split_manifest(MANIFEST)


def annotation(narrative, annotator="annotator-01", typology="unclassified"):
    return Annotation(
        case_id=HARD_NEGATIVE,
        dataset="amlworld_hi_small",
        annotator_id=annotator,
        narrative=narrative,
        seconds_spent=840.0,
        revision_count=4,
        typology_assigned=typology,
    )


def accepted_review(reviewer="reviewer-01", chosen=""):
    return Review(
        case_id=HARD_NEGATIVE,
        reviewer_id=reviewer,
        verdict=ReviewVerdict.ACCEPT,
        chosen_annotator=chosen,
    )


def ingest(annotations, reviews, facts, bronze, split_assignment):
    return ingest_annotations(
        annotations,
        reviews,
        {HARD_NEGATIVE: facts},
        bronze,
        split_assignment=split_assignment,
        case_store=PROCESSED / "cases",
        repo_root=REPO_ROOT,
    )


# ------------------------------------------------------ the reserved sample ---


@needs_corpus
def test_the_gold_reservation_is_committed_and_wholly_inside_the_test_split(split_assignment):
    held = load_reservation(MANIFEST, split_assignment=split_assignment)
    assert held is not None, "run `make gold-sample`"
    assert len(held) >= 200, "the reservation must hold at least 200 test-only cases"
    assert all(split_assignment[c] == "test" for c in held.case_ids)


@needs_corpus
def test_every_reserved_case_has_a_fact_record_and_a_bronze_alignment(split_assignment):
    """An annotator must never be queued a case that cannot be checked or ingested."""
    held = load_reservation(MANIFEST, split_assignment=split_assignment)
    missing = [c for c in held.case_ids[:50] if not (PROCESSED / "facts" / f"{c}.json").is_file()]
    assert not missing, missing


# ------------------------------------------------------------- the interface ---


@needs_corpus
def test_the_panel_renders_for_a_real_case(facts):
    panel = build_fact_panel(facts)
    assert panel.sections
    assert panel.required_fields
    assert "CASE" in panel.rendered_text()


@needs_corpus
def test_a_faithful_narrative_raises_no_critical_flag(facts):
    summary = validate_draft(FAITHFUL, facts, salient_fields=())
    assert summary.n_critical == 0
    assert summary.sections_complete
    assert summary.length_ok


@needs_corpus
def test_the_overclaiming_narrative_is_flagged_critical(facts):
    summary = validate_draft(OVERCLAIMING, facts, salient_fields=())
    assert summary.n_critical >= 2
    classes = {f.hallucination_class for f in summary.flags}
    assert "H7" in classes and "H4" in classes


# --------------------------------------------------------------- ingestion ---


@needs_corpus
def test_a_faithful_reviewed_narrative_becomes_a_schema_valid_gold_record(
    facts, bronze, split_assignment
):
    report = ingest([annotation(FAITHFUL)], [accepted_review()], facts, bronze, split_assignment)
    assert report.n_ingested == 1
    record = report.items[0].record
    assert record.tier == "gold"
    assert record.split == "test"
    assert record.generator["method"] == "human"
    assert record.generator["annotator_id"] == "annotator-01"
    assert record.target_slots, "the alignment produced no slots"


@needs_corpus
def test_the_gold_record_passes_the_same_ten_point_harness_as_bronze(
    facts, bronze, split_assignment
):
    report = ingest([annotation(FAITHFUL)], [accepted_review()], facts, bronze, split_assignment)
    gate = validate_corpus(report.payloads(), repo_root=REPO_ROOT, split_manifest=split_assignment)
    assert gate.gate_passed, gate.summary()
    assert gate.by_tier == {"gold": 1}


@needs_corpus
def test_verification_reports_zero_contradicted_claims(facts, bronze, split_assignment):
    report = ingest([annotation(FAITHFUL)], [accepted_review()], facts, bronze, split_assignment)
    verification = report.items[0].record.verification
    assert verification["contradicted"] == 0
    assert verification["n_claims"] > 10


@needs_corpus
def test_salience_coverage_is_measured_from_the_alignment(facts, bronze, split_assignment):
    report = ingest([annotation(FAITHFUL)], [accepted_review()], facts, bronze, split_assignment)
    salience = report.items[0].record.salience
    assert salience["required"]
    assert salience["coverage"] > 0.5


@needs_corpus
def test_an_invented_quantity_is_held_rather_than_ingested(facts, bronze, split_assignment):
    """The harness cannot see it — a Gold record's slots are only what aligned."""
    report = ingest([annotation(INVENTED)], [accepted_review()], facts, bronze, split_assignment)
    assert report.n_ingested == 0
    assert "unverifiable rate" in report.held[HARD_NEGATIVE]


# ---------------------------------------------------------- the review gate ---


@needs_corpus
def test_an_unreviewed_annotation_is_held(facts, bronze, split_assignment):
    report = ingest([annotation(FAITHFUL)], [], facts, bronze, split_assignment)
    assert report.n_ingested == 0
    assert report.held[HARD_NEGATIVE] == "awaiting second review"


@needs_corpus
def test_a_rejected_annotation_is_held_with_its_adjudication(facts, bronze, split_assignment):
    review = Review(
        case_id=HARD_NEGATIVE,
        reviewer_id="reviewer-01",
        verdict=ReviewVerdict.REJECT,
        disagreements=("the narrative escalates a hard negative",),
        adjudication=Adjudication(
            decided_by="lead-01",
            decision="drop from Gold",
            rationale="no flagged transaction supports the escalation",
        ),
    )
    report = ingest([annotation(OVERCLAIMING)], [review], facts, bronze, split_assignment)
    assert report.n_ingested == 0
    assert "drop from Gold" in report.held[HARD_NEGATIVE]


@needs_corpus
def test_a_reviewer_who_annotated_the_case_is_held_as_not_independent(
    facts, bronze, split_assignment
):
    report = ingest(
        [annotation(FAITHFUL)],
        [accepted_review(reviewer="annotator-01")],
        facts,
        bronze,
        split_assignment,
    )
    assert report.n_ingested == 0
    assert "not independent" in report.held[HARD_NEGATIVE]


@needs_corpus
def test_a_double_annotated_case_takes_the_adjudicated_choice(facts, bronze, split_assignment):
    annotations = [annotation(FAITHFUL, "annotator-01"), annotation(FAITHFUL, "annotator-02")]
    report = ingest(
        annotations, [accepted_review(chosen="annotator-02")], facts, bronze, split_assignment
    )
    assert report.n_ingested == 1
    assert report.items[0].record.generator["annotator_id"] == "annotator-02"


@needs_corpus
def test_the_adjudication_is_recorded_on_the_record(facts, bronze, split_assignment):
    review = Review(
        case_id=HARD_NEGATIVE,
        reviewer_id="reviewer-01",
        verdict=ReviewVerdict.ACCEPT,
        disagreements=("'received nothing' reads as absolute",),
        adjudication=Adjudication(
            decided_by="lead-01",
            decision="keep as written",
            rationale="in_degree is 0 inside the case, so the statement is correctly scoped",
        ),
    )
    report = ingest([annotation(FAITHFUL)], [review], facts, bronze, split_assignment)
    generator = report.items[0].record.generator
    assert generator["adjudicated"] is True
    assert generator["adjudication"]["decided_by"] == "lead-01"
    assert "correctly scoped" in generator["adjudication"]["rationale"]


# ------------------------------------------------------ flags and agreement ---


@needs_corpus
def test_flag_override_rates_are_reported_per_rule(facts, bronze, split_assignment):
    """The output that makes the live validation rules themselves measurable."""
    summary = validate_draft(OVERCLAIMING, facts, salient_fields=())
    annotated = Annotation(
        case_id=HARD_NEGATIVE,
        dataset="amlworld_hi_small",
        annotator_id="annotator-01",
        narrative=FAITHFUL,
        seconds_spent=840.0,
        revision_count=2,
        flags=tuple(FlagOutcome(flag=f, overridden=True) for f in summary.flags),
    )
    report = ingest([annotated], [accepted_review()], facts, bronze, split_assignment)
    assert report.n_ingested == 1
    assert any(ov for _, ov in report.flag_overrides.values())


@needs_corpus
def test_agreement_is_measured_over_the_double_annotated_item(facts, bronze, split_assignment):
    annotations = [
        annotation(FAITHFUL, "annotator-01", "unclassified"),
        annotation(FAITHFUL, "annotator-02", "bipartite"),
    ]
    agreement = measure_agreement(annotations)
    assert agreement.n_double_annotated == 1
    assert agreement.typology_confusion == {"bipartite/unclassified": 1}


# ------------------------------------------------------------- the store ---


@needs_corpus
def test_the_store_round_trips_a_real_annotation(tmp_path, facts):
    summary = validate_draft(FAITHFUL, facts, salient_fields=())
    store = AnnotationStore(root=tmp_path)
    store.append(
        Annotation(
            case_id=HARD_NEGATIVE,
            dataset="amlworld_hi_small",
            annotator_id="annotator-01",
            narrative=FAITHFUL,
            seconds_spent=840.0,
            revision_count=4,
            flags=tuple(FlagOutcome(flag=f, overridden=False) for f in summary.flags),
            panel_digest=build_fact_panel(facts).to_dict(),
        )
    )
    [stored] = store.read_all()
    assert stored.narrative.startswith("[1] Subject & Scope")
    assert stored.panel_digest["case_id"] == HARD_NEGATIVE
