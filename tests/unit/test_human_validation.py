"""Live validation: what it catches, and that it never blocks."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.factories import as_laundering_stream, elliptic2_case, fan_out_case

from g2t_aml.facts.extractor import extract_facts
from g2t_aml.human.validation import (
    MAX_TOKENS,
    MIN_TOKENS,
    SECTION_HEADINGS,
    Severity,
    validate_draft,
)


@pytest.fixture
def facts():
    return extract_facts(as_laundering_stream(fan_out_case(width=6), "fan_out"))


@pytest.fixture
def elliptic2_facts():
    return extract_facts(elliptic2_case())


def rules(summary):
    return {flag.rule for flag in summary.flags}


PADDING = " ".join(["The subject account dispersed funds to counterparties."] * 12)


# ------------------------------------------------------------ guilt / H7 ---


@pytest.mark.parametrize(
    "phrase", ["is money laundering", "the criminal", "proves", "is guilty of", "conclusively"]
)
def test_guilt_phrases_are_flagged_critical(facts, phrase):
    summary = validate_draft(f"The account {phrase} something. {PADDING}", facts)
    guilt = [f for f in summary.flags if f.rule == "forbidden:guilt"]
    assert guilt, phrase
    assert guilt[0].severity is Severity.CRITICAL
    assert guilt[0].hallucination_class == "H7"
    assert guilt[0].is_critical


def test_a_correctly_hedged_narrative_raises_no_forbidden_flag(facts):
    summary = validate_draft(
        f"The activity appears consistent with layering and warrants further review. {PADDING}",
        facts,
    )
    assert not [f for f in summary.flags if f.rule.startswith("forbidden:")]


# ------------------------------------------------------- entity type / H4 ---


@pytest.mark.parametrize("phrase", ["mixer", "shell company", "darknet market", "hawala"])
def test_entity_type_mentions_are_flagged_critical(facts, phrase):
    summary = validate_draft(f"Funds went to a {phrase}. {PADDING}", facts)
    hits = [f for f in summary.flags if f.rule == "forbidden:entity_type"]
    assert hits, phrase
    assert hits[0].hallucination_class == "H4"
    assert hits[0].is_critical


def test_the_flag_message_says_what_to_write_instead(facts):
    summary = validate_draft(f"Funds went to a mixer. {PADDING}", facts)
    message = next(f.message for f in summary.flags if f.rule == "forbidden:entity_type")
    assert "entity-type column" in message


# ----------------------------------------------------------- completeness ---


def test_completeness_claims_are_flagged(facts):
    summary = validate_draft(f"This describes the entire scheme. {PADDING}", facts)
    assert "forbidden:completeness" in rules(validate_draft(f"the entire scheme {PADDING}", facts))
    assert any(f.hallucination_class == "H8" for f in summary.flags)


def test_motive_claims_are_flagged(facts):
    summary = validate_draft(f"Transfers were deliberately structured. {PADDING}", facts)
    assert "forbidden:motive" in rules(summary)


# ------------------------------------------------------------- entity / H1 ---


def test_an_account_not_in_the_case_is_flagged(facts):
    summary = validate_draft(f"Funds went to 999|DEADBEEF. {PADDING}", facts)
    hits = [f for f in summary.flags if f.rule == "entity:not_in_case"]
    assert hits
    assert hits[0].hallucination_class == "H1"
    assert hits[0].excerpt == "999|DEADBEEF"


def test_an_account_in_the_case_is_not_flagged(facts):
    known = facts.entity_inventory.node_ids[0]
    summary = validate_draft(f"Funds went to {known}. {PADDING}", facts)
    assert not [f for f in summary.flags if f.rule == "entity:not_in_case"]


# --------------------------------------------------- masked families / inv 4 ---


def test_an_amount_on_elliptic2_is_flagged_critical(elliptic2_facts):
    summary = validate_draft(f"The node sent 1,234.00 in value. {PADDING}", elliptic2_facts)
    hits = [f for f in summary.flags if f.rule == "masked:flow"]
    assert hits
    assert hits[0].severity is Severity.CRITICAL


def test_a_clock_time_on_elliptic2_is_flagged(elliptic2_facts):
    summary = validate_draft(f"Activity began at 14:20. {PADDING}", elliptic2_facts)
    assert "masked:temporal" in rules(summary)


def test_the_same_amount_on_amlworld_is_not_flagged(facts):
    summary = validate_draft(f"The account sent 1,234.00 US Dollar. {PADDING}", facts)
    assert "masked:flow" not in rules(summary)


def test_a_topology_only_elliptic2_narrative_raises_no_masking_flag(elliptic2_facts):
    summary = validate_draft(
        "The subgraph comprises 3 nodes connected by 3 transactions with a density of "
        f"0.500. No structural pattern was detected. {PADDING}",
        elliptic2_facts,
    )
    assert not [f for f in summary.flags if f.rule.startswith("masked:")]


# ------------------------------------------------------------------ length ---


def test_a_short_draft_is_flagged(facts):
    summary = validate_draft("Too short.", facts)
    assert "length:out_of_bounds" in rules(summary)
    assert not summary.length_ok


def test_a_long_draft_is_flagged(facts):
    summary = validate_draft("word " * (MAX_TOKENS * 2), facts)
    assert "length:out_of_bounds" in rules(summary)


def test_a_draft_inside_the_bounds_is_not_flagged(facts):
    summary = validate_draft("word " * ((MIN_TOKENS + MAX_TOKENS) // 4), facts)
    assert "length:out_of_bounds" not in rules(summary)
    assert summary.length_ok


# ---------------------------------------------------------------- structure ---


def test_all_four_sections_are_detected_in_title_case(facts):
    draft = "\n\n".join(
        f"[{i}] {h.title()}\nSome content." for i, h in enumerate(SECTION_HEADINGS, 1)
    )
    summary = validate_draft(draft + PADDING, facts)
    assert summary.sections_found == SECTION_HEADINGS
    assert summary.sections_complete


def test_missing_sections_are_reported_as_info_not_a_failure(facts):
    summary = validate_draft(f"[1] Subject & Scope\nSomething. {PADDING}", facts)
    missing = [f for f in summary.flags if f.rule == "structure:missing_section"]
    assert missing
    assert missing[0].severity is Severity.INFO


def test_sections_out_of_order_do_not_count_as_complete(facts):
    draft = "[2] Activity Observed\nx\n\n[1] Subject & Scope\ny\n\n" + PADDING
    summary = validate_draft(draft, facts)
    assert not summary.sections_complete


# ------------------------------------------------------------ never blocking ---


def test_validation_never_raises_on_a_half_typed_draft(facts):
    for draft in ("", "[", "The account is guilty of |||", "0x", "9,", "\n\n\n"):
        validate_draft(draft, facts)


def test_flags_are_ordered_most_severe_first(facts):
    summary = validate_draft(
        f"[1] Subject & Scope\nThe account is guilty of using a mixer. {PADDING}", facts
    )
    severities = [f.severity for f in summary.flags]
    assert severities == sorted(
        severities, key=lambda s: ["critical", "warning", "info"].index(s.value)
    )


def test_every_occurrence_is_flagged_not_just_the_first_per_group(facts):
    """One at a time would make an annotator fix, resubmit, and meet the next."""
    summary = validate_draft(f"A mixer paid a tumbler. {PADDING}", facts)
    entity_flags = [f for f in summary.flags if f.rule == "forbidden:entity_type"]
    assert len(entity_flags) == 2


def test_summary_serialises_to_json(facts):
    import json

    json.dumps(validate_draft(f"Something. {PADDING}", facts).to_dict())
