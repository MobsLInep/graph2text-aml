"""The fact panel: what an annotator sees, and — crucially — what they do not."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.factories import (
    as_laundering_stream,
    cycle_case,
    elliptic2_case,
    fan_out_case,
    flat_case,
)

from g2t_aml.corpus.bronze.templates import ROLE_DISPLAY
from g2t_aml.corpus.claims import _ROLE_INVERSE
from g2t_aml.facts.extractor import extract_facts
from g2t_aml.facts.vocab import load_vocabulary
from g2t_aml.human.factpanel import build_fact_panel

MONETARY_SECTIONS = {"Value"}
CURRENCY_WORDS = ("Dollar", "Euro", "Peso", "Riyal", "currency", "USD", "EUR")


@pytest.fixture
def amlworld_panel():
    return build_fact_panel(extract_facts(as_laundering_stream(fan_out_case(width=6), "fan_out")))


@pytest.fixture
def elliptic2_panel():
    return build_fact_panel(extract_facts(elliptic2_case()))


# ------------------------------------------------------------- invariant 4 ---


def test_elliptic2_panel_has_no_value_section(elliptic2_panel):
    assert elliptic2_panel.section("Value") is None


def test_elliptic2_panel_has_no_timing_section(elliptic2_panel):
    assert elliptic2_panel.section("Timing") is None


def test_elliptic2_panel_has_no_counterparty_label_section(elliptic2_panel):
    assert elliptic2_panel.section("Counterparty labels") is None


def test_elliptic2_panel_shows_no_monetary_or_currency_value_anywhere(elliptic2_panel):
    """The acceptance criterion, asserted over every rendered value in the panel."""
    for row in elliptic2_panel.all_rows():
        for word in CURRENCY_WORDS:
            assert word not in row.value, (row.label, row.value)
        assert "flow." not in row.field_path
        assert "temporal." not in row.field_path
        assert "labels." not in row.field_path


def test_elliptic2_panel_states_the_masked_families_once(elliptic2_panel):
    """Told once, as a property of the substrate; never as forty individual absences."""
    assert len(elliptic2_panel.masked_families) == 3
    assert any("no monetary amounts" in m for m in elliptic2_panel.masked_families)


def test_amlworld_panel_does_have_the_masked_sections(amlworld_panel):
    assert amlworld_panel.section("Value") is not None
    assert amlworld_panel.section("Timing") is not None
    assert amlworld_panel.masked_families == ()


def test_no_row_ever_renders_none_or_the_word_unavailable(amlworld_panel, elliptic2_panel):
    for panel in (amlworld_panel, elliptic2_panel):
        for row in panel.all_rows():
            assert row.value not in ("None", "none", "")
            assert "unavailable" not in row.value.lower()


# ---------------------------------------------------------------- salience ---


def test_salient_rows_are_marked(amlworld_panel):
    marked = {row.field_path for row in amlworld_panel.all_rows() if row.salient}
    assert marked, "no row was marked salient"
    assert marked <= set(amlworld_panel.required_fields)


def test_every_required_field_the_panel_can_show_is_marked_on_a_row(amlworld_panel):
    shown = {row.field_path for row in amlworld_panel.all_rows()}
    missing = [
        p
        for p in amlworld_panel.required_fields
        if p in shown and not any(r.salient for r in amlworld_panel.all_rows() if r.field_path == p)
    ]
    assert not missing, missing


def test_excused_fields_are_reported_separately(elliptic2_panel):
    assert "flow.total_outflow" in elliptic2_panel.excused_fields
    assert "flow.total_outflow" not in elliptic2_panel.required_fields


# -------------------------------------------------- measured null vs masked ---


def test_an_absent_motif_is_shown_as_a_measured_absence_not_omitted(amlworld_panel):
    patterns = amlworld_panel.section("Structural patterns")
    assert patterns is not None
    absent = [r for r in patterns.rows if r.value == "not detected"]
    assert absent, "no 'not detected' row: an absent detector is evidence and must be shown"
    assert all(r.measured_null for r in absent)


def test_a_detected_motif_carries_its_descriptors():
    panel = build_fact_panel(extract_facts(as_laundering_stream(cycle_case(length=4), "cycle")))
    patterns = panel.section("Structural patterns")
    cycle_row = next(r for r in patterns.rows if r.label == "cycle")
    assert cycle_row.value.startswith("detected")
    assert not cycle_row.measured_null


# --------------------------------------------- alignment-compatible spelling ---


def test_role_is_rendered_in_the_spelling_the_claim_parser_inverts(amlworld_panel):
    """A role an annotator copies must parse back, or its salience can never be met."""
    row = next(r for r in amlworld_panel.all_rows() if r.field_path == "focal_entity.role")
    assert row.value in _ROLE_INVERSE, row.value


@pytest.mark.parametrize("role", sorted(ROLE_DISPLAY))
def test_every_role_display_form_round_trips_through_the_claim_parser(role):
    assert _ROLE_INVERSE[ROLE_DISPLAY[role][0]] == role


def test_monetary_values_are_rendered_in_bronze_s_own_format(amlworld_panel):
    """Panel and alignment must agree, or a correctly-copied amount aligns to nothing."""
    from g2t_aml.corpus.bronze.format import parse_money

    value = amlworld_panel.section("Value")
    assert value is not None
    money_rows = [r for r in value.rows if r.field_path.startswith("flow.total_")]
    assert money_rows
    for row in money_rows:
        parse_money(row.value)  # raises FormatError if the panel invented a format


def test_the_threshold_is_shown_in_its_whitelisted_words(amlworld_panel):
    """Citing it any other way is H6; the panel must not teach a non-whitelisted phrase."""
    value = amlworld_panel.section("Value")
    row = next(r for r in value.rows if r.field_path == "flow.n_transfers_near_threshold")
    permitted = {
        phrase
        for reference in load_vocabulary().regulatory.values()
        for phrase in reference.phrase_variants
    }
    assert any(phrase in row.label for phrase in permitted), row.label


# ------------------------------------------------------------------- shape ---


def test_a_two_account_case_still_produces_a_readable_panel():
    panel = build_fact_panel(extract_facts(flat_case()))
    assert panel.sections
    assert "CASE" in panel.rendered_text()


def test_panel_serialises_to_json(amlworld_panel):
    import json

    json.dumps(amlworld_panel.to_dict())


def test_empty_sections_are_dropped_rather_than_rendered_as_a_heading_over_nothing(
    elliptic2_panel,
):
    assert all(section.rows for section in elliptic2_panel.sections)
