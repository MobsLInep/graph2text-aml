"""Serialiser: the B7 baseline must be complete, honest and strong.

The tests here are integrity tests as much as correctness tests. A serialiser that quietly
dropped a fact family would weaken the baseline our contribution is measured against, and
that is misconduct rather than a bug.
"""

from __future__ import annotations

import pytest
from tests.factories import (
    EUR,
    USD,
    acct,
    as_laundering_stream,
    at,
    elliptic2_case,
    fan_out_case,
    gather_scatter_case,
    make_case,
)

from g2t_aml.facts.extractor import extract_facts
from g2t_aml.facts.schema import ModelSignal
from g2t_aml.facts.serialiser import COMPACT_DELIMITER, serialise_facts


@pytest.mark.parametrize("style", ["verbose", "compact"])
def test_every_fact_family_reaches_the_text(style):
    # Completeness is a research-integrity requirement: B7 must see everything our own
    # system sees, or the comparison proves nothing.
    facts = extract_facts(gather_scatter_case(gather=4, scatter=3))
    text = serialise_facts(facts, style).lower()
    for marker in ("node", "focal", "inflow", "outflow", "gather", "scatter", "typology"):
        assert marker in text, marker


@pytest.mark.parametrize("style", ["verbose", "compact"])
def test_every_motif_is_named_whether_present_or_not(style):
    facts = extract_facts(fan_out_case(width=5))
    text = serialise_facts(facts, style).lower()
    for motif in (
        ("fan-in", "fan-out", "chain", "cycle", "bipartite", "stack")
        if (style == "verbose")
        else ("fan_in", "fan_out", "chain", "cycle", "bipartite", "stack")
    ):
        assert motif in text, motif


@pytest.mark.parametrize("style", ["verbose", "compact"])
def test_quantities_appear_with_their_descriptors(style):
    facts = extract_facts(fan_out_case(width=7))
    text = serialise_facts(facts, style)
    assert "7" in text  # the fan width is quoted, not merely "a fan-out was detected"


def test_verbose_reports_the_actual_numbers():
    facts = extract_facts(fan_out_case(width=5))
    text = serialise_facts(facts, "verbose")
    assert str(facts.structure.n_nodes) in text
    assert str(facts.structure.n_edges) in text
    assert facts.focal_entity.id in text
    assert facts.focal_entity.role in text


@pytest.mark.parametrize("style", ["verbose", "compact"])
def test_unavailability_is_stated_rather_than_implied_by_silence(style):
    # A baseline told "this substrate has no amounts" is better informed than one left to
    # infer it, and the fair comparison is against the better one.
    facts = extract_facts(elliptic2_case())
    text = serialise_facts(facts, style).lower()
    assert "unavailable" in text
    assert "monetary" in text or "flow" in text


def test_multi_currency_breakdown_survives_the_withheld_aggregate():
    focal = acct(0)
    facts = extract_facts(
        make_case(
            [
                {
                    "src": acct(1),
                    "dst": focal,
                    "amount": 100.0,
                    "currency": USD,
                    "timestamp": at(0),
                },
                {
                    "src": acct(2),
                    "dst": focal,
                    "amount": 250.0,
                    "currency": EUR,
                    "timestamp": at(1),
                },
            ],
            seed_node=focal,
        )
    )
    text = serialise_facts(facts, "verbose")
    assert "unavailable" in text  # the meaningless sum is withheld
    assert "100" in text and "250" in text  # but the real numbers are still there
    assert USD in text and EUR in text


def test_measured_null_and_masked_value_render_differently():
    # "no cycle exists" and "this substrate has no clock" are different statements, and a
    # baseline that could not tell them apart would be at an artificial disadvantage.
    facts = extract_facts(fan_out_case(width=5))
    text = serialise_facts(facts, "verbose")
    assert "none" in text.lower()
    assert "unavailable" in text.lower()


def test_stream_membership_caveat_is_carried_into_the_text():
    # D-019: a case holds 65% of its stream on average. The baseline must know it too.
    facts = extract_facts(as_laundering_stream(fan_out_case(width=5), "fan_out"))
    text = serialise_facts(facts, "verbose")
    assert "part of a stream" in text


def test_model_signal_appears_once_written_back():
    facts = extract_facts(fan_out_case()).with_model_signal(
        ModelSignal(
            gnn_risk_score=0.87,
            score_percentile=99.0,
            model_version="gat-v1",
            top_contributing_nodes=((acct(1), 0.4),),
        )
    )
    text = serialise_facts(facts, "verbose")
    assert "0.87" in text
    assert "gat-v1" in text


def test_model_signal_is_absent_from_the_verbose_text_before_phase_7():
    facts = extract_facts(fan_out_case())
    assert "risk score" not in serialise_facts(facts, "verbose").lower()


def test_serialiser_states_facts_not_conclusions():
    # Handing B7 an interpretation would give it a conclusion our own system must earn.
    facts = extract_facts(gather_scatter_case(gather=5, scatter=5))
    text = serialise_facts(facts, "verbose").lower()
    for verdict in ("suspicious activity report", "is laundering", "criminal", "guilty"):
        assert verdict not in text


def test_compact_is_pipe_delimited_key_value():
    facts = extract_facts(fan_out_case(width=4))
    text = serialise_facts(facts, "compact")
    fields = text.split(COMPACT_DELIMITER)
    assert all("=" in f for f in fields)
    assert any(f.startswith("case_id=") for f in fields)


def test_compact_delimiter_is_unambiguous_against_amlworld_account_ids():
    # AMLworld ids are "<bank>|<account>" (D-011), so a BARE pipe delimiter would split
    # an account in half and any consumer parsing the record back would recover the wrong
    # one. The spaced pipe is what makes the format reversible.
    facts = extract_facts(fan_out_case(width=4))
    text = serialise_facts(facts, "compact")
    assert "|" in facts.focal_entity.id
    fields = dict(f.split("=", 1) for f in text.split(COMPACT_DELIMITER))
    assert fields["focal"] == facts.focal_entity.id


def test_compact_is_denser_than_verbose():
    facts = extract_facts(gather_scatter_case())
    assert len(serialise_facts(facts, "compact")) < len(serialise_facts(facts, "verbose"))


def test_serialisation_is_deterministic():
    facts = extract_facts(gather_scatter_case(gather=4, scatter=4))
    for style in ("verbose", "compact"):
        assert serialise_facts(facts, style) == serialise_facts(facts, style)


def test_unknown_style_raises():
    facts = extract_facts(fan_out_case())
    with pytest.raises(ValueError, match="unknown serialisation style"):
        serialise_facts(facts, "yaml")  # type: ignore[arg-type]
