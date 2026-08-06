"""Checkers: all three verdicts, and the tolerance boundaries on both sides.

The boundary tests are the ones that matter. A checker that is lenient at the edge
inflates the published faithfulness number by exactly the amount it gives away, and that
kind of leniency is invisible in an aggregate.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from tests.factories import (
    EUR,
    USD,
    acct,
    as_laundering_stream,
    at,
    elliptic2_case,
    fan_out_case,
    make_case,
)

from g2t_aml.facts.checkers import (
    CheckContext,
    Claim,
    ClaimType,
    DurationClaim,
    Verdict,
    check_claim,
    check_narrative_text,
    summarise,
)
from g2t_aml.facts.config import FactConfig
from g2t_aml.facts.extractor import extract_facts
from g2t_aml.facts.schema import Money

CONFIG = FactConfig()


def ctx_for(case):
    return CheckContext(facts=extract_facts(case), config=CONFIG)


def claim(path, value, kind=ClaimType.NUMERIC, text="x"):
    return Claim(
        text_span=(0, len(text)), field_path=path, claim_type=kind, value=value, raw_text=text
    )


# ------------------------------------------------------------------ counts ---


def test_count_exact_match_is_supported():
    ctx = ctx_for(fan_out_case(width=5))
    result = check_claim(claim("motifs.fan_out.width", 5), ctx)
    assert result.verdict is Verdict.SUPPORTED
    assert result.hallucination_class is None


def test_count_off_by_one_is_contradicted():
    # Counts get NO tolerance. "Six accounts" when there are five is simply wrong.
    ctx = ctx_for(fan_out_case(width=5))
    result = check_claim(claim("motifs.fan_out.width", 6), ctx)
    assert result.verdict is Verdict.CONTRADICTED
    assert result.hallucination_class == "H2"


def test_count_claim_carries_the_producing_computation():
    ctx = ctx_for(fan_out_case(width=5))
    result = check_claim(claim("structure.n_nodes", 6), ctx)
    assert result.producer == "structure.node_count"


# ---------------------------------------------------------------- monetary ---


def test_monetary_within_one_percent_is_supported():
    # "Approximately USD 5,000" against 4,975: an investigator wants a magnitude, and
    # marking good writing unfaithful would be the wrong failure.
    focal = acct(0)
    ctx = ctx_for(
        make_case(
            [{"src": acct(1), "dst": focal, "amount": 5000.0, "timestamp": at(0)}],
            seed_node=focal,
        )
    )
    assert check_claim(claim("flow.total_inflow", 5040.0), ctx).verdict is Verdict.SUPPORTED


def test_monetary_just_inside_the_one_percent_boundary():
    focal = acct(0)
    ctx = ctx_for(
        make_case(
            [{"src": acct(1), "dst": focal, "amount": 10_000.0, "timestamp": at(0)}],
            seed_node=focal,
        )
    )
    # 1% of 10,000 is exactly 100.
    assert check_claim(claim("flow.total_inflow", 10_100.0), ctx).verdict is Verdict.SUPPORTED
    assert check_claim(claim("flow.total_inflow", 10_101.0), ctx).verdict is Verdict.CONTRADICTED


def test_monetary_absolute_floor_protects_tiny_amounts():
    # 1% of 0.01 BTC is 0.0001, which no written rounding could hit. The floor is what
    # stops the tolerance collapsing to nothing on Bitcoin rows.
    focal = acct(0)
    ctx = ctx_for(
        make_case(
            [
                {
                    "src": acct(1),
                    "dst": focal,
                    "amount": 0.01,
                    "currency": "Bitcoin",
                    "timestamp": at(0),
                }
            ],
            seed_node=focal,
        )
    )
    assert check_claim(claim("flow.total_inflow", 0.02), ctx).verdict is Verdict.SUPPORTED


def test_wrong_currency_is_contradicted_regardless_of_the_amount():
    focal = acct(0)
    ctx = ctx_for(
        make_case(
            [{"src": acct(1), "dst": focal, "amount": 5000.0, "currency": USD, "timestamp": at(0)}],
            seed_node=focal,
        )
    )
    result = check_claim(claim("flow.total_inflow", Money(5000.0, EUR)), ctx)
    assert result.verdict is Verdict.CONTRADICTED
    assert "currency" in result.reason


def test_multi_currency_aggregate_is_unverifiable_not_contradicted():
    # A narrative cannot contradict a sum that has no defined value.
    focal = acct(0)
    ctx = ctx_for(
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
                    "amount": 100.0,
                    "currency": EUR,
                    "timestamp": at(1),
                },
            ],
            seed_node=focal,
        )
    )
    result = check_claim(claim("flow.total_inflow", 200.0), ctx)
    assert result.verdict is Verdict.UNVERIFIABLE


# --------------------------------------------------------------- durations ---


def test_vague_day_claim_is_supported_within_one_day():
    # "About 3 days" against 76 hours. The narrative claimed a precision of one day and
    # is right at that precision.
    focal = acct(0)
    ctx = ctx_for(
        make_case(
            [
                {"src": focal, "dst": acct(1), "timestamp": at(0)},
                {"src": focal, "dst": acct(2), "timestamp": at(76)},
            ],
            seed_node=focal,
        )
    )
    result = check_claim(
        claim("temporal.span_hours", DurationClaim(3, "days"), ClaimType.TEMPORAL), ctx
    )
    assert result.verdict is Verdict.SUPPORTED


def test_precise_hour_claim_is_contradicted_by_a_four_hour_error():
    # "76 hours" against 80 claims a precision of one hour and misses by four.
    focal = acct(0)
    ctx = ctx_for(
        make_case(
            [
                {"src": focal, "dst": acct(1), "timestamp": at(0)},
                {"src": focal, "dst": acct(2), "timestamp": at(80)},
            ],
            seed_node=focal,
        )
    )
    result = check_claim(
        claim("temporal.span_hours", DurationClaim(76, "hours"), ClaimType.TEMPORAL), ctx
    )
    assert result.verdict is Verdict.CONTRADICTED
    assert result.hallucination_class == "H3"


def test_same_error_is_supported_when_stated_vaguely():
    # The identical 4-hour error, stated as "3 days", is SUPPORTED. This asymmetry is the
    # whole point of reading granularity from the claim rather than imposing one.
    focal = acct(0)
    ctx = ctx_for(
        make_case(
            [
                {"src": focal, "dst": acct(1), "timestamp": at(0)},
                {"src": focal, "dst": acct(2), "timestamp": at(80)},
            ],
            seed_node=focal,
        )
    )
    result = check_claim(
        claim("temporal.span_hours", DurationClaim(3, "days"), ClaimType.TEMPORAL), ctx
    )
    assert result.verdict is Verdict.SUPPORTED


def test_duration_boundary_is_exactly_one_unit():
    focal = acct(0)
    ctx = ctx_for(
        make_case(
            [
                {"src": focal, "dst": acct(1), "timestamp": at(0)},
                {"src": focal, "dst": acct(2), "timestamp": at(10)},
            ],
            seed_node=focal,
        )
    )
    assert (
        check_claim(
            claim("temporal.span_hours", DurationClaim(11, "hours"), ClaimType.TEMPORAL), ctx
        ).verdict
        is Verdict.SUPPORTED
    )
    assert (
        check_claim(
            claim("temporal.span_hours", DurationClaim(11.5, "hours"), ClaimType.TEMPORAL), ctx
        ).verdict
        is Verdict.CONTRADICTED
    )


# -------------------------------------------------------------- timestamps ---


def test_timestamp_within_the_substrates_own_resolution_is_supported():
    ctx = ctx_for(fan_out_case(width=4))
    actual = ctx.facts.temporal.window_start
    result = check_claim(
        claim("temporal.window_start", actual + timedelta(seconds=30), ClaimType.TEMPORAL), ctx
    )
    assert result.verdict is Verdict.SUPPORTED


def test_timestamp_off_by_hours_is_contradicted():
    ctx = ctx_for(fan_out_case(width=4))
    actual = ctx.facts.temporal.window_start
    result = check_claim(
        claim("temporal.window_start", actual + timedelta(hours=3), ClaimType.TEMPORAL), ctx
    )
    assert result.verdict is Verdict.CONTRADICTED
    assert result.hallucination_class == "H3"


# ----------------------------------------------------------------- entity ---


def test_entity_in_the_subgraph_is_supported():
    ctx = ctx_for(fan_out_case(width=4))
    result = check_claim(claim(None, ctx.facts.focal_entity.id, ClaimType.ENTITY), ctx)
    assert result.verdict is Verdict.SUPPORTED


def test_fabricated_entity_is_h1_and_never_unverifiable():
    # The inventory is always complete, so an entity is either present or invented.
    ctx = ctx_for(fan_out_case(width=4))
    result = check_claim(claim(None, "999|DEADBEEF", ClaimType.ENTITY), ctx)
    assert result.verdict is Verdict.CONTRADICTED
    assert result.hallucination_class == "H1"


# ------------------------------------------------------------- regulatory ---


def test_whitelisted_citation_is_supported():
    ctx = ctx_for(fan_out_case())
    result = check_claim(
        claim(None, "the USD 10,000 reporting threshold", ClaimType.REGULATORY), ctx
    )
    assert result.verdict is Verdict.SUPPORTED


def test_invented_rule_is_h6_and_critical():
    ctx = ctx_for(fan_out_case())
    result = check_claim(
        claim(None, "section 12 of the Panama Banking Act", ClaimType.REGULATORY), ctx
    )
    assert result.verdict is Verdict.CONTRADICTED
    assert result.hallucination_class == "H6"
    assert result.is_critical is True


# ------------------------------------------------------------ qualitative ---


def test_descriptor_holding_its_binding_is_supported():
    # 6 transactions inside 2 hours: a burst of span 2, and rapid_dispersal binds <= 6.
    focal = acct(0)
    ctx = ctx_for(
        make_case(
            [{"src": focal, "dst": acct(i + 1), "timestamp": at(i * 0.4)} for i in range(6)],
            seed_node=focal,
        )
    )
    assert ctx.facts.temporal.burst_detected is True
    result = check_claim(claim(None, "rapid dispersal", ClaimType.QUALITATIVE), ctx)
    assert result.verdict is Verdict.SUPPORTED


def test_descriptor_failing_its_binding_is_contradicted():
    # 6 transactions spread over 20 hours. "Rapid dispersal" is then a contradiction,
    # not a stylistic quibble.
    focal = acct(0)
    ctx = ctx_for(
        make_case(
            [{"src": focal, "dst": acct(i + 1), "timestamp": at(i * 4.0)} for i in range(6)],
            seed_node=focal,
        )
    )
    result = check_claim(claim(None, "rapid dispersal", ClaimType.QUALITATIVE), ctx)
    assert result.verdict is Verdict.CONTRADICTED
    assert result.hallucination_class == "H2"


def test_uncontrolled_intensifier_is_unverifiable():
    # "Highly suspicious" resolves to no measurement, so it is neither supported nor
    # contradicted -- it is exactly the compliance-dangerous vagueness the third bucket
    # exists to collect.
    ctx = ctx_for(fan_out_case(width=5))
    result = check_claim(claim(None, "highly suspicious", ClaimType.QUALITATIVE), ctx)
    assert result.verdict is Verdict.UNVERIFIABLE


def test_descriptor_is_unverifiable_when_the_substrate_lacks_the_flag():
    # rapid_dispersal requires absolute_timestamps, which Elliptic2 does not have.
    ctx = ctx_for(elliptic2_case())
    result = check_claim(claim(None, "rapid dispersal", ClaimType.QUALITATIVE), ctx)
    assert result.verdict is Verdict.UNVERIFIABLE


def test_high_fan_out_descriptor_binds_to_the_measured_width():
    ctx = ctx_for(fan_out_case(width=9))
    assert (
        check_claim(claim(None, "high fan-out", ClaimType.QUALITATIVE), ctx).verdict
        is Verdict.SUPPORTED
    )
    ctx_small = ctx_for(fan_out_case(width=4))
    assert (
        check_claim(claim(None, "high fan-out", ClaimType.QUALITATIVE), ctx_small).verdict
        is Verdict.CONTRADICTED
    )


# --------------------------------------------------------------- typology ---


def test_ground_truth_typology_match_is_supported():
    ctx = ctx_for(as_laundering_stream(fan_out_case(width=5), "fan_out"))
    result = check_claim(claim("typology.label", "fan_out", ClaimType.CATEGORICAL), ctx)
    assert result.verdict is Verdict.SUPPORTED


def test_ground_truth_typology_mismatch_is_h5():
    ctx = ctx_for(as_laundering_stream(fan_out_case(width=5), "fan_out"))
    result = check_claim(claim("typology.label", "cycle", ClaimType.CATEGORICAL), ctx)
    assert result.verdict is Verdict.CONTRADICTED
    assert result.hallucination_class == "H5"


def test_inferred_typology_is_unverifiable_even_when_it_matches():
    # An inferred label is this system's own detector talking, not a fact about the case.
    case = fan_out_case(width=6)
    case.typology = None
    case.availability = elliptic2_case().availability
    ctx = ctx_for(case)
    assert ctx.facts.typology.source == "inferred"
    result = check_claim(
        claim("typology.label", ctx.facts.typology.label, ClaimType.CATEGORICAL), ctx
    )
    assert result.verdict is Verdict.UNVERIFIABLE


# ------------------------------------------------------ availability gating ---


def test_masked_field_is_unverifiable_never_contradicted():
    # THE availability test at claim level: Elliptic2 has no amounts, so a monetary claim
    # is unverifiable. Returning CONTRADICTED here would punish a narrative for a fact
    # the substrate cannot carry.
    ctx = ctx_for(elliptic2_case())
    for path in ("flow.total_inflow", "flow.total_outflow", "flow.max_single_transfer"):
        result = check_claim(claim(path, 1000.0), ctx)
        assert result.verdict is Verdict.UNVERIFIABLE, path
        assert result.verdict is not Verdict.CONTRADICTED


def test_cross_border_is_unverifiable_on_every_substrate():
    for case in (fan_out_case(), elliptic2_case()):
        ctx = ctx_for(case)
        result = check_claim(claim("flow.cross_border", True, ClaimType.CATEGORICAL), ctx)
        assert result.verdict is Verdict.UNVERIFIABLE


def test_unregistered_field_is_unverifiable_not_supported():
    # Leniency is a bug. An unrecognised claim has not been verified.
    ctx = ctx_for(fan_out_case())
    result = check_claim(claim("structure.invented_field", 3), ctx)
    assert result.verdict is Verdict.UNVERIFIABLE


# ------------------------------------------------------------------- text ---


def test_guilt_language_is_h7():
    ctx = ctx_for(fan_out_case())
    results = check_narrative_text("The account holder is guilty of money laundering.", ctx)
    assert any(r.hallucination_class == "H7" and r.is_critical for r in results)


def test_entity_type_attribution_is_h4():
    ctx = ctx_for(fan_out_case())
    results = check_narrative_text("Funds were routed through a mixer.", ctx)
    assert any(r.hallucination_class == "H4" and r.is_critical for r in results)


def test_completeness_claim_is_h8():
    ctx = ctx_for(fan_out_case())
    results = check_narrative_text("This describes the entire scheme.", ctx)
    assert any(r.hallucination_class == "H8" for r in results)


def test_motive_claim_is_h8():
    ctx = ctx_for(fan_out_case())
    results = check_narrative_text("The funds were deliberately structured.", ctx)
    assert any(r.hallucination_class == "H8" for r in results)


def test_clean_hedged_narrative_produces_no_text_findings():
    ctx = ctx_for(as_laundering_stream(fan_out_case(width=5), "fan_out"))
    text = (
        "The account dispersed funds to five counterparties, which appears consistent "
        "with a fan-out pattern and warrants further review."
    )
    assert check_narrative_text(text, ctx) == []


def test_unhedged_inferred_typology_is_h5():
    case = fan_out_case(width=6)
    case.typology = None
    case.availability = elliptic2_case().availability
    ctx = ctx_for(case)
    results = check_narrative_text("The subgraph is a fan_out scheme.", ctx)
    assert any(r.hallucination_class == "H5" for r in results)


def test_hedged_inferred_typology_passes():
    case = fan_out_case(width=6)
    case.typology = None
    case.availability = elliptic2_case().availability
    ctx = ctx_for(case)
    results = check_narrative_text("The structure appears consistent with a fan-out pattern.", ctx)
    assert results == []


# -------------------------------------------------------------- summarise ---


def test_summarise_reports_critical_rate_separately():
    ctx = ctx_for(fan_out_case(width=5))
    results = [
        check_claim(claim("motifs.fan_out.width", 5), ctx),  # supported
        check_claim(claim("motifs.fan_out.width", 9), ctx),  # contradicted, H2
        check_claim(claim(None, "made up rule", ClaimType.REGULATORY), ctx),  # H6, critical
        check_claim(claim(None, "highly suspicious", ClaimType.QUALITATIVE), ctx),  # unverif.
    ]
    summary = summarise(results)
    assert summary["n_claims"] == 4
    assert summary["by_verdict"]["supported"] == 1
    assert summary["by_verdict"]["contradicted"] == 2
    assert summary["by_verdict"]["unverifiable"] == 1
    assert summary["n_critical"] == 1
    assert summary["critical_error_rate"] == pytest.approx(0.25)
    # Faithfulness alone would hide the critical error entirely.
    assert summary["supported_rate"] == pytest.approx(0.25)


def test_summarise_of_nothing_does_not_raise():
    assert summarise([])["critical_error_rate"] == 0.0
