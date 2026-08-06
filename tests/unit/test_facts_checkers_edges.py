"""Checker edge cases: malformed claims, guard paths, and the leniency traps.

Every test here asserts the checker refuses to guess. The recurring temptation in a
verifier is to be helpful — coerce a string to a number, read a bool as 1, wave through a
claim it cannot parse — and each of those inflates the published faithfulness number by
exactly the amount it gives away.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from tests.factories import acct, at, elliptic2_case, fan_out_case, make_case

from g2t_aml.facts.checkers import (
    CheckContext,
    Claim,
    ClaimType,
    DurationClaim,
    Verdict,
    check_claim,
    register,
)
from g2t_aml.facts.extractor import extract_facts
from g2t_aml.facts.schema import Money


@pytest.fixture
def ctx():
    return CheckContext(facts=extract_facts(fan_out_case(width=5)))


def claim(path, value, kind=ClaimType.NUMERIC):
    return Claim(text_span=(0, 1), field_path=path, claim_type=kind, value=value, raw_text="x")


# --------------------------------------------------------- malformed claims ---


def test_non_numeric_count_claim_is_unverifiable(ctx):
    result = check_claim(claim("structure.n_nodes", "six"), ctx)
    assert result.verdict is Verdict.UNVERIFIABLE
    assert "not a number" in result.reason


def test_boolean_is_not_silently_read_as_a_number(ctx):
    # True is an int in Python. Reading it as 1 would let a categorical claim be checked
    # as a numeric one and quietly agree with a count of 1.
    result = check_claim(claim("structure.n_nodes", True), ctx)
    assert result.verdict is Verdict.UNVERIFIABLE


def test_non_numeric_ratio_claim_is_unverifiable(ctx):
    assert check_claim(claim("structure.density", "high"), ctx).verdict is Verdict.UNVERIFIABLE


def test_non_boolean_claim_on_a_boolean_field_is_unverifiable(ctx):
    result = check_claim(claim("motifs.fan_out.present", "yes", ClaimType.CATEGORICAL), ctx)
    assert result.verdict is Verdict.UNVERIFIABLE


def test_non_sequence_claim_on_an_ordering_field_is_unverifiable(ctx):
    result = check_claim(claim("temporal.event_ordering", "outflow", ClaimType.TEMPORAL), ctx)
    assert result.verdict is Verdict.UNVERIFIABLE


def test_non_amount_claim_on_a_monetary_field_is_unverifiable():
    focal = acct(0)
    ctx = CheckContext(
        facts=extract_facts(
            make_case(
                [{"src": acct(1), "dst": focal, "amount": 500.0, "timestamp": at(0)}],
                seed_node=focal,
            )
        )
    )
    result = check_claim(claim("flow.total_inflow", "a lot"), ctx)
    assert result.verdict is Verdict.UNVERIFIABLE


def test_money_claim_against_a_non_money_field_is_unverifiable(ctx):
    result = check_claim(claim("flow.threshold_reference", Money(10_000.0, "US Dollar")), ctx)
    # threshold_reference is a bare number, not a Money object.
    assert result.verdict is Verdict.UNVERIFIABLE


def test_non_timestamp_claim_on_a_timestamp_field_is_unverifiable(ctx):
    result = check_claim(claim("temporal.window_start", "yesterday", ClaimType.TEMPORAL), ctx)
    assert result.verdict is Verdict.UNVERIFIABLE


def test_non_duration_claim_on_a_duration_field_is_unverifiable(ctx):
    result = check_claim(claim("temporal.span_hours", "ages", ClaimType.TEMPORAL), ctx)
    assert result.verdict is Verdict.UNVERIFIABLE


def test_bare_number_on_a_duration_field_is_read_as_hours(ctx):
    actual = ctx.facts.temporal.span_hours
    assert (
        check_claim(claim("temporal.span_hours", actual, ClaimType.TEMPORAL), ctx).verdict
        is Verdict.SUPPORTED
    )


def test_claim_with_no_field_and_no_special_type_is_unverifiable(ctx):
    result = check_claim(claim(None, 5, ClaimType.NUMERIC), ctx)
    assert result.verdict is Verdict.UNVERIFIABLE
    assert "names no fact field" in result.reason


def test_unknown_duration_unit_raises():
    with pytest.raises(ValueError, match="unknown duration unit"):
        DurationClaim(3, "fortnights").to_hours()


def test_unknown_duration_unit_raises_in_tolerance():
    from g2t_aml.facts.config import ToleranceConfig

    with pytest.raises(ValueError, match="unknown duration unit"):
        DurationClaim(3, "fortnights").tolerance_hours(ToleranceConfig())


# ------------------------------------------------------------- measured null ---


def test_claim_against_a_measured_null_is_unverifiable_not_contradicted(ctx):
    # cycle.length is None because there IS no cycle. That null is a measured value, but
    # there is no number to compare against, so the honest answer is unverifiable.
    assert ctx.facts.motifs.cycle.present is False
    result = check_claim(claim("motifs.cycle.length", 4), ctx)
    assert result.verdict is Verdict.UNVERIFIABLE


def test_always_reported_descriptors_allow_a_real_contradiction(ctx):
    # chain.max_length is reported even when absent, precisely so an overclaim lands as
    # CONTRADICTED rather than being excused as unverifiable.
    assert ctx.facts.motifs.chain.present is False
    result = check_claim(claim("motifs.chain.max_length", 9), ctx)
    assert result.verdict is Verdict.CONTRADICTED
    assert result.hallucination_class == "H2"


# ------------------------------------------------------------- string sets ---


def test_membership_claim_on_a_set_field(ctx):
    assert (
        check_claim(
            claim("flow.currencies_involved", "US Dollar", ClaimType.CATEGORICAL), ctx
        ).verdict
        is Verdict.SUPPORTED
    )
    assert (
        check_claim(claim("flow.currencies_involved", "Yen", ClaimType.CATEGORICAL), ctx).verdict
        is Verdict.CONTRADICTED
    )


def test_categorical_mismatch_is_h5(ctx):
    result = check_claim(claim("focal_entity.role", "beneficiary", ClaimType.CATEGORICAL), ctx)
    assert result.verdict is Verdict.CONTRADICTED
    assert result.hallucination_class == "H5"


def test_ordering_claim_direction_matters():
    # [inflow, outflow] and [outflow, inflow] describe opposite cases.
    focal = acct(0)
    ctx = CheckContext(
        facts=extract_facts(
            make_case(
                [
                    {"src": acct(1), "dst": focal, "timestamp": at(0)},
                    {"src": focal, "dst": acct(2), "timestamp": at(6)},
                ],
                seed_node=focal,
            )
        )
    )
    recorded = list(ctx.facts.temporal.event_ordering)
    assert (
        check_claim(claim("temporal.event_ordering", recorded, ClaimType.TEMPORAL), ctx).verdict
        is Verdict.SUPPORTED
    )
    assert (
        check_claim(
            claim("temporal.event_ordering", list(reversed(recorded)), ClaimType.TEMPORAL), ctx
        ).verdict
        is Verdict.CONTRADICTED
    )


# ------------------------------------------------------------- qualitative ---


def test_descriptor_named_directly_resolves(ctx):
    # Both the descriptor key and its surface phrases must resolve.
    by_key = check_claim(claim(None, "high_fan_out", ClaimType.QUALITATIVE), ctx)
    by_phrase = check_claim(claim(None, "high fan-out", ClaimType.QUALITATIVE), ctx)
    assert by_key.verdict is by_phrase.verdict


def test_descriptor_bound_to_a_masked_field_is_unverifiable():
    ctx = CheckContext(facts=extract_facts(elliptic2_case()))
    result = check_claim(claim(None, "near_threshold_structuring", ClaimType.QUALITATIVE), ctx)
    assert result.verdict is Verdict.UNVERIFIABLE


def test_descriptor_bound_to_a_null_descriptor_is_unverifiable(ctx):
    # layered_structure binds to motifs.stack.depth, which is null on a fan-out.
    assert ctx.facts.motifs.stack.present is False
    result = check_claim(claim(None, "layered structure", ClaimType.QUALITATIVE), ctx)
    assert result.verdict is Verdict.UNVERIFIABLE


# ---------------------------------------------------------------- registry ---


def test_duplicate_registration_raises():
    # Two checkers for one field would make the verdict depend on registration order.
    with pytest.raises(ValueError, match="already has a registered checker"):
        register("structure.n_nodes")(lambda c, x: None)  # type: ignore[arg-type,return-value]


def test_producer_is_none_for_a_fieldless_claim(ctx):
    result = check_claim(claim(None, ctx.facts.focal_entity.id, ClaimType.ENTITY), ctx)
    assert result.producer is None


def test_result_without_a_class_is_not_critical(ctx):
    result = check_claim(claim("structure.n_nodes", ctx.facts.structure.n_nodes), ctx)
    assert result.hallucination_class is None
    assert result.is_critical is False


def test_timestamp_claim_against_a_masked_timestamp_is_unverifiable():
    ctx = CheckContext(facts=extract_facts(elliptic2_case()))
    result = check_claim(
        claim("focal_entity.first_seen", datetime(2022, 9, 3), ClaimType.TEMPORAL), ctx
    )
    assert result.verdict is Verdict.UNVERIFIABLE
