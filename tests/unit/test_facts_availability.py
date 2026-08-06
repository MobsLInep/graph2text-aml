"""Invariant 4 end to end: nothing may assert a fact that does not exist for its substrate.

The brief's availability test, written as its own file because this is the invariant a
reviewer will probe hardest. Elliptic2 has no amounts, no currencies, no wall-clock
timestamps and no per-transaction labels. The record must carry explicit sentinels — never
zeros, never bare nulls — and the checker must return UNVERIFIABLE for claims about them,
never CONTRADICTED.

The distinction between the last two is the point. CONTRADICTED says "the data says
otherwise"; UNVERIFIABLE says "the data cannot say". Marking a masked claim CONTRADICTED
would punish a narrative for a fact the substrate cannot carry, and would put a
compliance-dangerous claim in the same bucket as an arithmetic slip.
"""

from __future__ import annotations

import pytest
from tests.factories import elliptic2_case, fan_out_case

from g2t_aml.data.canonical import ELLIPTIC2_AVAILABILITY
from g2t_aml.facts.checkers import CheckContext, Claim, ClaimType, Verdict, check_claim
from g2t_aml.facts.extractor import extract_facts
from g2t_aml.facts.schema import (
    CaseFacts,
    Money,
    Unavailable,
    facts_to_dict,
    is_available,
    validate_facts,
)

#: Every monetary or currency-bearing path a narrative could reach for.
MONETARY_PATHS = (
    "flow.total_inflow",
    "flow.total_outflow",
    "flow.retained",
    "flow.max_single_transfer",
    "flow.n_transfers_near_threshold",
    "flow.currencies_involved",
    "flow.n_distinct_banks",
)

TEMPORAL_PATHS = (
    "temporal.window_start",
    "temporal.window_end",
    "temporal.span_hours",
    "temporal.burst_detected",
    "temporal.burst_window_hours",
    "temporal.event_ordering",
)


@pytest.fixture
def elliptic2_facts() -> CaseFacts:
    return extract_facts(elliptic2_case())


def claim(path, value, kind=ClaimType.NUMERIC):
    return Claim(text_span=(0, 1), field_path=path, claim_type=kind, value=value, raw_text="x")


# ------------------------------------------------------------ the record ---


def test_flow_block_is_a_sentinel_not_a_zeroed_block(elliptic2_facts):
    assert isinstance(elliptic2_facts.flow, Unavailable)
    assert elliptic2_facts.flow.reason == "substrate_has_no_monetary_amounts"


def test_temporal_block_is_a_sentinel(elliptic2_facts):
    assert isinstance(elliptic2_facts.temporal, Unavailable)


def test_labels_block_is_a_sentinel(elliptic2_facts):
    assert isinstance(elliptic2_facts.labels, Unavailable)


def test_no_monetary_field_is_populated_anywhere(elliptic2_facts):
    # The brief's requirement stated literally: assert no monetary or currency field is
    # populated. A sentinel is not a value, so reaching one is the correct outcome.
    payload = facts_to_dict(elliptic2_facts)
    assert payload["flow"] == {"available": False, "reason": "substrate_has_no_monetary_amounts"}

    def walk(node):
        if isinstance(node, dict):
            if set(node) == {"value", "currency"}:
                pytest.fail(f"a Money object survived onto an Elliptic2 record: {node}")
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)


def test_absence_is_never_encoded_as_zero(elliptic2_facts):
    # The failure this whole design guards against: 0.0 reads as "nothing moved", which
    # is an assertion the substrate does not license.
    assert not isinstance(elliptic2_facts.flow, Money)
    assert elliptic2_facts.flow != 0
    assert elliptic2_facts.flow is not None


def test_sentinel_is_falsy_but_distinguishable_from_none(elliptic2_facts):
    assert not elliptic2_facts.flow
    assert elliptic2_facts.flow is not None
    assert isinstance(elliptic2_facts.flow, Unavailable)


def test_structure_survives_because_topology_always_exists(elliptic2_facts):
    # The one family no mask can take away, which is what a masked substrate falls back on.
    assert elliptic2_facts.structure.n_nodes == 3
    assert elliptic2_facts.structure.n_edges == 3
    assert is_available(elliptic2_facts.motifs.chain)


def test_focal_timestamps_are_sentinels(elliptic2_facts):
    assert isinstance(elliptic2_facts.focal_entity.first_seen, Unavailable)
    assert isinstance(elliptic2_facts.focal_entity.last_seen, Unavailable)


def test_motif_window_hours_is_null_without_a_clock():
    facts = extract_facts(elliptic2_case())
    for motif in (facts.motifs.fan_in, facts.motifs.fan_out):
        assert motif.descriptors.get("window_hours") is None


def test_record_carries_the_substrates_mask_verbatim(elliptic2_facts):
    assert elliptic2_facts.availability == ELLIPTIC2_AVAILABILITY
    assert facts_to_dict(elliptic2_facts)["availability"] == ELLIPTIC2_AVAILABILITY.to_dict()


def test_elliptic2_record_validates_against_the_frozen_schema(elliptic2_facts):
    validate_facts(facts_to_dict(elliptic2_facts))


# ------------------------------------------------------------ the checker ---


@pytest.mark.parametrize("path", MONETARY_PATHS)
def test_monetary_claims_are_unverifiable_not_contradicted(path, elliptic2_facts):
    ctx = CheckContext(facts=elliptic2_facts)
    result = check_claim(claim(path, 50_000.0), ctx)
    assert result.verdict is Verdict.UNVERIFIABLE
    assert result.verdict is not Verdict.CONTRADICTED
    assert "unavailable" in result.reason or "not present" in result.reason


@pytest.mark.parametrize("path", TEMPORAL_PATHS)
def test_temporal_claims_are_unverifiable_not_contradicted(path, elliptic2_facts):
    ctx = CheckContext(facts=elliptic2_facts)
    result = check_claim(claim(path, 42.0), ctx)
    assert result.verdict is Verdict.UNVERIFIABLE


def test_currency_claim_is_unverifiable(elliptic2_facts):
    ctx = CheckContext(facts=elliptic2_facts)
    result = check_claim(claim("flow.currencies_involved", "US Dollar", ClaimType.CATEGORICAL), ctx)
    assert result.verdict is Verdict.UNVERIFIABLE


def test_label_claims_are_unverifiable(elliptic2_facts):
    ctx = CheckContext(facts=elliptic2_facts)
    for path in ("labels.n_illicit_counterparties", "labels.min_hops_to_known_illicit"):
        assert check_claim(claim(path, 3), ctx).verdict is Verdict.UNVERIFIABLE


def test_structural_claims_still_resolve_on_a_masked_substrate(elliptic2_facts):
    # A masked substrate is not an unverifiable substrate. Topology stays checkable, and
    # both a correct and an incorrect structural claim get a definite answer.
    ctx = CheckContext(facts=elliptic2_facts)
    assert check_claim(claim("structure.n_nodes", 3), ctx).verdict is Verdict.SUPPORTED
    assert check_claim(claim("structure.n_nodes", 9), ctx).verdict is Verdict.CONTRADICTED


def test_entity_claims_still_resolve_on_a_masked_substrate(elliptic2_facts):
    ctx = CheckContext(facts=elliptic2_facts)
    real = elliptic2_facts.entity_inventory.node_ids[0]
    assert check_claim(claim(None, real, ClaimType.ENTITY), ctx).verdict is Verdict.SUPPORTED
    assert (
        check_claim(claim(None, "999|FAKE", ClaimType.ENTITY), ctx).verdict is Verdict.CONTRADICTED
    )


def test_the_same_claim_is_checkable_on_amlworld_and_not_on_elliptic2():
    # The comparison that makes the mask's effect visible: identical claim, different
    # substrate, different verdict -- and neither verdict is wrong.
    amlworld = CheckContext(facts=extract_facts(fan_out_case(width=5)))
    elliptic2 = CheckContext(facts=extract_facts(elliptic2_case()))
    c = claim("flow.total_outflow", 5000.0)
    assert check_claim(c, amlworld).verdict in {Verdict.SUPPORTED, Verdict.CONTRADICTED}
    assert check_claim(c, elliptic2).verdict is Verdict.UNVERIFIABLE


def test_no_probe_claim_on_elliptic2_touches_a_masked_field():
    # A mask-respecting generator makes no unverifiable claims at all. The probe models
    # one, so its Elliptic2 output must be 100% resolvable.
    from tests.probe import run_probe

    from g2t_aml.facts.vocab import load_vocabulary

    facts = extract_facts(elliptic2_case())
    results = run_probe(facts, load_vocabulary())
    assert results
    assert all(r.verdict is Verdict.SUPPORTED for r in results), [
        (r.claim.field_path, r.verdict.value, r.reason)
        for r in results
        if r.verdict is not Verdict.SUPPORTED
    ]
