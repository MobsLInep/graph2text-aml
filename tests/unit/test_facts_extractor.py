"""Focal selection, role assignment, typology resolution, and record-level invariants."""

from __future__ import annotations

import pytest
from tests.factories import (
    acct,
    as_laundering_stream,
    at,
    elliptic2_case,
    fan_out_case,
    flat_case,
    gather_scatter_case,
    make_case,
    view_of,
)

from g2t_aml.facts.config import FactConfig
from g2t_aml.facts.extractor import (
    ROLE_VOCABULARY,
    FactExtractionError,
    assign_role,
    extract_facts,
    select_focal_entity,
)
from g2t_aml.facts.schema import Unavailable, facts_to_dict, validate_facts

CONFIG = FactConfig()


# --------------------------------------------------------- focal selection ---


def test_focal_entity_is_the_extraction_seed_when_present():
    # A case was cut AROUND an account. Re-centring on the busiest node would silently
    # move every narrative onto whichever counterparty happened to be a hub.
    seed = acct(1)
    case = make_case(
        [
            {"src": seed, "dst": acct(2), "timestamp": at(0)},
            # acct(9) is far busier, but it is not what the case is about.
            *[{"src": acct(9), "dst": acct(20 + i), "timestamp": at(i)} for i in range(6)],
        ],
        seed_node=seed,
    )
    node, rule = select_focal_entity(view_of(case))
    assert node == seed
    assert rule == "extraction_seed"


def test_focal_entity_falls_back_to_max_degree_on_a_provided_subgraph():
    view = view_of(elliptic2_case())
    node, rule = select_focal_entity(view)
    assert rule == "max_degree"
    assert node == max(view.node_ids, key=view.total_degree)


def test_focal_fallback_breaks_ties_lexicographically():
    # A 2-cycle: both accounts have degree 1. The smaller identifier must win, so the
    # choice never depends on row order.
    case = make_case(
        [
            {"src": acct(5), "dst": acct(3), "timestamp": at(0)},
            {"src": acct(3), "dst": acct(5), "timestamp": at(1)},
        ]
    )
    node, rule = select_focal_entity(view_of(case))
    assert rule == "max_degree"
    assert node == min(acct(3), acct(5))


def test_seed_outside_the_case_falls_back_rather_than_raising():
    case = make_case([{"src": acct(1), "dst": acct(2), "timestamp": at(0)}], seed_node=acct(77))
    _, rule = select_focal_entity(view_of(case))
    assert rule == "max_degree"


# ------------------------------------------------------------------ roles ---


@pytest.mark.parametrize(
    ("edges", "expected"),
    [
        # Sends only.
        ([(0, 1)], "originator"),
        # Receives only.
        ([(1, 0)], "beneficiary"),
        # Exactly one each way, with DIFFERENT counterparties.
        ([(1, 0), (0, 2)], "pass_through"),
        # Two in, one out: neither a pass-through nor a hub.
        ([(1, 0), (2, 0), (0, 3)], "intermediary"),
    ],
)
def test_roles_from_in_case_degree(edges, expected):
    focal = acct(0)
    case = make_case(
        [{"src": acct(a), "dst": acct(b), "timestamp": at(i)} for i, (a, b) in enumerate(edges)],
        seed_node=focal,
    )
    assert assign_role(view_of(case), focal, CONFIG) == expected


def test_hub_needs_the_minimum_on_both_sides():
    # 5 in and 5 out at hub_min_degree=5.
    focal = acct(0)
    edges = [{"src": acct(i + 1), "dst": focal, "timestamp": at(i)} for i in range(5)]
    edges += [{"src": focal, "dst": acct(100 + i), "timestamp": at(10 + i)} for i in range(5)]
    assert assign_role(view_of(make_case(edges, seed_node=focal)), focal, CONFIG) == "hub"


def test_wide_on_one_side_only_is_not_a_hub():
    focal = acct(0)
    edges = [{"src": acct(i + 1), "dst": focal, "timestamp": at(i)} for i in range(9)]
    edges += [{"src": focal, "dst": acct(100), "timestamp": at(20)}]
    assert assign_role(view_of(make_case(edges, seed_node=focal)), focal, CONFIG) == "intermediary"


def test_self_loop_only_account_is_terminal():
    focal = acct(0)
    case = make_case(
        [
            {"src": focal, "dst": focal, "timestamp": at(0)},
            {"src": acct(1), "dst": acct(2), "timestamp": at(1)},
        ],
        seed_node=focal,
    )
    assert assign_role(view_of(case), focal, CONFIG) == "terminal"


def test_every_role_is_in_the_controlled_vocabulary():
    for case in (fan_out_case(), gather_scatter_case(), flat_case(), elliptic2_case()):
        facts = extract_facts(case)
        assert facts.focal_entity.role in ROLE_VOCABULARY


# -------------------------------------------------------------- typology ---


def test_ground_truth_typology_is_scoped_to_stream_membership():
    # D-019: a case holds 65% of its stream on average, so the label means "part of a
    # stream of this typology", never "exhibits it in full".
    case = as_laundering_stream(fan_out_case(width=5), "fan_out")
    facts = extract_facts(case)
    assert facts.typology.label == "fan_out"
    assert facts.typology.source == "ground_truth"
    assert facts.typology.confidence == 1.0
    assert facts.typology.scope == "stream_membership"


def test_licit_amlworld_case_is_ground_truth_unclassified_not_an_inferred_typology():
    # THE regression this guards. A licit AMLworld case is shaped like a fan-out -- so is
    # an ordinary payroll run -- but the ground truth positively says it belongs to no
    # laundering stream. Inferring "fan_out" from the shape would attach a laundering
    # typology to 25,571 licit cases, and every Bronze narrative built from them would
    # assert a scheme the data says is not there. See D-035.
    case = fan_out_case(width=6)
    assert case.typology is None
    assert case.availability.typology_ground_truth is True

    facts = extract_facts(case)
    assert facts.typology.label == "unclassified"
    assert facts.typology.source == "ground_truth"
    assert facts.typology.scope == "stream_membership"
    assert facts.typology.confidence == 1.0
    # The structural finding is not lost -- it is in the motifs block, as a shape.
    assert facts.motifs.fan_out.present is True
    assert facts.motifs.fan_out.descriptors["width"] == 6


def test_a_typology_is_never_claimed_without_evidence_in_the_case():
    # THE record-level invariant (D-036): a named typology always has flagged transactions
    # backing it. Phase 2's CaseRecord carries the SEEDING STREAM's typology, and on the
    # built corpus 346 of 30,000 cases carry one while holding no laundering transaction at
    # all -- the 48-hour window caught the seed account but none of its stream's flagged
    # edges. Trusting that label would emit `{"label": "cycle", "source": "ground_truth"}`
    # for a subgraph containing no cycle and no evidence whatsoever.
    stream_labelled_but_empty = fan_out_case(width=5)
    stream_labelled_but_empty.typology = "cycle"  # as CaseCollection.materialise sets it
    stream_labelled_but_empty.label = "suspicious"

    facts = extract_facts(stream_labelled_but_empty)
    assert facts.labels is not None
    assert facts.labels.n_illicit_transactions == 0
    assert (
        facts.typology.label == "unclassified"
    ), "a case with no flagged transaction must not claim a laundering typology"

    # And the converse: real evidence yields the real label.
    real = as_laundering_stream(fan_out_case(width=5), "cycle")
    real_facts = extract_facts(real)
    assert real_facts.labels.n_illicit_transactions > 0
    assert real_facts.typology.label == "cycle"


def test_typology_is_inferred_without_ground_truth():
    facts = extract_facts(elliptic2_case())
    assert facts.typology.source in {"inferred", "none"}
    assert facts.typology.scope == "case_structure"
    assert facts.typology.confidence < 1.0


def test_inferred_typology_prefers_the_more_specific_composite():
    # A gather-scatter is also a fan-in and a fan-out. The composite is more informative.
    case = gather_scatter_case(gather=4, scatter=4)
    case.typology = None
    case.availability = elliptic2_case().availability
    facts = extract_facts(case)
    assert facts.typology.source == "inferred"
    assert facts.typology.label == "gather_scatter"


def test_structureless_case_is_unclassified_with_zero_confidence():
    case = flat_case()
    case.typology = None
    case.availability = elliptic2_case().availability
    facts = extract_facts(case)
    assert facts.typology.label == "unclassified"
    assert facts.typology.source == "none"
    assert facts.typology.confidence == 0.0


# ---------------------------------------------------------------- record ---


def test_record_validates_against_the_frozen_schema():
    for case in (fan_out_case(), gather_scatter_case(), flat_case(), elliptic2_case()):
        validate_facts(facts_to_dict(extract_facts(case)))


def test_entity_inventory_is_complete_and_contains_the_focal_entity():
    # H1 is checked against this list, so an incomplete one makes fabrication invisible.
    case = gather_scatter_case(gather=4, scatter=3)
    facts = extract_facts(case)
    assert set(facts.entity_inventory.node_ids) == set(view_of(case).node_ids)
    assert facts.entity_inventory.focal_id in facts.entity_inventory.node_ids
    assert facts.entity_inventory.focal_id == facts.focal_entity.id


def test_oversized_case_raises_rather_than_truncating_the_inventory():
    case = fan_out_case(width=20)
    with pytest.raises(FactExtractionError, match="max_inventory_nodes"):
        extract_facts(case, FactConfig(max_inventory_nodes=5))


def test_field_producers_cover_every_extracted_family():
    facts = extract_facts(fan_out_case())
    assert facts.provenance is not None
    producers = facts.provenance.field_producers
    for path in (
        "structure.n_nodes",
        "temporal.span_hours",
        "flow.total_inflow",
        "labels.n_illicit_counterparties",
        "motifs.fan_out.width",
        "focal_entity.role",
        "typology.label",
    ):
        assert path in producers, path


def test_config_is_recorded_so_a_detector_verdict_is_reproducible():
    facts = extract_facts(fan_out_case(), FactConfig(fan_min_width=99))
    assert facts.provenance is not None
    assert facts.provenance.config["fan_min_width"] == 99
    assert facts.motifs.fan_out.present is False


def test_extraction_is_deterministic():
    case = gather_scatter_case(gather=5, scatter=4)
    a = facts_to_dict(extract_facts(case))
    b = facts_to_dict(extract_facts(case))
    a["provenance"].pop("computed_at")
    b["provenance"].pop("computed_at")
    assert a == b


def test_focal_degrees_are_in_case_not_global():
    # The node table's precomputed degrees are GLOBAL aggregates. Reading them here
    # would make every narrative in the corpus unfaithful.
    focal = acct(0)
    case = make_case(
        [
            {"src": focal, "dst": acct(1), "timestamp": at(0)},
            {"src": focal, "dst": acct(2), "timestamp": at(1)},
        ],
        seed_node=focal,
    )
    facts = extract_facts(case)
    assert facts.focal_entity.out_degree == 2
    assert facts.focal_entity.in_degree == 0


def test_focal_timestamps_are_sentinels_without_a_clock():
    facts = extract_facts(elliptic2_case())
    assert isinstance(facts.focal_entity.first_seen, Unavailable)
    assert isinstance(facts.focal_entity.last_seen, Unavailable)


def test_model_signal_starts_null_and_accepts_a_write_back():
    from g2t_aml.facts.schema import ModelSignal

    facts = extract_facts(fan_out_case())
    assert facts.model_signal.gnn_risk_score is None
    updated = facts.with_model_signal(
        ModelSignal(gnn_risk_score=0.87, score_percentile=99.1, model_version="gat-v1")
    )
    assert updated.model_signal.gnn_risk_score == 0.87
    # The original is frozen and untouched, so a hash taken earlier stays valid.
    assert facts.model_signal.gnn_risk_score is None
    validate_facts(facts_to_dict(updated))
