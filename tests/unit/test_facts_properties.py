"""Property-based tests: a detected structure must really BE that structure.

Hand-built fixtures prove a detector fires on the shapes we thought of. These prove it
does not fire on shapes we did not — the failure mode that matters, because a detector
that over-reports produces confident narratives about structures that are not there, and
no amount of fixture coverage rules that out.

Every property is an invariant checkable against the edge list itself, so a violation is
a proof of a bug rather than a disagreement between two implementations.
"""

from __future__ import annotations

from itertools import pairwise

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from tests.factories import acct, at, make_case, view_of

from g2t_aml.facts.config import FactConfig
from g2t_aml.facts.motifs import (
    detect_bipartite,
    detect_chain,
    detect_cycle,
    detect_fan_in,
    detect_fan_out,
    detect_gather_scatter,
    detect_scatter_gather,
    detect_stack,
)
from g2t_aml.facts.schema import facts_to_dict, validate_facts
from g2t_aml.facts.structure import density, reciprocity

CONFIG = FactConfig()
SETTINGS = settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)

# Small node pool: dense random graphs over few nodes are far more likely to contain the
# structures under test than sparse graphs over many.
_NODES = 7


@st.composite
def random_case(draw):
    """Draw a random directed multigraph over a small account pool."""
    n_edges = draw(st.integers(min_value=1, max_value=14))
    pairs = draw(
        st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=_NODES - 1),
                st.integers(min_value=0, max_value=_NODES - 1),
            ),
            min_size=n_edges,
            max_size=n_edges,
        )
    )
    edges = [
        {"src": acct(a), "dst": acct(b), "timestamp": at(i), "amount": 100.0 * (i + 1)}
        for i, (a, b) in enumerate(pairs)
    ]
    return make_case(edges, seed_node=acct(pairs[0][0]))


# ------------------------------------------------------- witness soundness ---


@given(random_case())
@SETTINGS
def test_a_detected_cycle_really_is_a_cycle(case):
    view = view_of(case)
    motif = detect_cycle(view, CONFIG)
    if not motif.present:
        return
    nodes = motif.witness
    assert len(nodes) == motif.descriptors["length"]
    assert len(set(nodes)) == len(nodes), "a reported cycle must not repeat an account"
    assert len(nodes) >= CONFIG.cycle_min_length
    for a, b in pairwise(nodes):
        assert b in view.successors[a], f"{a} -> {b} is not an edge"
    assert nodes[0] in view.successors[nodes[-1]], "the reported cycle does not close"


@given(random_case())
@SETTINGS
def test_a_detected_chain_really_is_a_simple_directed_path(case):
    view = view_of(case)
    motif = detect_chain(view, CONFIG)
    path = motif.witness
    if not path:
        return
    assert len(set(path)) == len(path), "a simple path must not repeat an account"
    for a, b in pairwise(path):
        assert b in view.successors[a]
    assert motif.descriptors["max_length"] == len(path) - 1


@given(random_case())
@SETTINGS
def test_a_detected_fan_really_has_that_many_distinct_counterparties(case):
    view = view_of(case)
    for detector, adjacency in (
        (detect_fan_out, view.successors),
        (detect_fan_in, view.predecessors),
    ):
        motif = detector(view, CONFIG)
        if not motif.present:
            continue
        hub = motif.descriptors["hub"]
        assert len(adjacency[hub]) == motif.descriptors["width"]
        assert hub not in adjacency[hub], "a self-loop must not count as a spoke"


@given(random_case())
@SETTINGS
def test_a_detected_gather_scatter_has_disjoint_sides_of_the_reported_widths(case):
    view = view_of(case)
    motif = detect_gather_scatter(view, CONFIG)
    if not motif.present:
        return
    hub = motif.descriptors["hub"]
    incoming = view.predecessors.get(hub, frozenset()) - {hub}
    outgoing = view.successors.get(hub, frozenset()) - {hub}
    assert len(incoming - outgoing) == motif.descriptors["gather_width"]
    assert len(outgoing - incoming) == motif.descriptors["scatter_width"]


@given(random_case())
@SETTINGS
def test_a_detected_scatter_gather_has_that_many_real_two_hop_paths(case):
    view = view_of(case)
    motif = detect_scatter_gather(view, CONFIG)
    if not motif.present:
        return
    origin, *middles, destination = motif.witness
    assert len(middles) == motif.descriptors["width"]
    assert len(set(middles)) == len(middles)
    for middle in middles:
        assert middle in view.successors[origin]
        assert destination in view.successors[middle]


@given(random_case())
@SETTINGS
def test_a_detected_bipartite_case_has_no_edge_inside_a_side(case):
    view = view_of(case)
    motif = detect_bipartite(view, CONFIG)
    if not motif.present:
        return
    left = set(motif.witness[: motif.descriptors["left_size"]])
    right = set(motif.witness[motif.descriptors["left_size"] :])
    assert not (left & right)
    for edge in view.non_loop_edges():
        assert (edge.src in left) != (edge.dst in left), "an edge stays inside one side"


@given(random_case())
@SETTINGS
def test_a_detected_stack_reports_as_many_widths_as_its_depth(case):
    view = view_of(case)
    motif = detect_stack(view, CONFIG)
    if not motif.present:
        return
    widths = motif.descriptors["layer_widths"]
    assert len(widths) == motif.descriptors["depth"]
    assert all(w >= CONFIG.stack_min_layer_width for w in widths)


# --------------------------------------------------------- global invariants ---


@given(random_case())
@SETTINGS
def test_bounded_quantities_stay_in_range(case):
    view = view_of(case)
    assert 0.0 <= density(view) <= 1.0
    assert 0.0 <= reciprocity(view) <= 1.0
    assert 0.0 <= detect_bipartite(view, CONFIG).descriptors["score"] <= 1.0


@given(random_case())
@SETTINGS
def test_a_present_motif_always_meets_its_configured_threshold(case):
    view = view_of(case)
    fan_out = detect_fan_out(view, CONFIG)
    if fan_out.present:
        assert fan_out.descriptors["width"] >= CONFIG.fan_min_width
    chain = detect_chain(view, CONFIG)
    if chain.present:
        assert chain.descriptors["max_length"] >= CONFIG.chain_min_length
    stack = detect_stack(view, CONFIG)
    if stack.present:
        assert stack.descriptors["depth"] >= CONFIG.stack_min_depth
    gs = detect_gather_scatter(view, CONFIG)
    if gs.present:
        assert min(gs.descriptors["gather_width"], gs.descriptors["scatter_width"]) >= (
            CONFIG.two_sided_min_width
        )


@given(random_case())
@SETTINGS
def test_an_absent_motif_nulls_every_descriptor_except_the_always_reported_ones(case):
    view = view_of(case)
    # chain.max_length and bipartite.score are reported regardless, by design: an
    # overclaim against them is CONTRADICTED rather than merely UNVERIFIABLE.
    for detector in (
        detect_fan_in,
        detect_fan_out,
        detect_cycle,
        detect_stack,
        detect_gather_scatter,
        detect_scatter_gather,
    ):
        motif = detector(view, CONFIG)
        if not motif.present:
            assert all(v is None for v in motif.descriptors.values()), detector.__name__


@given(random_case())
@SETTINGS
def test_any_random_case_produces_a_schema_valid_record(case):
    from g2t_aml.facts.extractor import extract_facts

    validate_facts(facts_to_dict(extract_facts(case)))


@given(random_case())
@SETTINGS
def test_extraction_is_deterministic_on_any_case(case):
    from g2t_aml.facts.extractor import extract_facts

    first = facts_to_dict(extract_facts(case))
    second = facts_to_dict(extract_facts(case))
    first["provenance"].pop("computed_at")
    second["provenance"].pop("computed_at")
    assert first == second
