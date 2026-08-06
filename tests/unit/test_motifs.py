"""Motif scoring: the machinery hard-negative mining depends on.

Two properties matter. Each motif must score high on the shape it is named for, so mining
selects the population it claims to. And scoring must never see a label, because a score
that could would select hard negatives *by* the label and the result would mean nothing.
"""

from __future__ import annotations

import polars as pl
import pytest

from g2t_aml.data.canonical import AMLWORLD_AVAILABILITY, CanonicalGraph
from g2t_aml.data.motifs import (
    FAN_SATURATION,
    SCORED_MOTIFS,
    score_edges,
    score_motifs,
)


def _edges(pairs: list[tuple[str, str]]) -> pl.DataFrame:
    return pl.DataFrame({"src": [p[0] for p in pairs], "dst": [p[1] for p in pairs]})


def _case(pairs: list[tuple[str, str]], **overrides) -> CanonicalGraph:
    edges = _edges(pairs)
    nodes = sorted({n for pair in pairs for n in pair})
    kwargs = {
        "graph_id": "case",
        "dataset": "fixture",
        "nodes": pl.DataFrame({"node_id": nodes, "node_type": ["account"] * len(nodes)}),
        "edges": edges,
        "node_feature_names": [],
        "edge_feature_names": [],
        "availability": AMLWORLD_AVAILABILITY,
    }
    return CanonicalGraph(**(kwargs | overrides))


def _fan_out(width: int) -> list[tuple[str, str]]:
    return [("HUB", f"L{i:03d}") for i in range(width)]


def _fan_in(width: int) -> list[tuple[str, str]]:
    return [(f"L{i:03d}", "HUB") for i in range(width)]


# ------------------------------------------------------------------ shapes ---


def test_a_wide_fan_out_saturates_its_own_motif():
    scores = score_edges(_edges(_fan_out(FAN_SATURATION)))
    assert scores.best == "fan_out"
    assert scores.best_score == 1.0


def test_a_wide_fan_in_saturates_its_own_motif():
    scores = score_edges(_edges(_fan_in(FAN_SATURATION)))
    assert scores.scores["fan_in"] == 1.0


def test_a_two_counterparty_payment_is_not_a_fan():
    """Two counterparties is a payment, not a pattern."""
    assert score_edges(_edges(_fan_out(2))).scores["fan_out"] == 0.0


def test_fan_score_rises_with_width():
    widths = [3, 6, 10, FAN_SATURATION]
    scores = [score_edges(_edges(_fan_out(w))).scores["fan_out"] for w in widths]
    assert scores == sorted(scores)
    assert scores[0] < scores[-1]


def test_gather_scatter_scores_a_collect_then_disperse_hub():
    pairs = [(f"S{i}", "HUB") for i in range(6)] + [("HUB", f"D{i}") for i in range(6)]
    assert score_edges(_edges(pairs)).scores["gather_scatter"] > 0.5


def test_money_returned_to_its_sender_is_not_a_gather_scatter():
    """The same counterparty on both sides is a round trip, not a collect-and-disperse."""
    pairs = [(f"P{i}", "HUB") for i in range(6)] + [("HUB", f"P{i}") for i in range(6)]
    assert score_edges(_edges(pairs)).scores["gather_scatter"] == 0.0


def test_scatter_gather_scores_a_split_then_recombine():
    pairs = [("SRC", f"M{i}") for i in range(6)] + [(f"M{i}", "DST") for i in range(6)]
    assert score_edges(_edges(pairs)).scores["scatter_gather"] > 0.5


def test_a_three_account_round_trip_saturates_the_cycle_motif():
    assert score_edges(_edges([("A", "B"), ("B", "C"), ("C", "A")])).scores["cycle"] == 1.0


def test_a_longer_closure_scores_below_a_three_cycle():
    long_cycle = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "A")]
    scores = score_edges(_edges(long_cycle))
    assert 0.0 < scores.scores["cycle"] < 1.0


def test_an_acyclic_graph_scores_zero_on_cycle():
    assert score_edges(_edges([("A", "B"), ("B", "C"), ("C", "D")])).scores["cycle"] == 0.0


def test_a_clean_two_sided_structure_scores_on_bipartite():
    pairs = [(f"L{i}", f"R{j}") for i in range(4) for j in range(4)]
    assert score_edges(_edges(pairs)).scores["bipartite"] > 0.8


def test_a_one_sided_structure_does_not_score_on_bipartite():
    assert score_edges(_edges([("A", "B")])).scores["bipartite"] == 0.0


def test_layered_forwarding_scores_on_stack():
    pairs = [
        ("A", "B0"),
        ("A", "B1"),
        ("B0", "C0"),
        ("B1", "C1"),
        ("C0", "D0"),
        ("C1", "D1"),
        ("D0", "E0"),
        ("D1", "E1"),
    ]
    assert score_edges(_edges(pairs)).scores["stack"] == 1.0


def test_a_single_chain_is_not_a_stack():
    """One account per layer is a chain, and a chain is not a stack."""
    pairs = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]
    assert score_edges(_edges(pairs)).scores["stack"] == 0.0


# ------------------------------------------------------------- properties ---


def test_every_scored_motif_is_present_and_bounded():
    scores = score_edges(_edges(_fan_out(8)))
    assert set(scores.scores) == set(SCORED_MOTIFS)
    assert all(0.0 <= v <= 1.0 for v in scores.scores.values())


def test_random_is_never_scored():
    """`random` is the typology defined by having no structure."""
    assert "random" not in SCORED_MOTIFS
    assert "random" not in score_edges(_edges(_fan_out(5))).scores


def test_a_structureless_case_has_no_best_motif():
    scores = score_edges(_edges([("A", "B")]))
    assert scores.best is None
    assert scores.best_score == 0.0


def test_self_loops_do_not_inflate_a_fan():
    """HI-Small is 11.6% self-loops; they are not evidence of any typology."""
    with_loops = _fan_out(4) + [("HUB", "HUB")] * 20
    assert (
        score_edges(_edges(with_loops)).scores["fan_out"]
        == score_edges(_edges(_fan_out(4))).scores["fan_out"]
    )


def test_scoring_is_deterministic_and_order_independent():
    pairs = [*_fan_out(9), ("L001", "L002"), ("L002", "L003")]
    first = score_edges(_edges(pairs))
    second = score_edges(_edges(list(reversed(pairs))))
    assert first.scores == second.scores
    assert first.best == second.best


def test_scoring_ignores_the_label_entirely():
    """The contract that makes mined hard negatives mean anything."""
    pairs = _fan_out(10)
    plain = _edges(pairs)
    labelled = plain.with_columns(
        pl.lit(True).alias("is_laundering"),
        pl.lit("fan_out").alias("typology"),
        pl.lit("fan_out_00001").alias("pattern_id"),
    )
    assert score_edges(plain).scores == score_edges(labelled).scores


def test_the_case_wrapper_agrees_with_the_edge_scorer():
    pairs = _fan_out(7)
    assert score_motifs(_case(pairs)).scores == score_edges(_edges(pairs)).scores


def test_scoring_needs_endpoints():
    with pytest.raises(ValueError, match="src"):
        score_edges(pl.DataFrame({"amount": [1.0]}))
