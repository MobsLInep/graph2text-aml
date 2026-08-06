"""An independent oracle: the same quantities, recomputed from the raw case tables.

**Why this exists.** The round-trip probe renders its claims *from the fact record*, so if
the extractor computes a span in seconds the probe states seconds and the checker verifies
seconds, and the round trip passes at 100% while every narrative in the corpus is wrong.
That circularity was found by mutation-testing the gate itself: injecting a
seconds-for-hours bug, an off-by-one node count and a degree-counts-transactions bug left
the round trip at 100% SUPPORTED. The round trip is still worth having — it is what
catches an extractor/checker *semantic disagreement*, which is the failure mode that
silently decouples the corpus from the metric — but on its own it is not sufficient.

This module closes the gap. Every function here recomputes a quantity **directly from the
Polars edge and node tables**, deliberately sharing no code with
:mod:`g2t_aml.facts` — not the case view, not the sub-extractors, not the config. The
implementations are the naive, obviously-correct ones: slower, and written to be read
rather than to be fast. If the oracle and the extractor disagree, one of them is wrong,
and neither can hide behind the other.

Invariant 1 says the fact layer is a measurement instrument. An instrument that has only
ever been compared against itself has not been calibrated.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from g2t_aml.data.canonical import CanonicalGraph

_SECONDS_PER_HOUR = 3600.0


def n_nodes(case: CanonicalGraph) -> int:
    """Count distinct accounts, straight off the node table."""
    return len(set(case.nodes["node_id"].to_list()))


def n_edges(case: CanonicalGraph) -> int:
    """Count transaction rows."""
    return case.edges.height


def n_self_loops(case: CanonicalGraph) -> int:
    """Count rows whose two endpoints are the same string."""
    return sum(
        1
        for s, d in zip(case.edges["src"].to_list(), case.edges["dst"].to_list(), strict=True)
        if s == d
    )


def distinct_in_degree(case: CanonicalGraph, node: str) -> int:
    """Count distinct senders into an account, self excluded."""
    return len(
        {
            s
            for s, d in zip(case.edges["src"].to_list(), case.edges["dst"].to_list(), strict=True)
            if d == node and s != node
        }
    )


def distinct_out_degree(case: CanonicalGraph, node: str) -> int:
    """Count distinct recipients from an account, self excluded."""
    return len(
        {
            d
            for s, d in zip(case.edges["src"].to_list(), case.edges["dst"].to_list(), strict=True)
            if s == node and d != node
        }
    )


def n_transactions_into(case: CanonicalGraph, node: str) -> int:
    """Count inbound transaction rows, self-loops excluded."""
    return sum(
        1
        for s, d in zip(case.edges["src"].to_list(), case.edges["dst"].to_list(), strict=True)
        if d == node and s != node
    )


def n_transactions_out_of(case: CanonicalGraph, node: str) -> int:
    """Count outbound transaction rows, self-loops excluded."""
    return sum(
        1
        for s, d in zip(case.edges["src"].to_list(), case.edges["dst"].to_list(), strict=True)
        if s == node and d != node
    )


def span_hours(case: CanonicalGraph) -> float | None:
    """Return the observed extent of the case's timestamps, in HOURS.

    Written with the unit spelled out in the name and the division visible, because the
    seconds-for-hours substitution is exactly the bug this oracle exists to catch.
    """
    stamps = [t for t in case.edges["timestamp"].to_list() if t is not None]
    if not stamps:
        return None
    return (max(stamps) - min(stamps)).total_seconds() / _SECONDS_PER_HOUR


def n_illicit_transactions(case: CanonicalGraph) -> int:
    """Count rows flagged as laundering."""
    if "is_laundering" not in case.edges.columns:
        return 0
    return sum(1 for v in case.edges["is_laundering"].to_list() if v)


def total_inflow(case: CanonicalGraph, node: str) -> tuple[float, str] | None:
    """Sum value credited to an account, but only when one currency is involved."""
    rows = case.edges.filter((pl.col("dst") == node) & (pl.col("src") != node))
    if rows.height == 0:
        return None
    currencies = set(rows["receiving_currency"].to_list())
    if len(currencies) != 1:
        return None
    return float(sum(rows["amount_received"].to_list())), next(iter(currencies))


def total_outflow(case: CanonicalGraph, node: str) -> tuple[float, str] | None:
    """Sum value debited from an account, but only when one currency is involved."""
    rows = case.edges.filter((pl.col("src") == node) & (pl.col("dst") != node))
    if rows.height == 0:
        return None
    currencies = set(rows["payment_currency"].to_list())
    if len(currencies) != 1:
        return None
    return float(sum(rows["amount_paid"].to_list())), next(iter(currencies))


def currencies_involved(case: CanonicalGraph) -> set[str]:
    """Collect every currency naming either side of any transaction."""
    values: set[str] = set()
    for column in ("payment_currency", "receiving_currency"):
        values |= {v for v in case.edges[column].to_list() if v is not None}
    return values


def max_out_degree(case: CanonicalGraph) -> int:
    """Return the widest distinct-recipient count over every account."""
    return max(
        (distinct_out_degree(case, n) for n in set(case.nodes["node_id"].to_list())), default=0
    )


def max_in_degree(case: CanonicalGraph) -> int:
    """Return the widest distinct-sender count over every account."""
    return max(
        (distinct_in_degree(case, n) for n in set(case.nodes["node_id"].to_list())), default=0
    )


def n_components(case: CanonicalGraph) -> int:
    """Count weakly-connected components by naive union-find over the raw edge list."""
    parent: dict[str, str] = {n: n for n in set(case.nodes["node_id"].to_list())}

    def find(x: str) -> str:
        while parent[x] != x:
            x = parent[x]
        return x

    for s, d in zip(case.edges["src"].to_list(), case.edges["dst"].to_list(), strict=True):
        if s == d:
            continue
        rs, rd = find(s), find(d)
        if rs != rd:
            parent[rs] = rd
    return len({find(n) for n in parent})


def all_quantities(case: CanonicalGraph, focal: str) -> dict[str, Any]:
    """Recompute every oracle-covered quantity for one case."""
    return {
        "structure.n_nodes": n_nodes(case),
        "structure.n_edges": n_edges(case),
        "structure.n_self_loops": n_self_loops(case),
        "structure.n_components": n_components(case),
        "structure.max_in_degree": max_in_degree(case),
        "structure.max_out_degree": max_out_degree(case),
        "focal_entity.in_degree": distinct_in_degree(case, focal),
        "focal_entity.out_degree": distinct_out_degree(case, focal),
        "focal_entity.n_transactions_in": n_transactions_into(case, focal),
        "focal_entity.n_transactions_out": n_transactions_out_of(case, focal),
        "temporal.span_hours": span_hours(case),
        "labels.n_illicit_transactions": n_illicit_transactions(case),
        "flow.total_inflow": total_inflow(case, focal),
        "flow.total_outflow": total_outflow(case, focal),
        "flow.currencies_involved": currencies_involved(case),
    }
