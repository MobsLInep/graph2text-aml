"""Hand-constructed case graphs with known-correct answers.

Every fixture here is built so its expected facts can be worked out by hand from the
picture in its docstring. That is the point: a sub-extractor tested against another
sub-extractor's output proves only that they agree, and two modules can agree while both
being wrong. Invariant 8 applies — every identifier is synthetic.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import polars as pl

from g2t_aml.data.canonical import (
    AMLWORLD_AVAILABILITY,
    ELLIPTIC2_AVAILABILITY,
    AvailabilityMask,
    CanonicalGraph,
)
from g2t_aml.facts.caseview import build_case_view

BASE = datetime(2022, 9, 3, 12, 0)
USD = "US Dollar"
EUR = "Euro"


def at(hours: float) -> datetime:
    """Return a timestamp offset from the fixture epoch."""
    return BASE + timedelta(hours=hours)


def acct(n: int, bank: str = "001") -> str:
    """Return a synthetic account id in the `bank|account` form AMLworld uses (D-011)."""
    return f"{bank}|8000{n:04X}"


def as_laundering_stream(case: CanonicalGraph, typology: str) -> CanonicalGraph:
    """Mark every transaction as belonging to a named laundering stream.

    A real AMLworld positive carries the typology on its *transactions*, not merely on the
    case: `attach_typologies` writes it per row, and the fact layer reads it from there so a
    record can never claim a typology whose evidence is not in the subgraph (D-036). A
    fixture that set only the case-level attribute would not resemble a real positive, and
    the tests written against it would not be testing the real path.
    """
    case.edges = case.edges.with_columns(
        pl.lit(True).alias("is_laundering"),
        pl.lit(typology).alias("typology"),
        pl.lit(f"{typology}_fixture").alias("pattern_id"),
    )
    case.typology = typology
    case.label = "suspicious"
    return case


def make_case(
    edges: list[dict[str, Any]],
    *,
    case_id: str = "fixture-0001",
    dataset: str = "amlworld_hi_small",
    availability: AvailabilityMask = AMLWORLD_AVAILABILITY,
    seed_node: str | None = None,
    typology: str | None = None,
    label: str | None = None,
    banks: dict[str, str] | None = None,
    extra_nodes: list[str] | None = None,
) -> CanonicalGraph:
    """Build a case graph from a list of edge dicts.

    Each edge dict needs `src` and `dst`; `timestamp`, `amount_paid`, `payment_currency`,
    `amount_received`, `receiving_currency`, `payment_format`, `is_laundering` and
    `typology` all default sensibly so a structural fixture stays readable.
    """
    rows = []
    for e in edges:
        paid = e.get("amount_paid", e.get("amount", 1000.0))
        received = e.get("amount_received", paid)
        pay_cur = e.get("payment_currency", e.get("currency", USD))
        rows.append(
            {
                "src": e["src"],
                "dst": e["dst"],
                "timestamp": e.get("timestamp", BASE),
                "amount_received": received,
                "receiving_currency": e.get("receiving_currency", pay_cur),
                "amount_paid": paid,
                "payment_currency": pay_cur,
                "payment_format": e.get("payment_format", "Wire"),
                "is_laundering": e.get("is_laundering", False),
                "typology": e.get("typology"),
                "pattern_id": e.get("pattern_id"),
            }
        )

    edge_frame = pl.DataFrame(
        rows,
        schema={
            "src": pl.Utf8,
            "dst": pl.Utf8,
            "timestamp": pl.Datetime("us"),
            "amount_received": pl.Float64,
            "receiving_currency": pl.Utf8,
            "amount_paid": pl.Float64,
            "payment_currency": pl.Utf8,
            "payment_format": pl.Utf8,
            "is_laundering": pl.Boolean,
            "typology": pl.Utf8,
            "pattern_id": pl.Utf8,
        },
    )

    node_ids = sorted({r["src"] for r in rows} | {r["dst"] for r in rows} | set(extra_nodes or []))
    node_frame = pl.DataFrame(
        {
            "node_id": node_ids,
            "node_type": ["account"] * len(node_ids),
            "bank": [(banks or {}).get(n, n.split("|")[0]) for n in node_ids],
        },
        schema={"node_id": pl.Utf8, "node_type": pl.Utf8, "bank": pl.Utf8},
    )

    provenance: dict[str, Any] = {"extraction_method": "constructed"}
    if seed_node is not None:
        provenance["seed_node"] = seed_node

    return CanonicalGraph(
        graph_id=case_id,
        dataset=dataset,
        nodes=node_frame,
        edges=edge_frame,
        node_feature_names=[],
        edge_feature_names=["amount_paid", "amount_received"],
        availability=availability,
        label=label,
        typology=typology,
        provenance=provenance,
    )


def view_of(case: CanonicalGraph) -> Any:
    """Build the case view, for tests that drive a sub-extractor directly."""
    return build_case_view(case)


# --------------------------------------------------------------- shapes ---


def fan_out_case(width: int = 5, *, hours_apart: float = 1.0) -> CanonicalGraph:
    """HUB -> S1..Sw. One sender, `width` distinct recipients, one hour apart."""
    hub = acct(0)
    return make_case(
        [
            {"src": hub, "dst": acct(i + 1), "timestamp": at(i * hours_apart), "amount": 1000.0}
            for i in range(width)
        ],
        seed_node=hub,
        case_id="fixture-fan-out",
    )


def fan_in_case(width: int = 5, *, hours_apart: float = 1.0) -> CanonicalGraph:
    """S1..Sw -> HUB. `width` distinct senders, one recipient."""
    hub = acct(0)
    return make_case(
        [
            {"src": acct(i + 1), "dst": hub, "timestamp": at(i * hours_apart), "amount": 1000.0}
            for i in range(width)
        ],
        seed_node=hub,
        case_id="fixture-fan-in",
    )


def chain_case(length: int = 4) -> CanonicalGraph:
    """A0 -> A1 -> ... -> A{length}. `length` edges, no branching."""
    return make_case(
        [
            {"src": acct(i), "dst": acct(i + 1), "timestamp": at(i), "amount": 1000.0}
            for i in range(length)
        ],
        seed_node=acct(0),
        case_id="fixture-chain",
    )


def cycle_case(length: int = 4) -> CanonicalGraph:
    """A0 -> A1 -> ... -> A{length-1} -> A0. A directed cycle of exactly `length` edges."""
    return make_case(
        [
            {
                "src": acct(i),
                "dst": acct((i + 1) % length),
                "timestamp": at(i),
                "amount": 1000.0,
            }
            for i in range(length)
        ],
        seed_node=acct(0),
        case_id="fixture-cycle",
    )


def bipartite_case(left: int = 3, right: int = 3) -> CanonicalGraph:
    """L1..Ln -> R1..Rm, complete across, nothing within a side. Exactly bipartite."""
    return make_case(
        [
            {"src": acct(i), "dst": acct(100 + j), "timestamp": at(i), "amount": 1000.0}
            for i in range(left)
            for j in range(right)
        ],
        seed_node=acct(0),
        case_id="fixture-bipartite",
    )


def gather_scatter_case(gather: int = 4, scatter: int = 3) -> CanonicalGraph:
    """G1..Gn -> HUB -> S1..Sm. Disjoint sides, so both widths are exact."""
    hub = acct(0)
    edges = [
        {"src": acct(i + 1), "dst": hub, "timestamp": at(i), "amount": 1000.0}
        for i in range(gather)
    ]
    edges += [
        {"src": hub, "dst": acct(100 + j), "timestamp": at(gather + j), "amount": 800.0}
        for j in range(scatter)
    ]
    return make_case(edges, seed_node=hub, case_id="fixture-gather-scatter")


def scatter_gather_case(width: int = 4) -> CanonicalGraph:
    """ORIGIN -> M1..Mw -> DEST. `width` disjoint two-hop paths."""
    origin, dest = acct(0), acct(999)
    edges = []
    for i in range(width):
        middle = acct(i + 1)
        edges.append({"src": origin, "dst": middle, "timestamp": at(i), "amount": 1000.0})
        edges.append({"src": middle, "dst": dest, "timestamp": at(width + i), "amount": 950.0})
    return make_case(edges, seed_node=origin, case_id="fixture-scatter-gather")


def stack_case(depth: int = 3, layer_width: int = 2) -> CanonicalGraph:
    """A source feeding `depth` successive layers, each `layer_width` accounts wide."""
    edges: list[dict[str, Any]] = []
    previous = [acct(0)]
    node = 1
    for layer in range(depth):
        current = []
        for _ in range(layer_width):
            current.append(acct(node))
            node += 1
        for src in previous:
            for dst in current:
                edges.append({"src": src, "dst": dst, "timestamp": at(layer), "amount": 1000.0})
        previous = current
    return make_case(edges, seed_node=acct(0), case_id="fixture-stack")


def flat_case() -> CanonicalGraph:
    """Two accounts, one transaction. No motif of any kind fires."""
    return make_case(
        [{"src": acct(1), "dst": acct(2), "timestamp": at(0), "amount": 500.0}],
        seed_node=acct(1),
        case_id="fixture-flat",
    )


def elliptic2_case() -> CanonicalGraph:
    """A provided subgraph with Elliptic2's mask: no amounts, no clock, no labels.

    Deliberately carries populated amount and timestamp COLUMNS. The mask, not the
    column, is what must drive the sentinels — a substrate whose numbers exist but mean
    nothing is exactly the case invariant 4 exists for.
    """
    return make_case(
        [
            # is_laundering is None, not False: Elliptic2 labels whole subgraphs, so a
            # transaction there is UNLABELLED rather than known-licit, and the labels
            # block must reach for a sentinel rather than counting everything as licit.
            {
                "src": acct(1),
                "dst": acct(2),
                "timestamp": at(0),
                "amount": 1234.0,
                "is_laundering": None,
            },
            {
                "src": acct(2),
                "dst": acct(3),
                "timestamp": at(1),
                "amount": 999.0,
                "is_laundering": None,
            },
            {
                "src": acct(1),
                "dst": acct(3),
                "timestamp": at(2),
                "amount": 500.0,
                "is_laundering": None,
            },
        ],
        case_id="fixture-elliptic2",
        dataset="elliptic2",
        availability=ELLIPTIC2_AVAILABILITY,
    )
