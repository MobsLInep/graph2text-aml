"""The case subgraph, laid out so a person can actually read it.

**A bad visualisation produces bad narratives**, and it does so invisibly: an annotator who
cannot see that eleven of the twelve recipients are fresh accounts writes a narrative that
does not say so, and the omission looks like a judgement call rather than a rendering
failure. This module is therefore held to the same standard as the fact panel — what is on
screen must be true, complete for what it claims to show, and honest about what it leaves
out.

Four decisions worth stating.

**Layout is deterministic.** ``spring_layout`` is seeded from the case id, so the same case
looks the same to both annotators of a double-annotated item and the same again at
adjudication. A layout that moved between sessions would make two people describing one
graph describe two different pictures, and the disagreement would be charged to them.

**Size encodes degree, colour encodes label, width encodes value — and each is dropped
when the substrate cannot support it.** On Elliptic2 there are no amounts, so every edge
is drawn at one width; there are no per-account labels, so every node is drawn unknown.
Encoding a mask as a visual default — thin edges reading as "small amounts", grey nodes
reading as "licit" — is invariant 4 violated in pixels rather than in text, and it is
harder to catch because nobody writes it down.

**Large cases are capped, and the cap is stated on screen.** Above
:data:`DEFAULT_MAX_NODES` the view keeps the focal entity, its neighbourhood, and then the
highest-degree remainder, and reports how many accounts and transactions are hidden. A
silently truncated graph is the worst outcome available: the annotator writes a confident
narrative about a case they have seen two thirds of.

**The timeline is the edges' own timestamps**, bucketed, so the scrubber moves through real
activity rather than through a linear interpolation of the window. Without absolute
timestamps there is no timeline and the control is absent rather than disabled.

The module builds a plain data structure first and a figure from it second, so every rule
above is testable without a browser or the ``human`` extra installed.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from g2t_aml.facts.caseview import CaseView

__all__ = [
    "DEFAULT_MAX_NODES",
    "NODE_COLOURS",
    "GraphEdge",
    "GraphNode",
    "GraphView",
    "build_graph_view",
    "to_plotly_figure",
]

#: Above this many accounts the view is capped and the remainder reported as hidden. 60
#: is about where a force-directed layout at screen size stops being readable; the
#: extraction cap is 150, so the largest cases really do hit this.
DEFAULT_MAX_NODES = 60

#: Label to colour. Chosen for a colour-vision-deficient reader: the flagged/unflagged
#: distinction is orange against blue rather than red against green.
NODE_COLOURS: dict[str, str] = {
    "flagged": "#d95f02",
    "unflagged": "#1b6ca8",
    "unknown": "#8a8a8a",
    "focal": "#111111",
}


@dataclass(frozen=True)
class GraphNode:
    """One account, positioned and encoded.

    Attributes:
        node_id: The account identifier.
        x: Layout x, in [-1, 1].
        y: Layout y, in [-1, 1].
        degree: Distinct counterparties, in and out, self excluded. Drives the marker
            size — never the transaction count, which is a different claim (D-008 note in
            CLAUDE.md §9.8).
        in_degree: Distinct in-neighbours.
        out_degree: Distinct out-neighbours.
        label: ``flagged``, ``unflagged`` or ``unknown``. Always ``unknown`` when the
            substrate carries no per-account labels.
        is_focal: Whether this is the account the case is about.
        bank: Institution code, or None when the substrate has no institution identity.
    """

    node_id: str
    x: float
    y: float
    degree: int
    in_degree: int
    out_degree: int
    label: str
    is_focal: bool
    bank: str | None = None

    @property
    def colour(self) -> str:
        """Return the marker colour.

        Returns:
            The focal colour when focal, otherwise the label's colour.
        """
        return NODE_COLOURS["focal"] if self.is_focal else NODE_COLOURS[self.label]

    @property
    def marker_size(self) -> float:
        """Return the marker size, scaled by degree.

        Sub-linear in degree on purpose: a degree-40 hub next to a degree-1 spoke would
        otherwise be 40 times the area and swamp everything around it.

        Returns:
            A size in plotly marker units.
        """
        return 10.0 + 6.0 * math.sqrt(self.degree)


@dataclass(frozen=True)
class GraphEdge:
    """One transaction, positioned and encoded.

    Attributes:
        src: Sending account.
        dst: Receiving account.
        timestamp: When it happened, or None without absolute timestamps.
        amount: Value debited from ``src``, or None without monetary amounts.
        currency: The currency, or None.
        payment_format: The rail, or None.
        is_laundering: Ground-truth flag, or None when unlabelled. **None is not False**:
            an unlabelled transaction is drawn as unknown, never as clean.
        width: Line width. A constant when the substrate has no amounts.
    """

    src: str
    dst: str
    timestamp: datetime | None
    amount: float | None
    currency: str | None
    payment_format: str | None
    is_laundering: bool | None
    width: float

    @property
    def is_self_loop(self) -> bool:
        """Report whether the transaction has one account at both ends.

        Returns:
            True when source and destination are the same.
        """
        return self.src == self.dst


@dataclass(frozen=True)
class GraphView:
    """A case laid out for display, with what was left out stated.

    Attributes:
        case_id: The case.
        nodes: The displayed accounts.
        edges: The displayed transactions.
        focal_id: The account the case is about.
        n_hidden_nodes: Accounts omitted by the cap.
        n_hidden_edges: Transactions omitted with them.
        timeline: Bucket boundaries for the scrubber, as datetimes. Empty without
            absolute timestamps.
        has_amounts: Whether edge width encodes anything.
        has_labels: Whether node colour encodes anything.
        max_nodes: The cap that was applied.
    """

    case_id: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    focal_id: str
    n_hidden_nodes: int = 0
    n_hidden_edges: int = 0
    timeline: tuple[datetime, ...] = ()
    has_amounts: bool = True
    has_labels: bool = True
    max_nodes: int = DEFAULT_MAX_NODES

    @property
    def truncated(self) -> bool:
        """Report whether anything was hidden.

        Returns:
            True when at least one account or transaction is not on screen.
        """
        return bool(self.n_hidden_nodes or self.n_hidden_edges)

    @property
    def caption(self) -> str:
        """Return the line shown under the plot.

        Returns:
            A statement of what is displayed and, when capped, what is not. The
            truncation notice is part of the view rather than left to the caller,
            because a caller that forgets it produces exactly the failure the cap exists
            to avoid.
        """
        base = f"{len(self.nodes)} accounts, {len(self.edges)} transactions"
        if not self.truncated:
            return base
        return (
            f"{base} — SHOWING THE {self.max_nodes} HIGHEST-DEGREE ACCOUNTS. "
            f"{self.n_hidden_nodes} accounts and {self.n_hidden_edges} transactions are "
            "not displayed; do not describe this case as complete."
        )

    def edges_before(self, moment: datetime | None) -> tuple[GraphEdge, ...]:
        """Return the transactions at or before a moment, for the timeline scrubber.

        Args:
            moment: The scrubber position, or None for everything.

        Returns:
            The edges. Edges without a timestamp are always included: hiding them would
            make the scrubber assert they happened outside the window, which the record
            does not say.
        """
        if moment is None:
            return self.edges
        return tuple(e for e in self.edges if e.timestamp is None or e.timestamp <= moment)

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised view.

        Returns:
            A JSON-serialisable summary — counts and encodings, not coordinates. Stored
            on the annotation record so a later reader knows what the annotator saw.
        """
        return {
            "case_id": self.case_id,
            "n_nodes_displayed": len(self.nodes),
            "n_edges_displayed": len(self.edges),
            "n_hidden_nodes": self.n_hidden_nodes,
            "n_hidden_edges": self.n_hidden_edges,
            "truncated": self.truncated,
            "has_amounts": self.has_amounts,
            "has_labels": self.has_labels,
            "max_nodes": self.max_nodes,
            "label_counts": dict(sorted(Counter(n.label for n in self.nodes).items())),
        }


def _layout(
    node_ids: list[str], edges: list[tuple[str, str]], case_id: str
) -> dict[str, tuple[float, float]]:
    """Compute a deterministic force-directed layout.

    Args:
        node_ids: Every account to place.
        edges: Undirected endpoint pairs, self-loops already removed.
        case_id: Seeds the layout, so the same case always looks the same.

    Returns:
        Account to ``(x, y)``, each coordinate in [-1, 1].
    """
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(node_ids)
    graph.add_edges_from(edges)
    seed = int.from_bytes(case_id.encode()[-4:] or b"seed", "big") % (2**31)
    positions = nx.spring_layout(graph, seed=seed, k=None, iterations=64)
    return {str(n): (float(p[0]), float(p[1])) for n, p in positions.items()}


def _keep(view: CaseView, focal_id: str, max_nodes: int) -> tuple[list[str], int]:
    """Choose which accounts to display when the case is larger than the cap.

    The focal entity first, then its immediate neighbourhood, then the highest-degree
    remainder. That ordering keeps the account the report is about and the counterparties
    that make it suspicious, which is what the narrative has to describe.

    Args:
        view: The case view.
        focal_id: The focal account.
        max_nodes: The cap.

    Returns:
        ``(kept ids sorted, number hidden)``.
    """
    all_ids = list(view.node_ids)
    if len(all_ids) <= max_nodes:
        return sorted(all_ids), 0

    degree = {
        n: len(view.successors.get(n, frozenset())) + len(view.predecessors.get(n, frozenset()))
        for n in all_ids
    }
    neighbours = set(view.successors.get(focal_id, frozenset())) | set(
        view.predecessors.get(focal_id, frozenset())
    )
    ranked = sorted(
        all_ids,
        key=lambda n: (
            0 if n == focal_id else 1 if n in neighbours else 2,
            -degree[n],
            n,
        ),
    )
    kept = ranked[:max_nodes]
    return sorted(kept), len(all_ids) - len(kept)


def build_graph_view(
    view: CaseView,
    focal_id: str,
    *,
    max_nodes: int = DEFAULT_MAX_NODES,
    timeline_buckets: int = 24,
) -> GraphView:
    """Lay a case out for display.

    Args:
        view: The case, as the fact layer sees it. Taking a
            :class:`~g2t_aml.facts.caseview.CaseView` rather than raw frames is
            deliberate: the view has already decided what a degree is and how self-loops
            are treated, and the picture must agree with the record on both.
        focal_id: The account the case is about.
        max_nodes: The display cap.
        timeline_buckets: How many steps the scrubber offers.

    Returns:
        The view, capped and encoded.

    Raises:
        ValueError: If ``focal_id`` is not an account in the case, which would centre the
            display on an entity the record does not contain.
    """
    if focal_id not in set(view.node_ids):
        raise ValueError(
            f"focal account {focal_id!r} is not in case {view.case_id!r}; the display "
            "would be centred on an entity the record does not contain"
        )

    kept_ids, n_hidden_nodes = _keep(view, focal_id, max_nodes)
    kept = set(kept_ids)
    has_amounts = bool(view.availability.monetary_amounts)
    # `view.has_labels`, NOT `availability.node_labels`. The mask flag is True on
    # Elliptic2 — it labels whole subgraphs — while no individual account is labelled at
    # all. Colouring from the mask would paint every Elliptic2 node "unflagged", which is
    # a per-account assertion the substrate cannot license: invariant 4 violated in
    # pixels, and harder to catch than the same claim in text because nobody writes it
    # down. The fact layer gates `LabelFacts` on exactly this property, and the picture
    # must agree with the record.
    has_labels = bool(view.has_labels)

    shown_edges = [e for e in view.edges if e.src in kept and e.dst in kept]
    n_hidden_edges = len(view.edges) - len(shown_edges)

    amounts = [e.amount_paid for e in shown_edges if has_amounts and e.amount_paid is not None]
    largest = max(amounts) if amounts else None

    edges: list[GraphEdge] = []
    for edge in shown_edges:
        if largest and edge.amount_paid is not None:
            width = 1.0 + 5.0 * math.sqrt(max(edge.amount_paid, 0.0) / largest)
        else:
            width = 1.5
        edges.append(
            GraphEdge(
                src=edge.src,
                dst=edge.dst,
                timestamp=edge.timestamp,
                amount=edge.amount_paid if has_amounts else None,
                currency=edge.payment_currency if has_amounts else None,
                payment_format=edge.payment_format,
                is_laundering=edge.is_laundering if has_labels else None,
                width=width,
            )
        )

    flagged: set[str] = set()
    if has_labels:
        for edge in view.edges:
            if edge.is_laundering:
                flagged.update((edge.src, edge.dst))

    positions = _layout(
        kept_ids,
        [(e.src, e.dst) for e in edges if not e.is_self_loop],
        view.case_id,
    )
    nodes = tuple(
        GraphNode(
            node_id=node_id,
            x=positions[node_id][0],
            y=positions[node_id][1],
            degree=len(view.successors.get(node_id, frozenset()))
            + len(view.predecessors.get(node_id, frozenset())),
            in_degree=len(view.predecessors.get(node_id, frozenset())),
            out_degree=len(view.successors.get(node_id, frozenset())),
            label=("flagged" if node_id in flagged else "unflagged") if has_labels else "unknown",
            is_focal=node_id == focal_id,
            bank=(view.banks or {}).get(node_id),
        )
        for node_id in kept_ids
    )

    stamps = sorted(e.timestamp for e in edges if e.timestamp is not None)
    timeline: tuple[datetime, ...] = ()
    if stamps and view.availability.absolute_timestamps:
        first, last = stamps[0], stamps[-1]
        span = (last - first).total_seconds()
        steps = max(1, min(timeline_buckets, len(stamps)))
        timeline = tuple(
            first + (last - first) * (i / steps) if span else first for i in range(steps + 1)
        )

    return GraphView(
        case_id=view.case_id,
        nodes=nodes,
        edges=tuple(edges),
        focal_id=focal_id,
        n_hidden_nodes=n_hidden_nodes,
        n_hidden_edges=n_hidden_edges,
        timeline=timeline,
        has_amounts=has_amounts,
        has_labels=has_labels,
        max_nodes=max_nodes,
    )


def _require_plotly() -> Any:
    """Import plotly, or explain how to install it.

    Returns:
        The ``plotly.graph_objects`` module.

    Raises:
        ImportError: If plotly is not installed, with the command that installs it.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "the annotation interface needs plotly. Install the human extra: "
            "`uv sync --group dev --extra human`"
        ) from exc
    return go


def to_plotly_figure(view: GraphView, *, until: datetime | None = None) -> Any:
    """Build the interactive figure for the annotation interface.

    Args:
        view: The laid-out case.
        until: Scrubber position. Only transactions at or before it are drawn; accounts
            are always all drawn, because an account that has not yet transacted still
            exists and hiding it would make the graph appear to grow.

    Returns:
        A ``plotly.graph_objects.Figure``.

    Raises:
        ImportError: If plotly is not installed.
    """
    go = _require_plotly()
    positions = {n.node_id: (n.x, n.y) for n in view.nodes}

    traces: list[Any] = []
    for edge in view.edges_before(until):
        if edge.is_self_loop:
            continue
        x0, y0 = positions[edge.src]
        x1, y1 = positions[edge.dst]
        colour = "#d95f02" if edge.is_laundering else "#b6c4d2"
        traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line={"width": edge.width, "color": colour},
                hoverinfo="text",
                text=_edge_hover(edge),
                showlegend=False,
            )
        )

    traces.append(
        go.Scatter(
            x=[n.x for n in view.nodes],
            y=[n.y for n in view.nodes],
            mode="markers+text",
            marker={
                "size": [n.marker_size for n in view.nodes],
                "color": [n.colour for n in view.nodes],
                "line": {"width": 2, "color": "#ffffff"},
            },
            text=["focal" if n.is_focal else "" for n in view.nodes],
            textposition="bottom center",
            hoverinfo="text",
            hovertext=[_node_hover(n) for n in view.nodes],
            showlegend=False,
        )
    )

    figure = go.Figure(data=traces)
    figure.update_layout(
        title=view.caption,
        showlegend=False,
        hovermode="closest",
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        xaxis={"visible": False},
        yaxis={"visible": False, "scaleanchor": "x"},
        plot_bgcolor="#ffffff",
    )
    return figure


def _node_hover(node: GraphNode) -> str:
    """Return the hover text for an account.

    Args:
        node: The account.

    Returns:
        Identifier, degrees, label and institution, omitting what is unavailable.
    """
    parts = [
        f"<b>{node.node_id}</b>",
        f"in {node.in_degree} / out {node.out_degree} counterparties",
    ]
    if node.label != "unknown":
        parts.append(f"label: {node.label}")
    if node.bank:
        parts.append(f"institution: {node.bank}")
    if node.is_focal:
        parts.append("<i>focal account</i>")
    return "<br>".join(parts)


def _edge_hover(edge: GraphEdge) -> str:
    """Return the hover text for a transaction.

    Args:
        edge: The transaction.

    Returns:
        Endpoints plus whatever the substrate supports. A masked family contributes
        nothing rather than a placeholder.
    """
    parts = [f"{edge.src} → {edge.dst}"]
    if edge.amount is not None:
        parts.append(f"{edge.amount:,.2f} {edge.currency or ''}".strip())
    if edge.timestamp is not None:
        parts.append(edge.timestamp.strftime("%Y-%m-%d %H:%M"))
    if edge.payment_format:
        parts.append(edge.payment_format)
    if edge.is_laundering:
        parts.append("<b>flagged</b>")
    return "<br>".join(parts)
