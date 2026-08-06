"""A plain-Python view of a case, built once and shared by every sub-extractor.

Each sub-extractor could read the Polars frames itself, and each would then have its own
opinion about self-loops, null amounts and whether a "degree" counts transactions or
counterparties. Those opinions would drift, and the drift would be invisible: the
extractor and the checker would disagree about semantics while both passing their own
tests. :class:`CaseView` exists so that decision is made once, here, and every sub-module
inherits it.

Three conventions are fixed in this file and hold everywhere downstream.

**Degree means distinct counterparties, not transactions.** "Received from twelve
accounts" is the claim a narrative makes; "twelve transactions from four accounts" is a
different claim, and the record carries both under different names
(:attr:`CaseView.in_neighbours` versus :attr:`CaseView.n_transactions_into`).

**Self-loops are excluded from structure and included in counts.** HI-Small is 11.6%
self-loops and D-017 keeps them, so ``structure.n_edges`` and ``structure.n_self_loops``
report them honestly, while adjacency, degree and every motif detector ignore them — an
account transacting with itself is not a counterparty relationship and would otherwise
manufacture a one-node "cycle" in a tenth of all cases.

**A missing column means the fact is unavailable, never zero.** The view reports
``None`` for an absent column and the sub-extractor turns that into an
:class:`~g2t_aml.facts.schema.Unavailable` sentinel. Nothing in this file substitutes a
default for a quantity the substrate does not carry.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import polars as pl

from g2t_aml.data.canonical import AvailabilityMask, CanonicalGraph

__all__ = ["CaseEdge", "CaseView", "build_case_view"]


@dataclass(frozen=True)
class CaseEdge:
    """One transaction, with everything the fact layer reads off it.

    Attributes:
        src: Sending account.
        dst: Receiving account.
        timestamp: When it happened, or None without absolute timestamps.
        amount_paid: Value debited from ``src``, in :attr:`payment_currency`.
        payment_currency: Currency ``src`` paid in.
        amount_received: Value credited to ``dst``, in :attr:`receiving_currency`. Not
            the same number as ``amount_paid`` on a cross-currency transfer — HI-Small
            has 72,170 of them.
        receiving_currency: Currency ``dst`` received in.
        payment_format: Rail, e.g. ``"Wire"``. None when the substrate has no such
            column.
        is_laundering: Ground-truth flag. **None means unlabelled**, which is different
            from False and is what makes ``n_unknown_counterparties`` meaningful.
        typology: Stream typology carried on the transaction, or None.
    """

    src: str
    dst: str
    timestamp: datetime | None
    amount_paid: float | None
    payment_currency: str | None
    amount_received: float | None
    receiving_currency: str | None
    payment_format: str | None
    is_laundering: bool | None
    typology: str | None

    @property
    def is_self_loop(self) -> bool:
        """Report whether both endpoints are the same account.

        Returns:
            True when ``src == dst``.
        """
        return self.src == self.dst


@dataclass(frozen=True)
class CaseView:
    """A case reduced to the structures the fact layer computes over.

    Attributes:
        case_id: The case identifier.
        dataset: Substrate key.
        availability: What may be asserted about this case.
        node_ids: Every account, sorted. Determinism depends on this ordering.
        edges: Every transaction in canonical order, self-loops included.
        successors: Distinct out-neighbours per node, self-loops excluded.
        predecessors: Distinct in-neighbours per node, self-loops excluded.
        banks: Account to bank code, or None when the substrate has no institution
            identity.
        label: The case-level label, or None.
        typology: The case-level typology from Phase 2, or None.
        provenance: The Phase 2 extraction provenance, verbatim.
    """

    case_id: str
    dataset: str
    availability: AvailabilityMask
    node_ids: tuple[str, ...]
    edges: tuple[CaseEdge, ...]
    successors: dict[str, frozenset[str]]
    predecessors: dict[str, frozenset[str]]
    banks: dict[str, str] | None
    label: str | None
    typology: str | None
    provenance: dict[str, Any]

    # ------------------------------------------------------------- counting ---

    @property
    def n_nodes(self) -> int:
        """Return the account count.

        Returns:
            Number of distinct accounts.
        """
        return len(self.node_ids)

    @property
    def n_edges(self) -> int:
        """Return the transaction count, self-loops included.

        Returns:
            Number of transactions.
        """
        return len(self.edges)

    @property
    def n_self_loops(self) -> int:
        """Return the self-loop transaction count.

        Returns:
            Number of transactions whose endpoints coincide.
        """
        return sum(1 for e in self.edges if e.is_self_loop)

    def non_loop_edges(self) -> tuple[CaseEdge, ...]:
        """Return every transaction between two distinct accounts.

        Returns:
            The edges, in canonical order.
        """
        return tuple(e for e in self.edges if not e.is_self_loop)

    def out_degree(self, node: str) -> int:
        """Return distinct out-neighbours of a node.

        Args:
            node: Account identifier.

        Returns:
            Count of distinct accounts this one sent to, self excluded.
        """
        return len(self.successors.get(node, frozenset()))

    def in_degree(self, node: str) -> int:
        """Return distinct in-neighbours of a node.

        Args:
            node: Account identifier.

        Returns:
            Count of distinct accounts this one received from, self excluded.
        """
        return len(self.predecessors.get(node, frozenset()))

    def total_degree(self, node: str) -> int:
        """Return the count of distinct counterparties in either direction.

        A node adjacent to the same account both ways is counted once, which is what a
        narrative means by "four counterparties".

        Args:
            node: Account identifier.

        Returns:
            Size of the union of in- and out-neighbours.
        """
        return len(self.neighbours(node))

    def neighbours(self, node: str) -> frozenset[str]:
        """Return every distinct counterparty of a node, in either direction.

        Args:
            node: Account identifier.

        Returns:
            The union of in- and out-neighbours, self excluded.
        """
        return self.successors.get(node, frozenset()) | self.predecessors.get(node, frozenset())

    def edges_into(self, node: str) -> tuple[CaseEdge, ...]:
        """Return transactions received by a node from another account.

        Args:
            node: Account identifier.

        Returns:
            Inbound transactions, self-loops excluded, in canonical order.
        """
        return tuple(e for e in self.edges if e.dst == node and not e.is_self_loop)

    def edges_out_of(self, node: str) -> tuple[CaseEdge, ...]:
        """Return transactions sent by a node to another account.

        Args:
            node: Account identifier.

        Returns:
            Outbound transactions, self-loops excluded, in canonical order.
        """
        return tuple(e for e in self.edges if e.src == node and not e.is_self_loop)

    def n_transactions_into(self, node: str) -> int:
        """Return the inbound transaction count.

        Args:
            node: Account identifier.

        Returns:
            Number of inbound transactions, self-loops excluded.
        """
        return len(self.edges_into(node))

    def n_transactions_out_of(self, node: str) -> int:
        """Return the outbound transaction count.

        Args:
            node: Account identifier.

        Returns:
            Number of outbound transactions, self-loops excluded.
        """
        return len(self.edges_out_of(node))

    # ------------------------------------------------------------ traversal ---

    def undirected_neighbours(self) -> dict[str, frozenset[str]]:
        """Return the undirected adjacency, self-loops excluded.

        Returns:
            Account to the set of accounts it transacts with in either direction.
        """
        return {node: self.neighbours(node) for node in self.node_ids}

    def hops_from(self, origin: str) -> dict[str, int]:
        """Return undirected hop distance from one account to every reachable account.

        Args:
            origin: The account to measure from.

        Returns:
            Account to hop count, including ``origin`` at 0. Unreachable accounts are
            absent from the mapping rather than present with a sentinel distance.
        """
        adjacency = self.undirected_neighbours()
        distance: dict[str, int] = {origin: 0}
        queue: deque[str] = deque([origin])
        while queue:
            node = queue.popleft()
            for neighbour in sorted(adjacency.get(node, frozenset())):
                if neighbour not in distance:
                    distance[neighbour] = distance[node] + 1
                    queue.append(neighbour)
        return distance

    def weakly_connected_components(self) -> list[tuple[str, ...]]:
        """Partition the accounts into weakly-connected components.

        Isolated accounts — reachable only through their own self-loops — form
        singleton components, which is why the case count can exceed one even when every
        transaction is accounted for.

        Returns:
            Components as sorted tuples, ordered by their smallest member.
        """
        adjacency = self.undirected_neighbours()
        seen: set[str] = set()
        components: list[tuple[str, ...]] = []
        for start in self.node_ids:
            if start in seen:
                continue
            block: list[str] = []
            queue: deque[str] = deque([start])
            seen.add(start)
            while queue:
                node = queue.popleft()
                block.append(node)
                for neighbour in sorted(adjacency.get(node, frozenset())):
                    if neighbour not in seen:
                        seen.add(neighbour)
                        queue.append(neighbour)
            components.append(tuple(sorted(block)))
        return sorted(components, key=lambda block: block[0])

    # ---------------------------------------------------------- availability ---

    @property
    def has_timestamps(self) -> bool:
        """Report whether the case carries usable transaction timestamps.

        Both conditions matter: the mask governs what may be *asserted* and the column
        governs what can be *computed*. A case is only temporally describable when both
        hold.

        Returns:
            True when the mask permits absolute timestamps and every transaction has one.
        """
        return self.availability.absolute_timestamps and all(
            e.timestamp is not None for e in self.edges
        )

    @property
    def has_amounts(self) -> bool:
        """Report whether the case carries usable monetary amounts.

        Returns:
            True when the mask permits amounts and every transaction has a paid and a
            received value.
        """
        return self.availability.monetary_amounts and all(
            e.amount_paid is not None and e.amount_received is not None for e in self.edges
        )

    @property
    def has_labels(self) -> bool:
        """Report whether the case carries per-transaction illicit labels.

        Elliptic2 labels whole subgraphs, not transactions, so this is False there even
        though ``availability.node_labels`` is True: a subgraph-level label licenses no
        statement about an individual counterparty.

        Returns:
            True when at least one transaction carries a non-null laundering flag.
        """
        return any(e.is_laundering is not None for e in self.edges)


def _column(frame: pl.DataFrame, name: str) -> list[Any]:
    """Return a column as a Python list, or a list of Nones when it is absent.

    Args:
        frame: The table to read.
        name: Column name.

    Returns:
        The column's values, or ``[None] * height`` when the column does not exist.
    """
    if name not in frame.columns:
        return [None] * frame.height
    values: list[Any] = frame[name].to_list()
    return values


def build_case_view(case: CanonicalGraph) -> CaseView:
    """Reduce a case graph to the view every sub-extractor computes over.

    Args:
        case: A case as materialised by Phase 2, or any :class:`CanonicalGraph`.

    Returns:
        The view, with adjacency built and self-loops separated.

    Raises:
        ValueError: If an edge endpoint is absent from the node table. The fact layer is
            a measurement instrument; silently tolerating a dangling endpoint would let
            a degree be computed against a node set that does not contain it.
    """
    nodes = case.nodes
    edges = case.edges

    node_ids = tuple(sorted(str(n) for n in nodes["node_id"].to_list()))
    known = set(node_ids)

    srcs = [str(v) for v in edges["src"].to_list()]
    dsts = [str(v) for v in edges["dst"].to_list()]
    if dangling := (set(srcs) | set(dsts)) - known:
        raise ValueError(
            f"case {case.graph_id!r} has {len(dangling)} edge endpoints absent from its "
            f"node table; first few: {sorted(dangling)[:5]}"
        )

    timestamps = _column(edges, "timestamp")
    amounts_paid = _column(edges, "amount_paid")
    payment_currencies = _column(edges, "payment_currency")
    amounts_received = _column(edges, "amount_received")
    receiving_currencies = _column(edges, "receiving_currency")
    formats = _column(edges, "payment_format")
    laundering = _column(edges, "is_laundering")
    typologies = _column(edges, "typology")

    case_edges: list[CaseEdge] = []
    successors: dict[str, set[str]] = defaultdict(set)
    predecessors: dict[str, set[str]] = defaultdict(set)

    for i in range(edges.height):
        src, dst = srcs[i], dsts[i]
        case_edges.append(
            CaseEdge(
                src=src,
                dst=dst,
                timestamp=timestamps[i],
                amount_paid=None if amounts_paid[i] is None else float(amounts_paid[i]),
                payment_currency=(
                    None if payment_currencies[i] is None else str(payment_currencies[i])
                ),
                amount_received=(
                    None if amounts_received[i] is None else float(amounts_received[i])
                ),
                receiving_currency=(
                    None if receiving_currencies[i] is None else str(receiving_currencies[i])
                ),
                payment_format=None if formats[i] is None else str(formats[i]),
                is_laundering=None if laundering[i] is None else bool(laundering[i]),
                typology=None if typologies[i] is None else str(typologies[i]),
            )
        )
        if src != dst:
            successors[src].add(dst)
            predecessors[dst].add(src)

    banks: dict[str, str] | None = None
    if case.availability.institution_identity and "bank" in nodes.columns:
        banks = {
            str(node): str(bank)
            for node, bank in zip(nodes["node_id"].to_list(), nodes["bank"].to_list(), strict=True)
            if bank is not None
        }

    return CaseView(
        case_id=case.graph_id,
        dataset=case.dataset,
        availability=case.availability,
        node_ids=node_ids,
        edges=tuple(case_edges),
        successors={k: frozenset(v) for k, v in successors.items()},
        predecessors={k: frozenset(v) for k, v in predecessors.items()},
        banks=banks,
        label=case.label,
        typology=case.typology,
        provenance=dict(case.provenance),
    )
