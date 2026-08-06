"""Structural motif scoring: how much does a case *look* like a laundering typology?

This module exists for one population: **hard negatives**. A licit case whose structure
mimics a typology is where a narrative generator overclaims, and the AMLworld authors say
plainly that fan-in and fan-out appear in both normal and alert categories because
criminals mimic legitimate activity. Legitimate payroll is a fan-out. Legitimate
collections are a fan-in. Supplier settlement is a chain. A model that has only ever seen
fan-outs that were laundering has learned the wrong thing, and no amount of held-out
accuracy on easy negatives will reveal it.

The scorer is deliberately **structure-only**. It reads the shape of the subgraph and
nothing else — no label, no typology column, no laundering flag. That is not a stylistic
choice: if scoring could see the label, the mined hard negatives would be selected by the
label and the population would be worthless. :func:`score_motifs` therefore takes only the
node and edge topology, and :mod:`g2t_aml.data.leakage_audit` checks the resulting scores
do not separate labels.

Every score is in ``[0, 1]``, is a deterministic function of the topology, and saturates
at a scale taken from the substrate rather than invented: AMLworld's own patterns file
describes its fan-outs as "Max 16-degree", and its streams run to 32 transactions.

``random`` is not scored. It is the typology defined by having no structure, so a
"random-like" negative is just a negative.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from g2t_aml.data.canonical import CanonicalGraph

#: Motifs scored, in a fixed order so a score vector is positional and comparable.
#: ``random`` is deliberately absent — see the module docstring.
SCORED_MOTIFS: tuple[str, ...] = (
    "fan_out",
    "fan_in",
    "gather_scatter",
    "scatter_gather",
    "cycle",
    "bipartite",
    "stack",
)

#: Degree at which a fan saturates to 1.0. AMLworld generates fans up to 16-degree
#: ("Max 16-degree Fan-Out" in the patterns file), so 16 is the substrate's own scale.
FAN_SATURATION = 16

#: Minimum branching that counts as a fan at all. Two counterparties is a payment, not a
#: pattern.
FAN_FLOOR = 3

#: Branching at which a two-sided motif (gather-scatter, scatter-gather) saturates.
TWO_SIDED_SATURATION = 8

#: Minimum branching on each side before a two-sided motif counts as one at all.
TWO_SIDED_FLOOR = 2

#: Shortest directed cycle that counts. A two-node back-and-forth is a refund.
MIN_CYCLE_LENGTH = 3

#: Minimum nodes on each side before a two-colouring counts as bipartite structure.
MIN_BIPARTITE_SIDE = 2

#: Minimum accounts in a layer before it counts as a layer rather than a chain step.
MIN_LAYER_WIDTH = 2

#: Longest directed cycle searched for, and the length at which the cycle score saturates.
#: Bounded because cycle enumeration is exponential and a case can hold 150 nodes.
MAX_CYCLE_LENGTH = 6

#: Layer count at which the stack score saturates. AMLworld stacks are shallow.
STACK_SATURATION = 4

#: Above this edge count the cycle search is skipped and its score reported as 0.0. A case
#: that dense is not a clean cycle motif anyway, and the guard keeps mining bounded.
CYCLE_EDGE_BUDGET = 2_000


@dataclass(frozen=True)
class MotifScores:
    """Structural similarity of one case to each laundering typology.

    Attributes:
        scores: Motif name to score in ``[0, 1]``. Keys are exactly
            :data:`SCORED_MOTIFS`.
        best: The highest-scoring motif, or None when every score is zero.
        best_score: The highest score, 0.0 when the case has no structure at all.
    """

    scores: dict[str, float] = field(default_factory=dict)
    best: str | None = None
    best_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return the scores as a plain, JSON-serialisable mapping.

        Returns:
            ``{"scores": ..., "best": ..., "best_score": ...}``.
        """
        return {"scores": dict(self.scores), "best": self.best, "best_score": self.best_score}


@dataclass(frozen=True)
class _Topology:
    """The label-free view of a case that scoring is allowed to see."""

    successors: dict[str, set[str]]
    predecessors: dict[str, set[str]]
    nodes: tuple[str, ...]
    num_edges: int


def _topology(edges: pl.DataFrame) -> _Topology:
    """Reduce an edge table to simple directed adjacency, dropping self-loops.

    HI-Small carries 591,212 self-loops, 11.6% of all edges. A self-loop is not evidence
    of any structural typology and would inflate every degree, so it is excluded here —
    which is a scoring decision, not the cleaning D-017 declined to do: the case's edge
    table itself is untouched.

    Args:
        edges: A case's edge table, carrying ``src`` and ``dst``.

    Returns:
        The label-free topology.
    """
    successors: dict[str, set[str]] = defaultdict(set)
    predecessors: dict[str, set[str]] = defaultdict(set)
    seen: set[str] = set()
    count = 0
    for src, dst in zip(edges["src"].to_list(), edges["dst"].to_list(), strict=True):
        seen.add(src)
        seen.add(dst)
        if src == dst:
            continue
        successors[src].add(dst)
        predecessors[dst].add(src)
        count += 1
    return _Topology(
        successors=successors,
        predecessors=predecessors,
        nodes=tuple(sorted(seen)),
        num_edges=count,
    )


def _saturating(value: int, floor: int, saturation: int) -> float:
    """Map a count onto ``[0, 1]``, zero below a floor and one at saturation.

    Args:
        value: The observed count.
        floor: Below this the score is 0.0.
        saturation: At or above this the score is 1.0.

    Returns:
        The normalised score.
    """
    if value < floor:
        return 0.0
    if value >= saturation:
        return 1.0
    return (value - floor) / (saturation - floor)


def _fan_score(adjacency: dict[str, set[str]]) -> float:
    """Score the widest branching in one direction.

    Args:
        adjacency: Successor or predecessor sets.

    Returns:
        The saturating score of the largest distinct neighbour count.
    """
    widest = max((len(v) for v in adjacency.values()), default=0)
    return _saturating(widest, FAN_FLOOR, FAN_SATURATION)


def _gather_scatter_score(topology: _Topology) -> float:
    """Score many-to-one-to-many: one account collects, then disperses.

    Args:
        topology: The case topology.

    Returns:
        The saturating score of the best hub's ``min(in-degree, out-degree)``, ignoring
        counterparties that appear on both sides — money returned to its sender is not a
        gather-scatter.
    """
    best = 0
    for node in topology.nodes:
        incoming = topology.predecessors.get(node, set()) - {node}
        outgoing = topology.successors.get(node, set()) - {node}
        best = max(best, min(len(incoming - outgoing), len(outgoing - incoming)))
    return _saturating(best, TWO_SIDED_FLOOR, TWO_SIDED_SATURATION)


def _scatter_gather_score(topology: _Topology) -> float:
    """Score one-to-many-to-one: funds split across intermediaries, then recombined.

    Args:
        topology: The case topology.

    Returns:
        The saturating score of the largest number of distinct two-hop paths between one
        origin and one destination through disjoint intermediaries.
    """
    best = 0
    for origin in topology.nodes:
        intermediaries = topology.successors.get(origin, set()) - {origin}
        if len(intermediaries) < TWO_SIDED_FLOOR:
            continue
        recombined: dict[str, int] = defaultdict(int)
        for middle in intermediaries:
            for destination in topology.successors.get(middle, set()):
                if destination not in (origin, middle):
                    recombined[destination] += 1
        best = max(best, max(recombined.values(), default=0))
    return _saturating(best, TWO_SIDED_FLOOR, TWO_SIDED_SATURATION)


def _cycle_score(topology: _Topology) -> float:
    """Score the presence of a short directed cycle.

    Args:
        topology: The case topology.

    Returns:
        A score rising with the shortest cycle's length up to
        :data:`MAX_CYCLE_LENGTH`, 0.0 when none is found or the case exceeds
        :data:`CYCLE_EDGE_BUDGET`. Shortest rather than longest: a three-account round
        trip is a far cleaner cycle motif than a long meandering walk that happens to
        close.
    """
    if topology.num_edges > CYCLE_EDGE_BUDGET:
        return 0.0
    shortest = 0
    for origin in topology.nodes:
        # Bounded BFS back to the origin. Depth is capped, so this is linear in the
        # reachable set rather than exponential in path count.
        queue: deque[tuple[str, int]] = deque([(origin, 0)])
        visited = {origin}
        while queue:
            node, depth = queue.popleft()
            if depth >= MAX_CYCLE_LENGTH:
                continue
            for nxt in topology.successors.get(node, set()):
                if nxt == origin and depth + 1 >= MIN_CYCLE_LENGTH:
                    shortest = depth + 1 if shortest == 0 else min(shortest, depth + 1)
                    break
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, depth + 1))
            if shortest == MIN_CYCLE_LENGTH:
                break
        if shortest == MIN_CYCLE_LENGTH:
            break
    if shortest == 0:
        return 0.0
    # A 3-cycle is the strongest signal; longer closures score lower.
    return 1.0 - (shortest - MIN_CYCLE_LENGTH) / (2 * (MAX_CYCLE_LENGTH - MIN_CYCLE_LENGTH))


def _bipartite_score(topology: _Topology) -> float:
    """Score how cleanly the case splits into two sides that only trade across.

    Args:
        topology: The case topology.

    Returns:
        The fraction of edges that respect a greedy two-colouring, scaled by how balanced
        and how large the two sides are. 0.0 unless both sides hold at least two nodes.
    """
    if topology.num_edges == 0:
        return 0.0
    colour: dict[str, int] = {}
    for start in topology.nodes:
        if start in colour:
            continue
        colour[start] = 0
        queue: deque[str] = deque([start])
        while queue:
            node = queue.popleft()
            neighbours = topology.successors.get(node, set()) | topology.predecessors.get(
                node, set()
            )
            for neighbour in sorted(neighbours):
                if neighbour not in colour:
                    colour[neighbour] = 1 - colour[node]
                    queue.append(neighbour)

    left = sum(1 for c in colour.values() if c == 0)
    right = len(colour) - left
    if min(left, right) < MIN_BIPARTITE_SIDE:
        return 0.0
    respected = sum(
        1
        for src, targets in topology.successors.items()
        for dst in targets
        if colour[src] != colour[dst]
    )
    purity = respected / topology.num_edges
    balance = min(left, right) / max(left, right)
    return purity * balance


def _stack_score(topology: _Topology) -> float:
    """Score layered forwarding: a chain of hops where each layer holds several accounts.

    Args:
        topology: The case topology.

    Returns:
        The saturating score of the number of consecutive layers holding at least two
        accounts, measured from the widest available source layer.
    """
    sources = [n for n in topology.nodes if not topology.predecessors.get(n)]
    if not sources:
        sources = list(topology.nodes)
    best_layers = 0
    for start in sources:
        layer = {start}
        visited = {start}
        layers = 0
        while layer:
            nxt: set[str] = set()
            for node in layer:
                nxt |= topology.successors.get(node, set()) - visited
            if len(nxt) < MIN_LAYER_WIDTH:
                break
            visited |= nxt
            layers += 1
            layer = nxt
        best_layers = max(best_layers, layers)
    return _saturating(best_layers, MIN_LAYER_WIDTH, STACK_SATURATION)


def score_edges(edges: pl.DataFrame) -> MotifScores:
    """Score an edge table's structural similarity to each laundering typology.

    This is the real scorer. It takes an edge table rather than a case precisely so it
    *cannot* reach a label: mining hard negatives by a score that had seen the label would
    select the population by the label, and the resulting number would mean nothing.

    Args:
        edges: An edge table carrying ``src`` and ``dst``. Any other column is ignored.

    Returns:
        Scores for every member of :data:`SCORED_MOTIFS`, plus the best-scoring motif.

    Raises:
        ValueError: If the table lacks ``src`` or ``dst``.
    """
    if not {"src", "dst"} <= set(edges.columns):
        raise ValueError("motif scoring needs 'src' and 'dst' columns on the edge table")
    topology = _topology(edges)
    scores = {
        "fan_out": _fan_score(topology.successors),
        "fan_in": _fan_score(topology.predecessors),
        "gather_scatter": _gather_scatter_score(topology),
        "scatter_gather": _scatter_gather_score(topology),
        "cycle": _cycle_score(topology),
        "bipartite": _bipartite_score(topology),
        "stack": _stack_score(topology),
    }
    scores = {k: round(float(v), 6) for k, v in scores.items()}
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], SCORED_MOTIFS.index(kv[0])))
    best_name, best_value = ranked[0]
    if best_value <= 0.0:
        return MotifScores(scores=scores, best=None, best_score=0.0)
    return MotifScores(scores=scores, best=best_name, best_score=best_value)


def score_motifs(case: CanonicalGraph) -> MotifScores:
    """Score a case's structural similarity to each laundering typology.

    A thin convenience wrapper over :func:`score_edges`, which is where the contract that
    scoring never sees a label is enforced.

    Args:
        case: The case to score. Only its ``src``/``dst`` edge columns are read.

    Returns:
        The motif scores.

    Raises:
        ValueError: If the edge table lacks ``src`` or ``dst``.
    """
    return score_edges(case.edges)
