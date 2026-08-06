"""Topology: degree, density, diameter, components, reciprocity.

Available on every substrate by construction — a graph always has a shape, and no
availability flag can take that away. This is the one fact family that never carries a
sentinel, which makes it the family a narrative can always fall back on.

Every quantity here is defined over the **simple, self-loop-free directed projection**,
for the reasons in :mod:`g2t_aml.facts.caseview`. The two exceptions are ``n_edges`` and
``n_self_loops``, which report the transaction table as it actually is, because a
narrative that says "sixty-one transactions" must be able to be checked against the real
count rather than a cleaned one.
"""

from __future__ import annotations

from collections import deque

from g2t_aml.facts.caseview import CaseView
from g2t_aml.facts.schema import StructureFacts

__all__ = ["FIELD_PRODUCERS", "extract_structure"]

#: Below this node count, density and diameter are undefined: there is no ordered pair to
#: be dense over and no path to be long.
MIN_NODES_FOR_PAIRWISE_STATISTICS = 2

#: Field path to the named computation that produced it. Merged into
#: ``provenance.field_producers`` so a disagreement can be attributed to a specific
#: computation rather than to the record as a whole.
FIELD_PRODUCERS: dict[str, str] = {
    "structure.n_nodes": "structure.node_count",
    "structure.n_edges": "structure.transaction_count",
    "structure.n_components": "structure.weakly_connected_components",
    "structure.density": "structure.simple_directed_density",
    "structure.diameter": "structure.bfs_eccentricity_max",
    "structure.max_in_degree": "structure.distinct_neighbour_degree",
    "structure.max_out_degree": "structure.distinct_neighbour_degree",
    "structure.reciprocity": "structure.reciprocal_pair_share",
    "structure.n_self_loops": "structure.self_loop_count",
}


def distinct_pairs(view: CaseView) -> set[tuple[str, str]]:
    """Return the distinct ordered account pairs that transacted.

    HI-Small has 561,575 multi-edge node pairs with up to 89 parallel transactions
    (D-017). Density and reciprocity are properties of the *relationship* structure, so
    both are computed over distinct pairs; parallel transactions are counted by
    ``n_edges`` instead.

    Args:
        view: The case view.

    Returns:
        ``(src, dst)`` pairs, self-loops excluded.
    """
    return {(e.src, e.dst) for e in view.edges if not e.is_self_loop}


def density(view: CaseView) -> float:
    """Compute directed simple-graph density.

    Args:
        view: The case view.

    Returns:
        ``|distinct non-loop pairs| / (n * (n - 1))``, and 0.0 for a case with fewer than
        two accounts — where the denominator is zero and "density" has no meaning, so 0.0
        is reported rather than an error, and the checker treats a density claim on a
        one-node case as CONTRADICTED against that 0.0.
    """
    n = view.n_nodes
    if n < MIN_NODES_FOR_PAIRWISE_STATISTICS:
        return 0.0
    return len(distinct_pairs(view)) / (n * (n - 1))


def reciprocity(view: CaseView) -> float:
    """Compute the share of relationships that run both ways.

    Args:
        view: The case view.

    Returns:
        Fraction of distinct non-loop ordered pairs whose reverse pair also exists. 0.0
        when there are no such pairs.
    """
    pairs = distinct_pairs(view)
    if not pairs:
        return 0.0
    mutual = sum(1 for src, dst in pairs if (dst, src) in pairs)
    return mutual / len(pairs)


def diameter(view: CaseView) -> int | None:
    """Compute the longest shortest path over the undirected projection.

    Undirected rather than directed, deliberately. A directed diameter is infinite for
    almost every case here — a fan-out has no path back to its hub — and an infinite
    diameter is not a fact a narrative can use. The undirected projection answers the
    question an investigator actually asks: how far apart are the two most distant
    accounts in this case.

    Maximised *within* each weakly-connected component, so a fragmented case reports the
    widest span it actually contains rather than a null.

    Args:
        view: The case view.

    Returns:
        The eccentricity maximum in hops, or None when the case has fewer than two
        accounts.
    """
    if view.n_nodes < MIN_NODES_FOR_PAIRWISE_STATISTICS:
        return None
    adjacency = view.undirected_neighbours()
    best = 0
    for origin in view.node_ids:
        distance: dict[str, int] = {origin: 0}
        queue: deque[str] = deque([origin])
        while queue:
            node = queue.popleft()
            for neighbour in sorted(adjacency.get(node, frozenset())):
                if neighbour not in distance:
                    distance[neighbour] = distance[node] + 1
                    queue.append(neighbour)
        best = max(best, *distance.values())
    return best


def extract_structure(view: CaseView) -> StructureFacts:
    """Extract the whole structure block.

    Args:
        view: The case view.

    Returns:
        The populated :class:`~g2t_aml.facts.schema.StructureFacts`. Never a sentinel:
        topology is available on every substrate.
    """
    return StructureFacts(
        n_nodes=view.n_nodes,
        n_edges=view.n_edges,
        n_components=len(view.weakly_connected_components()),
        density=round(density(view), 6),
        diameter=diameter(view),
        max_in_degree=max((view.in_degree(n) for n in view.node_ids), default=0),
        max_out_degree=max((view.out_degree(n) for n in view.node_ids), default=0),
        reciprocity=round(reciprocity(view), 6),
        n_self_loops=view.n_self_loops,
    )
