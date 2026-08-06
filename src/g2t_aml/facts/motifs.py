"""Eight structural detectors, each returning presence plus quantitative descriptors.

**These are not the scorers in :mod:`g2t_aml.data.motifs`, and the difference matters.**
That module answers "how much does this case *resemble* a typology", on a continuous [0,1]
scale, so hard-negative mining can rank licit cases by how deceptive they look. This module
answers "does this case *contain* a fan-out, and how wide is it" — a boolean decision
against a recorded threshold, plus the numbers a narrative will quote. A soft score cannot
be put in a sentence; a width can be, and can then be checked. The two modules deliberately
do not share code: collapsing them would force one set of thresholds to serve two purposes
and would let a change made for mining silently move a published faithfulness number. See
D-031.

Every detector carries a **witness** — the accounts evidencing the structure. Witnesses are
not serialised into the fact record, but they are what the property-based tests assert
against: a reported cycle of length 4 must be four accounts each of which really does send
to the next. A detector that reports a structure it cannot exhibit is the exact failure mode
this module has to be free of.

Self-loops are excluded from every detector, per :mod:`g2t_aml.facts.caseview`. Without
that, HI-Small's 591,212 self-loops would each register as a one-node cycle.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime

from g2t_aml.facts.caseview import CaseEdge, CaseView
from g2t_aml.facts.config import FactConfig
from g2t_aml.facts.schema import MotifFacts, MotifResult

__all__ = [
    "FIELD_PRODUCERS",
    "detect_bipartite",
    "detect_chain",
    "detect_cycle",
    "detect_fan_in",
    "detect_fan_out",
    "detect_gather_scatter",
    "detect_scatter_gather",
    "detect_stack",
    "extract_motifs",
]

FIELD_PRODUCERS: dict[str, str] = {
    "motifs.fan_in.present": "motifs.widest_in_star",
    "motifs.fan_in.width": "motifs.widest_in_star",
    "motifs.fan_in.hub": "motifs.widest_in_star",
    "motifs.fan_in.window_hours": "motifs.widest_in_star",
    "motifs.fan_out.present": "motifs.widest_out_star",
    "motifs.fan_out.width": "motifs.widest_out_star",
    "motifs.fan_out.hub": "motifs.widest_out_star",
    "motifs.fan_out.window_hours": "motifs.widest_out_star",
    "motifs.chain.present": "motifs.longest_simple_path",
    "motifs.chain.max_length": "motifs.longest_simple_path",
    "motifs.cycle.present": "motifs.shortest_directed_cycle",
    "motifs.cycle.length": "motifs.shortest_directed_cycle",
    "motifs.bipartite.present": "motifs.two_colouring",
    "motifs.bipartite.score": "motifs.two_colouring",
    "motifs.bipartite.left_size": "motifs.two_colouring",
    "motifs.bipartite.right_size": "motifs.two_colouring",
    "motifs.stack.present": "motifs.layered_bfs",
    "motifs.stack.depth": "motifs.layered_bfs",
    "motifs.stack.layer_widths": "motifs.layered_bfs",
    "motifs.gather_scatter.present": "motifs.hub_in_then_out",
    "motifs.gather_scatter.gather_width": "motifs.hub_in_then_out",
    "motifs.gather_scatter.scatter_width": "motifs.hub_in_then_out",
    "motifs.gather_scatter.hub": "motifs.hub_in_then_out",
    "motifs.scatter_gather.present": "motifs.two_hop_recombination",
    "motifs.scatter_gather.width": "motifs.two_hop_recombination",
    "motifs.scatter_gather.origin": "motifs.two_hop_recombination",
    "motifs.scatter_gather.destination": "motifs.two_hop_recombination",
}

_SECONDS_PER_HOUR = 3600.0


def _span_hours(stamps: list[datetime]) -> float | None:
    """Return the extent of a set of timestamps, in hours.

    Args:
        stamps: Transaction timestamps.

    Returns:
        The span, 0.0 for a single timestamp, or None when the list is empty — which is
        what a substrate without a clock produces.
    """
    if not stamps:
        return None
    return round((max(stamps) - min(stamps)).total_seconds() / _SECONDS_PER_HOUR, 6)


def _edge_span(edges: tuple[CaseEdge, ...]) -> float | None:
    """Return the time span of a set of transactions.

    Args:
        edges: The transactions forming a motif.

    Returns:
        The span in hours, or None when no transaction carries a timestamp.
    """
    return _span_hours([e.timestamp for e in edges if e.timestamp is not None])


def _absent(**descriptors: object) -> MotifResult:
    """Build a not-present result with every descriptor nulled.

    Args:
        **descriptors: Descriptor names, whose values are ignored and replaced by None.

    Returns:
        A :class:`~g2t_aml.facts.schema.MotifResult` with ``present=False``.
    """
    return MotifResult(present=False, descriptors=dict.fromkeys(descriptors))


# ----------------------------------------------------------------- fans ---


def _widest_star(
    view: CaseView, adjacency: dict[str, frozenset[str]], config: FactConfig, *, inbound: bool
) -> MotifResult:
    """Find the widest one-directional star and report it.

    Args:
        view: The case view.
        adjacency: ``predecessors`` for a fan-in, ``successors`` for a fan-out.
        config: Supplies ``fan_min_width``.
        inbound: Whether the star is inbound, which decides how the forming
            transactions are collected for the window span.

    Returns:
        The widest star, present only when it reaches ``fan_min_width``. Ties on width
        are broken by hub identifier, so the result never depends on iteration order.
    """
    best_hub: str | None = None
    best_width = 0
    # node_ids is sorted, so a strict > keeps the FIRST hub achieving the maximum, which
    # is the lexicographically smallest. That is the whole tie-break rule.
    for node in view.node_ids:
        width = len(adjacency.get(node, frozenset()))
        if width > best_width:
            best_hub, best_width = node, width

    if best_hub is None or best_width < config.fan_min_width:
        return _absent(width=None, hub=None, window_hours=None)

    forming = view.edges_into(best_hub) if inbound else view.edges_out_of(best_hub)
    return MotifResult(
        present=True,
        descriptors={
            "width": best_width,
            "hub": best_hub,
            "window_hours": _edge_span(forming),
        },
        witness=(best_hub, *sorted(adjacency[best_hub])),
    )


def detect_fan_in(view: CaseView, config: FactConfig) -> MotifResult:
    """Detect many distinct senders converging on one account.

    Args:
        view: The case view.
        config: Supplies ``fan_min_width``.

    Returns:
        Presence, the widest hub's identifier, its distinct-sender count, and the time
        span of the transactions forming it.
    """
    return _widest_star(view, view.predecessors, config, inbound=True)


def detect_fan_out(view: CaseView, config: FactConfig) -> MotifResult:
    """Detect one account dispersing to many distinct recipients.

    Args:
        view: The case view.
        config: Supplies ``fan_min_width``.

    Returns:
        Presence, the widest hub's identifier, its distinct-recipient count, and the time
        span of the transactions forming it.
    """
    return _widest_star(view, view.successors, config, inbound=False)


# ---------------------------------------------------------------- chain ---


def detect_chain(view: CaseView, config: FactConfig) -> MotifResult:
    """Detect a long simple directed path, and report the longest one found.

    Longest simple path is NP-hard, and a case may hold 150 accounts. The search is an
    exhaustive DFS bounded by ``config.path_node_budget``; above that budget it falls
    back to a **layered lower bound** — the longest path found by greedy forward
    extension from each source — which is admissible in the direction that matters: it
    can under-report a long chain but never invent one. Under-reporting turns a narrative
    claim into CONTRADICTED rather than SUPPORTED, which is the safe failure for a
    measurement instrument, and the fallback is recorded because ``path_node_budget`` is
    written into ``provenance.config``.

    ``max_length`` is reported whether or not the motif is present, so a narrative
    claiming a five-step chain in a case whose longest is two is CONTRADICTED rather than
    UNVERIFIABLE.

    Args:
        view: The case view.
        config: Supplies ``chain_min_length`` and ``path_node_budget``.

    Returns:
        Presence and the longest simple directed path length, in edges.
    """
    if view.n_nodes <= config.path_node_budget:
        length, witness = _longest_path_exact(view)
    else:
        length, witness = _longest_path_greedy(view)
    return MotifResult(
        present=length >= config.chain_min_length,
        descriptors={"max_length": length},
        witness=witness,
    )


def _longest_path_exact(view: CaseView) -> tuple[int, tuple[str, ...]]:
    """Find the longest simple directed path by exhaustive DFS.

    Args:
        view: The case view.

    Returns:
        ``(length_in_edges, path)``.
    """
    best_length = 0
    best_path: tuple[str, ...] = ()

    def walk(node: str, visited: set[str], path: list[str]) -> None:
        nonlocal best_length, best_path
        if len(path) - 1 > best_length:
            best_length = len(path) - 1
            best_path = tuple(path)
        for nxt in sorted(view.successors.get(node, frozenset())):
            if nxt not in visited:
                visited.add(nxt)
                path.append(nxt)
                walk(nxt, visited, path)
                path.pop()
                visited.remove(nxt)

    for start in view.node_ids:
        walk(start, {start}, [start])
    return best_length, best_path


def _longest_path_greedy(view: CaseView) -> tuple[int, tuple[str, ...]]:
    """Find a long simple directed path by greedy forward extension.

    The admissible fallback described in :func:`detect_chain`: it returns a real path, so
    it never over-reports, and it is linear per start node.

    Args:
        view: The case view.

    Returns:
        ``(length_in_edges, path)`` for the longest path found.
    """
    best_length = 0
    best_path: tuple[str, ...] = ()
    for start in view.node_ids:
        path = [start]
        visited = {start}
        while True:
            candidates = sorted(view.successors.get(path[-1], frozenset()) - visited)
            if not candidates:
                break
            # Extend toward the successor with the most onward options, ties by
            # identifier, so the walk is deterministic and tends to stay on the longer
            # branch.
            nxt = max(candidates, key=lambda n: (len(view.successors.get(n, frozenset())), n))
            path.append(nxt)
            visited.add(nxt)
        if len(path) - 1 > best_length:
            best_length = len(path) - 1
            best_path = tuple(path)
    return best_length, best_path


# ---------------------------------------------------------------- cycle ---


def detect_cycle(view: CaseView, config: FactConfig) -> MotifResult:
    """Detect the shortest directed cycle at or above the configured minimum length.

    Shortest rather than longest: a three-account round trip is a far cleaner cycle motif
    than a long meandering walk that happens to close, and it is the one an investigator
    would describe.

    Args:
        view: The case view.
        config: Supplies ``cycle_min_length``, ``cycle_max_length`` and
            ``cycle_edge_budget``.

    Returns:
        Presence and the shortest qualifying cycle's length in edges, with the cycle's
        accounts as the witness. Absent — rather than raising — when the case exceeds
        ``cycle_edge_budget``; that budget is recorded in ``provenance.config``, so the
        skip is never invisible.
    """
    if len(view.non_loop_edges()) > config.cycle_edge_budget:
        return _absent(length=None)

    best_length = 0
    best_cycle: tuple[str, ...] = ()
    for origin in view.node_ids:
        # BFS from the origin, recording predecessors so the cycle can be reconstructed
        # as a witness. Depth is capped, so this is linear in the reachable set rather
        # than exponential in path count.
        parent: dict[str, str] = {}
        depth: dict[str, int] = {origin: 0}
        queue: deque[str] = deque([origin])
        while queue:
            node = queue.popleft()
            if depth[node] >= config.cycle_max_length:
                continue
            for nxt in sorted(view.successors.get(node, frozenset())):
                if nxt == origin:
                    length = depth[node] + 1
                    if length >= config.cycle_min_length and (
                        best_length == 0 or length < best_length
                    ):
                        best_length = length
                        best_cycle = _reconstruct(origin, node, parent)
                    continue
                if nxt not in depth:
                    depth[nxt] = depth[node] + 1
                    parent[nxt] = node
                    queue.append(nxt)
        if best_length == config.cycle_min_length:
            break  # cannot do better; stop early

    if best_length == 0:
        return _absent(length=None)
    return MotifResult(present=True, descriptors={"length": best_length}, witness=best_cycle)


def _reconstruct(origin: str, tail: str, parent: dict[str, str]) -> tuple[str, ...]:
    """Rebuild the cycle path from a BFS parent map.

    Args:
        origin: The node the cycle closes on.
        tail: The last node before closing back to ``origin``.
        parent: BFS predecessor map.

    Returns:
        The cycle's accounts in traversal order, starting at ``origin``.
    """
    path = [tail]
    while path[-1] != origin:
        path.append(parent[path[-1]])
    return tuple(reversed(path))


# ------------------------------------------------------------ bipartite ---


def detect_bipartite(view: CaseView, config: FactConfig) -> MotifResult:
    """Detect a clean two-sided structure: two groups that only trade across.

    ``present`` requires the case to be **exactly** bipartite — no odd cycle survives the
    two-colouring — with both sides at least ``bipartite_min_side``. ``score`` is the
    continuous ``purity * balance`` measure and is reported regardless, so a narrative
    overclaiming clean two-sidedness in a nearly-bipartite case is CONTRADICTED against a
    number rather than left UNVERIFIABLE.

    Args:
        view: The case view.
        config: Supplies ``bipartite_min_side``.

    Returns:
        Presence, the score, and the two side sizes.
    """
    edges = view.non_loop_edges()
    if not edges:
        return MotifResult(
            present=False,
            descriptors={"score": 0.0, "left_size": None, "right_size": None},
        )

    adjacency = view.undirected_neighbours()
    colour: dict[str, int] = {}
    conflict = False
    for start in view.node_ids:
        if start in colour:
            continue
        colour[start] = 0
        queue: deque[str] = deque([start])
        while queue:
            node = queue.popleft()
            for neighbour in sorted(adjacency.get(node, frozenset())):
                if neighbour not in colour:
                    colour[neighbour] = 1 - colour[node]
                    queue.append(neighbour)
                elif colour[neighbour] == colour[node]:
                    conflict = True

    left = sorted(n for n, c in colour.items() if c == 0)
    right = sorted(n for n, c in colour.items() if c == 1)
    respected = sum(1 for e in edges if colour[e.src] != colour[e.dst])
    purity = respected / len(edges)
    balance = min(len(left), len(right)) / max(len(left), len(right)) if right else 0.0
    score = round(purity * balance, 6)

    present = (
        not conflict and min(len(left), len(right)) >= config.bipartite_min_side and purity == 1.0
    )
    return MotifResult(
        present=present,
        descriptors={
            "score": score,
            "left_size": len(left) if present else None,
            "right_size": len(right) if present else None,
        },
        witness=tuple(left + right) if present else (),
    )


# ---------------------------------------------------------------- stack ---


def detect_stack(view: CaseView, config: FactConfig) -> MotifResult:
    """Detect layered forwarding: successive layers each holding several accounts.

    Layers are built by BFS from each source account (one with no in-neighbours), taking
    the frontier of not-yet-visited successors as the next layer. A layer narrower than
    ``stack_min_layer_width`` ends the stack — a single onward hop is a chain, not a
    layer.

    Args:
        view: The case view.
        config: Supplies ``stack_min_depth`` and ``stack_min_layer_width``.

    Returns:
        Presence, the depth in layers, and each layer's width.
    """
    sources = [n for n in view.node_ids if not view.predecessors.get(n)]
    if not sources:
        sources = list(view.node_ids)

    best_widths: list[int] = []
    best_witness: tuple[str, ...] = ()
    for start in sources:
        layer = {start}
        visited = {start}
        widths: list[int] = []
        members: list[str] = [start]
        while True:
            nxt: set[str] = set()
            for node in layer:
                nxt |= view.successors.get(node, frozenset()) - visited
            if len(nxt) < config.stack_min_layer_width:
                break
            visited |= nxt
            widths.append(len(nxt))
            members.extend(sorted(nxt))
            layer = nxt
        if len(widths) > len(best_widths):
            best_widths = widths
            best_witness = tuple(members)

    depth = len(best_widths)
    if depth < config.stack_min_depth:
        return _absent(depth=None, layer_widths=None)
    return MotifResult(
        present=True,
        descriptors={"depth": depth, "layer_widths": list(best_widths)},
        witness=best_witness,
    )


# --------------------------------------------------- two-sided composites ---


def detect_gather_scatter(view: CaseView, config: FactConfig) -> MotifResult:
    """Detect one account collecting from many, then dispersing to many others.

    Counterparties appearing on *both* sides are excluded from both widths: money
    returned to the account that sent it is not a gather-scatter, and counting it as one
    would make every reciprocal relationship look like layering.

    Args:
        view: The case view.
        config: Supplies ``two_sided_min_width``.

    Returns:
        Presence, the hub, and its gather and scatter widths.
    """
    best_hub: str | None = None
    best_gather = best_scatter = 0
    best_rank = (-1, -1)
    # Rank on the weaker side first: a hub with 12 in and 1 out is a fan-in, not a
    # gather-scatter, and min() is what refuses to call it one. node_ids is sorted and
    # the comparison is strict, so ties keep the lexicographically smallest hub.
    for node in view.node_ids:
        incoming = view.predecessors.get(node, frozenset()) - {node}
        outgoing = view.successors.get(node, frozenset()) - {node}
        gather = len(incoming - outgoing)
        scatter = len(outgoing - incoming)
        rank = (min(gather, scatter), gather + scatter)
        if rank > best_rank:
            best_hub, best_gather, best_scatter, best_rank = node, gather, scatter, rank

    minimum = config.two_sided_min_width
    if best_hub is None or min(best_gather, best_scatter) < minimum:
        return _absent(gather_width=None, scatter_width=None, hub=None)

    incoming = view.predecessors.get(best_hub, frozenset()) - {best_hub}
    outgoing = view.successors.get(best_hub, frozenset()) - {best_hub}
    return MotifResult(
        present=True,
        descriptors={
            "gather_width": best_gather,
            "scatter_width": best_scatter,
            "hub": best_hub,
        },
        witness=(best_hub, *sorted(incoming - outgoing), *sorted(outgoing - incoming)),
    )


def detect_scatter_gather(view: CaseView, config: FactConfig) -> MotifResult:
    """Detect one origin splitting across intermediaries that recombine at one destination.

    Args:
        view: The case view.
        config: Supplies ``two_sided_min_width``.

    Returns:
        Presence, the origin, the destination, and the number of distinct intermediaries
        on two-hop paths between them.
    """
    best: tuple[int, str, str, tuple[str, ...]] | None = None
    for origin in view.node_ids:
        intermediaries = view.successors.get(origin, frozenset()) - {origin}
        if len(intermediaries) < config.two_sided_min_width:
            continue
        recombined: dict[str, list[str]] = {}
        for middle in sorted(intermediaries):
            for destination in sorted(view.successors.get(middle, frozenset())):
                if destination in (origin, middle):
                    continue
                recombined.setdefault(destination, []).append(middle)
        for destination in sorted(recombined):
            width = len(recombined[destination])
            # Widest wins. Origins and destinations are both visited in sorted order and
            # the comparison is strict, so ties keep the lexicographically smallest
            # (origin, destination) pair rather than whichever the iteration reached
            # first — a total order over case content, not over dictionary layout.
            if best is None or width > best[0]:
                best = (width, origin, destination, tuple(recombined[destination]))

    if best is None or best[0] < config.two_sided_min_width:
        return _absent(width=None, origin=None, destination=None)
    width, origin, destination, middles = best
    return MotifResult(
        present=True,
        descriptors={"width": width, "origin": origin, "destination": destination},
        witness=(origin, *middles, destination),
    )


def extract_motifs(view: CaseView, config: FactConfig) -> MotifFacts:
    """Run every detector.

    Args:
        view: The case view.
        config: Every detector threshold.

    Returns:
        The populated :class:`~g2t_aml.facts.schema.MotifFacts`.
    """
    return MotifFacts(
        fan_in=detect_fan_in(view, config),
        fan_out=detect_fan_out(view, config),
        chain=detect_chain(view, config),
        cycle=detect_cycle(view, config),
        bipartite=detect_bipartite(view, config),
        stack=detect_stack(view, config),
        gather_scatter=detect_gather_scatter(view, config),
        scatter_gather=detect_scatter_gather(view, config),
    )
