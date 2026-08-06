"""Case construction: turning a transaction stream into bounded, provenanced cases.

A SAR narrative describes a **case** — a bounded set of related transactions and entities.
The two substrates hand us that boundary very differently, and the difference is the whole
reason this module exists.

**Elliptic2 provides the case.** A labelled subgraph *is* the case, so
:func:`passthrough_case` records ``extraction_method="provided"`` and constructs nothing.

**AMLworld does not.** It is a flat stream of 5,078,345 transactions with no case
boundary anywhere in it, so we construct one, and the construction rule is a design choice
a reviewer will interrogate. It is therefore deterministic, parameterised, and recorded in
full on every case it produces::

    extract_case(seed account a, window W, k_hops, n_max, prune_rule, seed):
      1. Collect all transactions incident to a within window W.
      2. Expand k hops along transaction edges (default k=2).
      3. If |V| > n_max: prune by edge amount descending, but ALWAYS retain every edge
         lying on a labelled-laundering path.
      4. Record provenance.

Three properties are load-bearing.

**Determinism.** Same inputs produce a byte-identical serialisation, node ordering
included. There is no RNG anywhere in extraction: ``seed`` is accepted, recorded in the
provenance and folded into the case identifier, but the algorithm never draws from it.
Every ordering decision is made by an explicit total order over edge *content*, so a
re-ordered input frame still yields an identical case. Sampling (``case_sampling``) is
where randomness lives.

**The neighbour cap.** HI-Small's out-degree runs to 168,672 against a median of 2. A
single un-capped hop through one hub produces a case larger than the entire pruning budget
and dominated by an account that has nothing to do with the seed. Expansion therefore
admits at most ``max_neighbours_per_node`` incident edges per node, ranked
laundering-first then by amount. The cap is a parameter, is recorded in the provenance,
and is reported per case as ``neighbour_cap_triggered``.

**Laundering-path preservation.** Pruning that severs the very path the case exists to
describe is worse than no pruning. Every edge carrying a laundering label is retained
before the budget is spent, even when doing so overruns ``n_max`` — in which case the
provenance says so via ``n_max_exceeded``, rather than the case quietly losing its
evidence.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

import numpy as np
import polars as pl

from g2t_aml.data.canonical import TYPOLOGY_VOCABULARY, CanonicalGraph
from g2t_aml.utils.hashing import canonical_json, short

#: Bumping this invalidates every case built under the old protocol, in the same way
#: CANONICAL_SCHEMA_VERSION invalidates an interim graph. It is recorded on every case.
EXTRACTION_PROTOCOL_VERSION = "1.0.0"

#: Prune rules the protocol understands. ``amount_desc`` is the default and the one
#: reported in the paper; the other two exist so the sensitivity analysis can show the
#: choice is not doing the work.
PruneRule = Literal["amount_desc", "recency", "degree"]
PRUNE_RULES: tuple[str, ...] = ("amount_desc", "recency", "degree")

#: Default per-node expansion cap. 64 is comfortably above the 16-degree fan-out AMLworld
#: actually generates ("Max 16-degree Fan-Out" in the patterns file), so no synthetic
#: typology can be truncated by it, while still bounding a 168,672-degree hub.
DEFAULT_MAX_NEIGHBOURS = 64

#: Columns used, in this priority order, to impose a canonical order on the edge table
#: before a case is returned. Only those actually present participate, so a fixture graph
#: without timestamps still sorts deterministically.
_EDGE_SORT_PRIORITY: tuple[str, ...] = (
    "timestamp",
    "src",
    "dst",
    "amount_paid",
    "amount_received",
    "transaction_key",
)

#: Edge columns that mark a transaction as lying on a labelled-laundering path.
_LAUNDERING_COLUMNS: tuple[str, ...] = ("is_laundering", "pattern_id")


class CaseExtractionError(ValueError):
    """Raised when a case cannot be constructed from the inputs given."""


@dataclass(frozen=True, order=True)
class TimeWindow:
    """A closed time interval ``[start, end]``.

    Both bounds are inclusive: a transaction stamped exactly at ``end`` belongs to the
    window. AMLworld timestamps have minute resolution, so an exclusive upper bound would
    drop whole minutes at a window edge for no benefit.

    Attributes:
        start: Inclusive lower bound.
        end: Inclusive upper bound.
    """

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        """Validate the interval.

        Raises:
            ValueError: If ``end`` precedes ``start``.
        """
        if self.end < self.start:
            raise ValueError(f"time window ends before it starts: {self.start} > {self.end}")

    @property
    def duration(self) -> timedelta:
        """Return the window length.

        Returns:
            ``end - start``.
        """
        return self.end - self.start

    def padded(self, hours: float) -> TimeWindow:
        """Return a copy widened by ``hours`` on each side.

        Args:
            hours: Padding applied symmetrically. Must not be negative.

        Returns:
            The widened window.

        Raises:
            ValueError: If ``hours`` is negative.
        """
        if hours < 0:
            raise ValueError(f"padding must not be negative, got {hours}")
        delta = timedelta(hours=hours)
        return TimeWindow(start=self.start - delta, end=self.end + delta)

    def contains(self, moment: datetime) -> bool:
        """Report whether a moment falls inside the closed interval.

        Args:
            moment: The timestamp to test.

        Returns:
            True if ``start <= moment <= end``.
        """
        return self.start <= moment <= self.end

    def overlaps(self, other: TimeWindow) -> bool:
        """Report whether two closed intervals intersect.

        Args:
            other: The window to test against.

        Returns:
            True if the intervals share at least one instant.
        """
        return self.start <= other.end and other.start <= self.end

    def straddles(self, boundary: datetime) -> bool:
        """Report whether a boundary falls strictly inside the window.

        Used by :mod:`g2t_aml.data.splits` to drop cases that would otherwise be assigned
        to one side of a temporal boundary while containing evidence from the other.

        Args:
            boundary: The instant to test.

        Returns:
            True if ``start < boundary < end``.
        """
        return self.start < boundary < self.end

    def to_dict(self) -> dict[str, str]:
        """Return the window as ISO-8601 strings.

        Returns:
            ``{"start": ..., "end": ...}``.
        """
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TimeWindow:
        """Rebuild a window from :meth:`to_dict` output.

        Args:
            data: Mapping with ``start`` and ``end`` ISO-8601 strings or datetimes.

        Returns:
            The reconstructed window.

        Raises:
            KeyError: If either bound is missing.
            ValueError: If a bound is not parseable, or the interval is inverted.
        """
        start, end = data["start"], data["end"]
        return cls(
            start=start if isinstance(start, datetime) else datetime.fromisoformat(str(start)),
            end=end if isinstance(end, datetime) else datetime.fromisoformat(str(end)),
        )


@dataclass(frozen=True)
class ExtractionParams:
    """The complete parameterisation of the case-construction protocol.

    Every field appears in each case's provenance and in the split manifest, so a case can
    always be traced back to the exact rule that built it.

    Attributes:
        k_hops: Expansion radius, in undirected transaction hops from the seed.
        n_max: Node budget. Exceeded only by laundering-path preservation, which is then
            flagged as ``n_max_exceeded``.
        prune_rule: Ranking used to spend the node budget. See :data:`PRUNE_RULES`.
        preserve_laundering_paths: Retain every labelled-laundering edge before the budget
            is spent.
        seed: Recorded and folded into the case identifier. Extraction draws no random
            numbers; see the module docstring.
        max_neighbours_per_node: Per-node expansion cap. See the module docstring.
        protocol_version: :data:`EXTRACTION_PROTOCOL_VERSION` at construction time.
    """

    k_hops: int = 2
    n_max: int = 150
    prune_rule: PruneRule = "amount_desc"
    preserve_laundering_paths: bool = True
    seed: int = 1337
    max_neighbours_per_node: int = DEFAULT_MAX_NEIGHBOURS
    protocol_version: str = EXTRACTION_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        """Validate the parameterisation.

        Raises:
            ValueError: If ``k_hops`` is negative, ``n_max`` or
                ``max_neighbours_per_node`` is not positive, or ``prune_rule`` is unknown.
        """
        if self.k_hops < 0:
            raise ValueError(f"k_hops must be >= 0, got {self.k_hops}")
        if self.n_max < 1:
            raise ValueError(f"n_max must be >= 1, got {self.n_max}")
        if self.max_neighbours_per_node < 1:
            raise ValueError(
                f"max_neighbours_per_node must be >= 1, got {self.max_neighbours_per_node}"
            )
        if self.prune_rule not in PRUNE_RULES:
            raise ValueError(f"unknown prune_rule {self.prune_rule!r}; expected {PRUNE_RULES}")

    def to_dict(self) -> dict[str, Any]:
        """Return the parameters as a plain dict.

        Returns:
            Field name to value, in declaration order.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExtractionParams:
        """Rebuild parameters from a mapping, ignoring unknown keys is *not* allowed.

        Args:
            data: Mapping whose keys are a subset of the field names.

        Returns:
            The reconstructed parameters.

        Raises:
            ValueError: If ``data`` carries a key that is not a field, so a stale manifest
                fails loudly rather than silently defaulting a parameter.
        """
        known = {f.name for f in dataclasses.fields(cls)}
        if unknown := set(data) - known:
            raise ValueError(f"unknown extraction parameters: {sorted(unknown)}")
        return cls(**data)


def case_id_for(
    dataset: str,
    seed_node: str,
    window: TimeWindow,
    params: ExtractionParams,
) -> str:
    """Compute the deterministic identifier for a case.

    The identifier is a function of everything that determines the case's content, so two
    runs of the pipeline over the same interim graph produce the same identifiers and a
    split manifest stays valid. It is *not* a function of the extracted content itself:
    that would make the identifier unavailable until after extraction, and would change
    whenever an upstream loader fix altered a single amount.

    Args:
        dataset: Substrate key, e.g. ``"amlworld_hi_small"``.
        seed_node: The seed account's canonical node id.
        window: The case's time window.
        params: The extraction parameterisation.

    Returns:
        ``"<dataset>-<16 hex chars>"``.
    """
    payload = {
        "dataset": dataset,
        "seed_node": seed_node,
        "window": window.to_dict(),
        "params": params.to_dict(),
    }
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"{dataset}-{short(digest, 16)}"


class GraphIndex:
    """A traversal index over a :class:`CanonicalGraph`, built once and reused.

    Extraction is called tens of thousands of times against the same 5-million-edge graph,
    so the adjacency is materialised once as sorted CSR-style offset arrays over integer
    node indices. Node identifiers, edge attributes and everything else stay in the
    original Polars frames: a case is materialised by gathering rows by position, which
    also carries every edge attribute across for free.

    Attributes:
        graph: The graph being indexed. Held by reference, not copied.
        num_nodes: Node count.
        num_edges: Edge count.
    """

    def __init__(self, graph: CanonicalGraph) -> None:
        """Build the index.

        Args:
            graph: The graph to index. Its node table must have unique ``node_id`` values
                and every edge endpoint must appear in it.

        Raises:
            CaseExtractionError: If ``node_id`` is not unique, or an edge endpoint is
                absent from the node table.
        """
        self.graph = graph
        self.num_nodes = graph.num_nodes
        self.num_edges = graph.num_edges
        # Materialising a case is a positional gather, and a gather across a chunked
        # frame is O(total rows): HI-Small arrives from Parquet in 19 chunks, which cost
        # 370 ms per case against 0.2 ms once contiguous. Paying one rechunk here turns a
        # two-hour build into a two-minute one.
        self.nodes = graph.nodes.rechunk()
        self.edges = graph.edges.rechunk()

        node_ids = self.nodes["node_id"]
        if node_ids.n_unique() != self.num_nodes:
            raise CaseExtractionError("node table has duplicate node_id values")
        self._node_ids = node_ids
        self._position = {nid: i for i, nid in enumerate(node_ids.to_list())}

        lookup = pl.DataFrame(
            {"node_id": node_ids, "_idx": np.arange(self.num_nodes, dtype=np.int64)}
        )
        endpoints = (
            self.edges.select("src", "dst")
            .join(lookup.rename({"node_id": "src", "_idx": "_src"}), on="src", how="left")
            .join(lookup.rename({"node_id": "dst", "_idx": "_dst"}), on="dst", how="left")
        )
        if endpoints["_src"].null_count() or endpoints["_dst"].null_count():
            raise CaseExtractionError(
                "edge endpoints missing from the node table; run "
                "CanonicalGraph.validate_referential_integrity() to see which"
            )
        self.src_index = endpoints["_src"].to_numpy().astype(np.int64, copy=False)
        self.dst_index = endpoints["_dst"].to_numpy().astype(np.int64, copy=False)

        self.timestamps: np.ndarray | None = None
        if "timestamp" in self.edges.columns:
            self.timestamps = (
                self.edges["timestamp"].cast(pl.Int64).to_numpy().astype(np.int64, copy=False)
            )

        amount_column = next(
            (c for c in ("amount_paid", "amount_received") if c in self.edges.columns), None
        )
        self.amounts = (
            self.edges[amount_column].fill_null(0.0).to_numpy().astype(np.float64, copy=False)
            if amount_column is not None
            else np.zeros(self.num_edges, dtype=np.float64)
        )
        self.laundering = _laundering_mask(self.edges)

        # Global degree, used by the "degree" prune rule and by activity matching.
        self.degree = np.bincount(self.src_index, minlength=self.num_nodes) + np.bincount(
            self.dst_index, minlength=self.num_nodes
        )

        self.first_seen, self.last_seen = _activity_spans(self.nodes)

        self._out_order, self._out_offsets = _csr(self.src_index, self.num_nodes)
        self._in_order, self._in_offsets = _csr(self.dst_index, self.num_nodes)

    # ------------------------------------------------------------- lookups ---

    def position_of(self, node_id: str) -> int:
        """Return a node's integer index.

        Args:
            node_id: Canonical node identifier.

        Returns:
            The node's position in the node table.

        Raises:
            CaseExtractionError: If the identifier is not in the graph.
        """
        try:
            return self._position[node_id]
        except KeyError as exc:
            raise CaseExtractionError(
                f"seed node {node_id!r} is not in {self.graph.dataset}"
            ) from exc

    def node_id_at(self, position: int) -> str:
        """Return the identifier of a node index.

        Args:
            position: Index into the node table.

        Returns:
            The canonical node identifier.
        """
        return str(self._node_ids[position])

    def incident_edges(self, position: int) -> np.ndarray:
        """Return every edge index incident to a node, in either direction.

        Args:
            position: Node index.

        Returns:
            Edge indices, ascending. Self-loops appear twice and are de-duplicated by the
            caller.
        """
        out = self._out_order[self._out_offsets[position] : self._out_offsets[position + 1]]
        inc = self._in_order[self._in_offsets[position] : self._in_offsets[position + 1]]
        return np.union1d(out, inc)

    def window_bounds(self, window: TimeWindow | None) -> tuple[int, int] | None:
        """Return the microsecond bounds used to filter edges by time.

        Args:
            window: The window, or None to accept every edge.

        Returns:
            ``(lo, hi)`` in microseconds, or None when no filtering applies. None is also
            returned when the graph has no ``timestamp`` column — as for Elliptic2, whose
            availability mask denies absolute timestamps entirely, so a window is
            meaningless rather than merely unset.
        """
        if window is None or self.timestamps is None:
            return None
        return to_micros(window.start), to_micros(window.end)

    def restrict_to_window(
        self, edge_indices: np.ndarray, bounds: tuple[int, int] | None
    ) -> np.ndarray:
        """Drop edge indices falling outside a window.

        Applied to the small per-node incidence lists rather than to a mask over all five
        million edges, which would otherwise dominate the cost of a case.

        Args:
            edge_indices: Candidate edge indices.
            bounds: Output of :meth:`window_bounds`.

        Returns:
            The subset lying inside the window, order preserved.
        """
        if bounds is None or self.timestamps is None:
            return edge_indices
        stamps = self.timestamps[edge_indices]
        return edge_indices[(stamps >= bounds[0]) & (stamps <= bounds[1])]

    def other_endpoint(self, edge_indices: np.ndarray, position: int) -> np.ndarray:
        """Return the far endpoint of each edge relative to a node.

        Args:
            edge_indices: Edge indices, all incident to ``position``.
            position: The near node index.

        Returns:
            The far endpoint of each edge. A self-loop yields ``position`` itself.
        """
        src = self.src_index[edge_indices]
        dst = self.dst_index[edge_indices]
        return np.where(src == position, dst, src)


def to_micros(moment: datetime) -> int:
    """Convert a naive datetime to microseconds since the epoch, without a timezone.

    ``datetime.timestamp()`` cannot be used here. It interprets a naive datetime as *local
    time*, while Polars stores ``Datetime("us")`` as timezone-naive microseconds. Mixing
    the two shifts every case window by the machine's UTC offset, which silently admits
    and excludes the wrong transactions and reproduces differently on a differently
    configured machine. AMLworld timestamps are local wall-clock with no zone
    (``TIMESTAMP_FORMAT`` in the loader), so the correct reading is the literal one.

    Args:
        moment: A naive datetime.

    Returns:
        Microseconds since 1970-01-01T00:00:00, reading the datetime literally.
    """
    return int(np.datetime64(moment.replace(tzinfo=None), "us").astype(np.int64))


def from_micros(micros: int | float) -> datetime:
    """Invert :func:`to_micros`.

    Args:
        micros: Microseconds since the epoch.

    Returns:
        The naive datetime it denotes.
    """
    return np.datetime64(int(micros), "us").astype(datetime)


def _csr(keys: np.ndarray, num_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a CSR ordering and offset array for an edge-endpoint column.

    Args:
        keys: Node index per edge.
        num_nodes: Total node count, so isolated nodes get empty slices.

    Returns:
        ``(order, offsets)`` where ``order`` lists edge indices grouped by key and
        ``offsets`` has length ``num_nodes + 1``.
    """
    order = np.argsort(keys, kind="stable")
    counts = np.bincount(keys, minlength=num_nodes)
    offsets = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return order, offsets


def _activity_spans(nodes: pl.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return per-node first/last activity in microseconds, when the table carries them.

    Cached on the index because negative sampling tests thousands of candidate windows
    against them, and re-casting a 515,088-row column each time would dominate the build.

    Args:
        nodes: A node table.

    Returns:
        ``(first_seen, last_seen)`` as int64 arrays, or ``(None, None)`` when the columns
        are absent — as they are for any substrate without absolute timestamps.
    """
    if "first_seen" not in nodes.columns or "last_seen" not in nodes.columns:
        return None, None
    return (
        nodes["first_seen"].cast(pl.Int64).to_numpy().astype(np.int64, copy=False),
        nodes["last_seen"].cast(pl.Int64).to_numpy().astype(np.int64, copy=False),
    )


def _laundering_mask(edges: pl.DataFrame) -> np.ndarray:
    """Return a boolean mask marking edges on a labelled-laundering path.

    An edge qualifies if ``is_laundering`` is true or ``pattern_id`` is non-null. The two
    are not the same set: HI-Small flags 5,177 transactions as laundering but places only
    3,209 of them inside a named pattern stream, and the 1,968 ``unclassified`` remainder
    is exactly as much evidence as the rest.

    Args:
        edges: An edge table.

    Returns:
        A boolean array of length ``edges.height``. All-False when the table carries
        neither column.
    """
    mask = np.zeros(edges.height, dtype=bool)
    if "is_laundering" in edges.columns:
        mask |= edges["is_laundering"].fill_null(False).to_numpy().astype(bool, copy=False)
    if "pattern_id" in edges.columns:
        mask |= edges["pattern_id"].is_not_null().to_numpy().astype(bool, copy=False)
    return mask


def canonical_sort(
    graph_nodes: pl.DataFrame, graph_edges: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Impose the canonical row order a case is serialised in.

    Determinism is a gate criterion, and Parquet preserves row order, so two extractions
    of the same case must agree on ordering or their serialisations differ. Nodes sort by
    ``node_id``; edges sort by whichever of :data:`_EDGE_SORT_PRIORITY` are present.

    Args:
        graph_nodes: The case's node table.
        graph_edges: The case's edge table.

    Returns:
        The two tables, canonically ordered.
    """
    keys = [c for c in _EDGE_SORT_PRIORITY if c in graph_edges.columns]
    return graph_nodes.sort("node_id"), graph_edges.sort(keys, maintain_order=True)


def _rank_order(
    index: GraphIndex,
    edge_indices: np.ndarray,
    rule: str,
    *,
    case_degree: np.ndarray | None = None,
) -> np.ndarray:
    """Order edge indices by a prune rule, most-retained first.

    Every rule is made total by falling back to descending amount and then to ascending
    edge index, so no two runs can disagree.

    Args:
        index: The graph index.
        edge_indices: Edges to order.
        rule: One of :data:`PRUNE_RULES`.
        case_degree: Per-edge structural weight used by the ``degree`` rule — the summed
            in-case degree of the edge's endpoints. Required for that rule only.

    Returns:
        ``edge_indices`` permuted into retention order.

    Raises:
        ValueError: If ``rule`` is unknown, or ``degree`` is requested without weights.
    """
    if rule == "amount_desc":
        primary = -index.amounts[edge_indices]
    elif rule == "recency":
        if index.timestamps is None:
            primary = np.zeros(edge_indices.size, dtype=np.float64)
        else:
            primary = -index.timestamps[edge_indices].astype(np.float64)
    elif rule == "degree":
        if case_degree is None:
            raise ValueError("the 'degree' prune rule needs per-edge endpoint degrees")
        primary = -case_degree.astype(np.float64)
    else:
        raise ValueError(f"unknown prune_rule {rule!r}; expected {PRUNE_RULES}")

    # np.lexsort applies the last key first, so this reads bottom-up: edge index breaks
    # ties in amount, which breaks ties in the rule itself.
    order = np.lexsort((edge_indices, -index.amounts[edge_indices], primary))
    return edge_indices[order]


def _expand(
    index: GraphIndex,
    seed_position: int,
    bounds: tuple[int, int] | None,
    params: ExtractionParams,
) -> tuple[set[int], set[int], dict[int, int], bool]:
    """Run the capped k-hop expansion from a seed.

    Args:
        index: The graph index.
        seed_position: Node index of the seed account.
        bounds: Microsecond window bounds from :meth:`GraphIndex.window_bounds`.
        params: Extraction parameters.

    Returns:
        ``(nodes, edges, hop_of_node, cap_triggered)`` — the reached node indices, the
        edges traversed to reach them, each node's hop distance from the seed, and whether
        the per-node neighbour cap bound anywhere.
    """
    nodes: set[int] = {seed_position}
    edges: set[int] = set()
    hop: dict[int, int] = {seed_position: 0}
    frontier: list[int] = [seed_position]
    cap_triggered = False

    for depth in range(params.k_hops + 1):
        # Hop 0 collects the seed's own incident transactions (protocol step 1); the
        # remaining k iterations are the expansion (step 2).
        if not frontier:
            break
        next_frontier: list[int] = []
        for position in sorted(frontier):
            incident = index.restrict_to_window(index.incident_edges(position), bounds)
            if incident.size == 0:
                continue
            if incident.size > params.max_neighbours_per_node:
                cap_triggered = True
                ranked = _rank_order(index, incident, "amount_desc")
                if params.preserve_laundering_paths:
                    flagged = index.laundering[ranked]
                    ranked = np.concatenate([ranked[flagged], ranked[~flagged]])
                incident = ranked[: params.max_neighbours_per_node]
            edges.update(int(e) for e in incident)
            if depth == params.k_hops:
                # Terminal hop: the traversed edges are kept, but their far endpoints are
                # only admitted if already inside the case, so the boundary stays closed.
                continue
            for far in index.other_endpoint(incident, position):
                far_int = int(far)
                if far_int not in hop:
                    hop[far_int] = depth + 1
                    nodes.add(far_int)
                    next_frontier.append(far_int)
        frontier = next_frontier

    return nodes, edges, hop, cap_triggered


def _induced_edges(
    index: GraphIndex,
    nodes: set[int],
    bounds: tuple[int, int] | None,
    max_per_node: int,
) -> np.ndarray:
    """Return the in-window edges whose endpoints both lie inside the node set.

    Expansion keeps only the edges it traversed; a case should also carry the edges
    *between* nodes it already admitted, since those are exactly the structure a typology
    is made of. The per-node cap is reapplied so a hub inside the case cannot reintroduce
    the blow-up the cap exists to prevent.

    Args:
        index: The graph index.
        nodes: Node indices in the case.
        bounds: Microsecond window bounds from :meth:`GraphIndex.window_bounds`.
        max_per_node: Per-node incidence cap.

    Returns:
        Edge indices, ascending.
    """
    collected: set[int] = set()
    member = np.zeros(index.num_nodes, dtype=bool)
    member[np.fromiter(nodes, dtype=np.int64, count=len(nodes))] = True
    for position in sorted(nodes):
        incident = index.restrict_to_window(index.incident_edges(position), bounds)
        if incident.size == 0:
            continue
        incident = incident[member[index.other_endpoint(incident, position)]]
        if incident.size > max_per_node:
            ranked = _rank_order(index, incident, "amount_desc")
            flagged = index.laundering[ranked]
            incident = np.concatenate([ranked[flagged], ranked[~flagged]])[:max_per_node]
        collected.update(int(e) for e in incident)
    return np.sort(np.fromiter(collected, dtype=np.int64, count=len(collected)))


def _prune(
    index: GraphIndex,
    seed_position: int,
    edge_indices: np.ndarray,
    params: ExtractionParams,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Spend the node budget, preserving laundering paths first.

    Protected edges are taken whole before the budget opens, so a case never loses the
    evidence it exists to describe. Once they are in, remaining edges are admitted in
    prune-rule order for as long as their new endpoints fit; an edge that introduces no
    new node is always free and is always taken.

    Args:
        index: The graph index.
        seed_position: The seed node, which is always retained.
        edge_indices: Candidate edges.
        params: Extraction parameters.

    Returns:
        ``(kept_edges, kept_nodes, n_max_exceeded)``. ``n_max_exceeded`` is True when
        preservation alone overran the budget.

    Raises:
        ValueError: If ``params.prune_rule`` is unknown.
    """
    protected = (
        edge_indices[index.laundering[edge_indices]]
        if params.preserve_laundering_paths
        else np.empty(0, dtype=np.int64)
    )
    kept: set[int] = {int(e) for e in protected}
    nodes: set[int] = {seed_position}
    nodes.update(int(n) for n in index.src_index[protected])
    nodes.update(int(n) for n in index.dst_index[protected])
    exceeded = len(nodes) > params.n_max

    remaining = np.setdiff1d(edge_indices, protected, assume_unique=False)
    case_degree = None
    if params.prune_rule == "degree":
        case_degree = (
            index.degree[index.src_index[remaining]] + index.degree[index.dst_index[remaining]]
        )
    for edge in _rank_order(index, remaining, params.prune_rule, case_degree=case_degree):
        endpoints = {int(index.src_index[edge]), int(index.dst_index[edge])}
        fresh = endpoints - nodes
        if not fresh:
            kept.add(int(edge))
        elif len(nodes) + len(fresh) <= params.n_max:
            kept.add(int(edge))
            nodes |= fresh

    kept_edges = np.sort(np.fromiter(kept, dtype=np.int64, count=len(kept)))
    kept_nodes = np.sort(np.fromiter(nodes, dtype=np.int64, count=len(nodes)))
    return kept_edges, kept_nodes, exceeded


def _dominant_typology(edges: pl.DataFrame) -> str | None:
    """Return the case's typology, or None when the substrate has no ground truth.

    Args:
        edges: The case's edge table.

    Returns:
        The most frequent non-null typology among the case's laundering edges, ties broken
        by :data:`TYPOLOGY_VOCABULARY` order so the result never depends on row order.
        None when the column is absent or no edge carries one.
    """
    if "typology" not in edges.columns:
        return None
    present = edges.filter(pl.col("typology").is_not_null())
    if present.is_empty():
        return None
    counts = present.group_by("typology").len().to_dicts()
    order = {name: i for i, name in enumerate(TYPOLOGY_VOCABULARY)}
    best = min(counts, key=lambda row: (-int(row["len"]), order.get(str(row["typology"]), 99)))
    return str(best["typology"])


@dataclass(frozen=True)
class CaseCut:
    """A case expressed as positions into the substrate graph, before materialisation.

    Sampling builds tens of thousands of candidate cases and keeps them by reference, so
    it never needs the attribute columns; motif scoring needs two of them. Cutting and
    materialising are therefore separate steps, and :func:`extract_case` is the one that
    does both.

    Attributes:
        case_id: The deterministic identifier.
        node_positions: Row positions into the graph's node table, ascending.
        edge_positions: Row positions into the graph's edge table, ascending.
        label: ``"suspicious"``, ``"licit"``, or None when the substrate has no labels.
        provenance: The full extraction record.
    """

    case_id: str
    node_positions: np.ndarray
    edge_positions: np.ndarray
    label: str | None
    provenance: dict[str, Any]


def cut_case(
    graph: CanonicalGraph,
    seed_node: str,
    window: TimeWindow,
    params: ExtractionParams,
    *,
    index: GraphIndex | None = None,
) -> CaseCut:
    """Run the extraction protocol and return positions rather than tables.

    Args:
        graph: The substrate graph to cut the case out of.
        seed_node: Canonical identifier of the seed account.
        window: The case's time window.
        params: The extraction parameterisation.
        index: A prebuilt index over ``graph``. Built on the fly when omitted, which is
            only sensible for a single case.

    Returns:
        The cut, carrying node and edge positions and the full provenance record.

    Raises:
        CaseExtractionError: If ``seed_node`` is unknown, or has no transaction inside
            ``window`` — an empty case is a sampling bug, not a result.
    """
    idx = index if index is not None else GraphIndex(graph)
    seed_position = idx.position_of(seed_node)
    bounds = idx.window_bounds(window)

    reached, _traversed, hops, cap_triggered = _expand(idx, seed_position, bounds, params)
    candidate_edges = _induced_edges(idx, reached, bounds, params.max_neighbours_per_node)
    if candidate_edges.size == 0:
        raise CaseExtractionError(
            f"seed {seed_node!r} has no transaction in {window.start}..{window.end}; "
            "an empty case is a sampling error, not a case"
        )

    pre_prune_nodes = len(reached)
    pruning_triggered = pre_prune_nodes > params.n_max
    if pruning_triggered:
        kept_edges, kept_nodes, exceeded = _prune(idx, seed_position, candidate_edges, params)
    else:
        kept_edges = candidate_edges
        kept_nodes = np.sort(np.fromiter(reached, dtype=np.int64, count=len(reached)))
        exceeded = False

    flagged = int(idx.laundering[kept_edges].sum())
    provenance = {
        "extraction_method": "constructed",
        "extraction_protocol_version": EXTRACTION_PROTOCOL_VERSION,
        "source_dataset": graph.dataset,
        "seed_node": seed_node,
        "window": window.to_dict(),
        **params.to_dict(),
        "pre_prune_nodes": pre_prune_nodes,
        "pre_prune_edges": int(candidate_edges.size),
        "post_prune_nodes": int(kept_nodes.size),
        "post_prune_edges": int(kept_edges.size),
        "pruning_triggered": pruning_triggered,
        "n_max_exceeded": exceeded,
        "neighbour_cap_triggered": cap_triggered,
        "preserved_laundering_edges": flagged,
        "max_hop_reached": max(hops[int(p)] for p in kept_nodes if int(p) in hops),
    }
    label = None
    if "is_laundering" in idx.edges.columns:
        label = "suspicious" if flagged else "licit"
    return CaseCut(
        case_id=case_id_for(graph.dataset, seed_node, window, params),
        node_positions=kept_nodes,
        edge_positions=kept_edges,
        label=label,
        provenance=provenance,
    )


def materialise_cut(cut: CaseCut, index: GraphIndex) -> CanonicalGraph:
    """Turn a :class:`CaseCut` into a full canonical graph.

    Node index *i* is row *i* of the node table by construction, so both tables come out
    of a positional gather over contiguous frames rather than a predicate scan.

    Args:
        cut: The positions to materialise.
        index: The index the cut was taken against.

    Returns:
        The case, canonically ordered and carrying the substrate's availability mask.
    """
    nodes, edges = canonical_sort(
        index.nodes[cut.node_positions.tolist()], index.edges[cut.edge_positions.tolist()]
    )
    return CanonicalGraph(
        graph_id=cut.case_id,
        dataset=index.graph.dataset,
        nodes=nodes,
        edges=edges,
        node_feature_names=list(index.graph.node_feature_names),
        edge_feature_names=list(index.graph.edge_feature_names),
        availability=index.graph.availability,
        label=cut.label,
        typology=_dominant_typology(edges),
        provenance=dict(cut.provenance),
    )


def extract_case(
    graph: CanonicalGraph,
    seed_node: str,
    window: TimeWindow,
    k_hops: int = 2,
    n_max: int = 150,
    prune_rule: Literal["amount_desc", "recency", "degree"] = "amount_desc",
    preserve_laundering_paths: bool = True,
    seed: int = 1337,
    *,
    index: GraphIndex | None = None,
    max_neighbours_per_node: int = DEFAULT_MAX_NEIGHBOURS,
) -> CanonicalGraph:
    """Construct one case around a seed account, with full provenance.

    Implements the four-step protocol in the module docstring. The returned graph is
    canonically ordered, so two extractions with the same inputs serialise byte-identically.

    Args:
        graph: The substrate graph to cut the case out of.
        seed_node: Canonical identifier of the seed account.
        window: The case's time window. Transactions outside it are never admitted.
        k_hops: Expansion radius in undirected transaction hops.
        n_max: Node budget.
        prune_rule: How to spend the budget. See :data:`PRUNE_RULES`.
        preserve_laundering_paths: Retain every labelled-laundering edge regardless of the
            budget. Turning this off is a sensitivity-analysis setting, not a normal one.
        seed: Recorded in the provenance and folded into the case id. Extraction itself
            draws no random numbers.
        index: A prebuilt :class:`GraphIndex`. Building one over HI-Small costs several
            seconds, so any caller extracting more than one case should build it once and
            pass it in.
        max_neighbours_per_node: Per-node expansion cap. See the module docstring.

    Returns:
        A :class:`CanonicalGraph` holding the case, carrying the substrate's availability
        mask, its derived ``label`` and ``typology``, and a ``provenance`` record with
        every parameter, the pre- and post-prune sizes, and whether pruning, the neighbour
        cap or budget overrun were triggered.

    Raises:
        CaseExtractionError: If ``seed_node`` is not in the graph, or the seed has no
            transaction at all inside ``window``.
        ValueError: If any parameter is out of range or ``prune_rule`` is unknown.
    """
    params = ExtractionParams(
        k_hops=k_hops,
        n_max=n_max,
        prune_rule=prune_rule,
        preserve_laundering_paths=preserve_laundering_paths,
        seed=seed,
        max_neighbours_per_node=max_neighbours_per_node,
    )
    idx = index if index is not None else GraphIndex(graph)
    return materialise_cut(cut_case(graph, seed_node, window, params, index=idx), idx)


def passthrough_case(graph: CanonicalGraph, *, case_id: str | None = None) -> CanonicalGraph:
    """Record a substrate-provided subgraph as a case, constructing nothing.

    Elliptic2 ships 122K labelled subgraphs, and a labelled subgraph *is* the case. There
    is no boundary to choose, so there is no construction rule to defend — which is
    precisely why the two substrates are worth having together. The provenance says
    ``extraction_method="provided"`` so a downstream consumer can tell the two apart
    without inspecting parameters that do not exist.

    Args:
        graph: The provided subgraph.
        case_id: Identifier to use. Defaults to the graph's own ``graph_id``.

    Returns:
        The same graph with its provenance extended. Node and edge tables are canonically
        sorted, so a provided case serialises as reproducibly as a constructed one.
    """
    nodes, edges = canonical_sort(graph.nodes, graph.edges)
    provenance = dict(graph.provenance) | {
        "extraction_method": "provided",
        "extraction_protocol_version": EXTRACTION_PROTOCOL_VERSION,
        "source_dataset": graph.dataset,
        "pre_prune_nodes": graph.num_nodes,
        "pre_prune_edges": graph.num_edges,
        "post_prune_nodes": graph.num_nodes,
        "post_prune_edges": graph.num_edges,
        "pruning_triggered": False,
        "n_max_exceeded": False,
        "neighbour_cap_triggered": False,
    }
    return CanonicalGraph(
        graph_id=case_id or graph.graph_id,
        dataset=graph.dataset,
        nodes=nodes,
        edges=edges,
        node_feature_names=list(graph.node_feature_names),
        edge_feature_names=list(graph.edge_feature_names),
        availability=graph.availability,
        label=graph.label,
        typology=graph.typology,
        provenance=provenance,
    )
