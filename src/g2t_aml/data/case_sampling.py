"""Case sampling: which cases get built, and why those.

Three populations, and the third is the one the paper turns on.

**Positive (suspicious).** Seeded from accounts participating in a labelled laundering
stream, windowed on the stream's temporal extent plus padding, carrying the stream's
typology. Stratified evenly across the eight typologies *and* ``unclassified``, because
HI-Small's 1,968 unclassified laundering transactions are a third of all flagged activity
and a narrative must be able to say "flagged, matching no known pattern". Even allocation
is deliberate: proportional sampling would let gather-scatter (716 transactions) crowd out
random (191) for no reason other than the generator's own frequency choices.

**Negative (licit).** Seeded from accounts that appear in no laundering transaction at
all, matched to the positive population on *activity level* and on *window*. Both matches
matter. Without activity matching a classifier wins on transaction count alone; without
window matching it wins on the calendar. Each negative copies a real positive's window
verbatim and draws a licit seed from the same log-degree bucket, so neither channel
carries signal.

**Hard negatives.** Licit cases whose structure mimics a typology: legitimate payroll is a
fan-out, legitimate collections are a fan-in, supplier settlement is a chain. The AMLworld
authors note explicitly that fan-in and fan-out appear in both normal and alert categories
because criminals mimic legitimate activity. These are mined by scoring a large licit
candidate pool with :mod:`g2t_aml.data.motifs` — which never sees a label — and taking the
top scorers. They are required to be at least
:data:`MINIMUM_HARD_NEGATIVE_RATE` of the negative population.

Two storage decisions worth knowing before reading the code.

**Cases are stored by reference, not by copy.** A case averages several hundred edges and
there are 15,000 of them, so materialising every attribute would write a case corpus
several times larger than the 172 MB substrate it was cut from. ``case_nodes.parquet`` and
``case_edges.parquet`` instead hold positions into the interim graph, and
:meth:`CaseCollection.materialise` gathers them back into a full
:class:`~g2t_aml.data.canonical.CanonicalGraph` in under a millisecond. Positions are only
meaningful against the exact graph they were cut from, so the collection records that
graph's manifest hash and refuses to materialise against a different one.

**Sampling is the only place randomness lives.** Extraction is deterministic (see
:mod:`g2t_aml.data.case_extraction`); everything stochastic here goes through a single
seeded :class:`numpy.random.Generator`, and the seed is recorded in the manifest.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from g2t_aml.data.canonical import TYPOLOGY_VOCABULARY, AvailabilityMask, CanonicalGraph
from g2t_aml.data.case_extraction import (
    CaseCut,
    CaseExtractionError,
    ExtractionParams,
    GraphIndex,
    TimeWindow,
    cut_case,
    to_micros,
)
from g2t_aml.data.motifs import score_edges
from g2t_aml.utils.hashing import canonical_json, hash_id_list, short
from g2t_aml.utils.io import read_json, read_jsonl, write_json, write_jsonl

#: Bumping this invalidates a sampled case corpus.
SAMPLING_SCHEMA_VERSION = "1.0.0"

#: The three populations a case can belong to. ``hard_negative`` is a *subset* of the
#: licit population, not a third label: its ``label`` is still ``"licit"``.
CASE_CLASSES: tuple[str, ...] = ("suspicious", "licit", "hard_negative")

#: The gate criterion: hard negatives must be at least this share of all negatives.
MINIMUM_HARD_NEGATIVE_RATE = 0.20

#: Strata the positive population is balanced across: the eight structural typologies plus
#: ``unclassified``.
POSITIVE_STRATA: tuple[str, ...] = TYPOLOGY_VOCABULARY

#: Number of log-spaced buckets used to match negatives to positives on activity level.
ACTIVITY_BUCKETS = 10


class CaseSamplingError(RuntimeError):
    """Raised when a case population cannot be sampled to the requested shape."""


@dataclass(frozen=True)
class SamplingParams:
    """How many cases of what kind, and how they are matched.

    Attributes:
        n_cases: Total cases to build across all three populations.
        positive_fraction: Share of ``n_cases`` that should be suspicious. Capped by how
            many laundering-participating accounts the substrate actually has — HI-Small
            has 6,357, which is why this cannot be 0.5.
        hard_negative_fraction: Share of the *negative* population that must be hard
            negatives. Must be at least :data:`MINIMUM_HARD_NEGATIVE_RATE`.
        hard_negative_oversample: Licit candidates extracted and motif-scored per hard
            negative retained. Larger means better-scoring hard negatives and a slower
            build. Measured on HI-Small at a 48-hour window: 77% of proposed licit seeds
            yield a licit case and 14.2% of those clear a 0.5 motif score, so roughly 11%
            of seeds become a usable hard negative and the pool has to be at least 2.3x
            the negative population for a 25% share to be reachable.
        hard_negative_min_score: Motif score below which a candidate is not admitted as a
            hard negative however short the population falls. A "hard" negative that
            mimics nothing is just a negative, and silently relabelling one would make the
            headline result meaningless.
        window_pad_hours: Padding applied on each side of a stream's temporal extent.
        max_window_hours: Ceiling on a case's window duration. A stream whose padded
            extent exceeds it gets a review window of exactly this length, centred on the
            stream's median transaction time. This exists because case *duration* is what
            makes a temporal split possible: HI-Small's laundering streams run to 202
            hours against a substrate span of 17.7 days, and windows that wide straddle
            every candidate boundary, leaving the val split empty. The cost is that a long
            stream is only partly covered, which is measured and reported as typology
            recovery. Set to None to take the full extent and accept the consequence.
        max_seeds_per_stream: Cap on positive seeds drawn from a single stream, so one
            32-transaction stream cannot supply a whole stratum.
        max_stratum_share: Ceiling on any one stratum's share of the positive population.
            Even allocation alone is not enough here: HI-Small offers 3,227 unclassified
            seeds against 232-448 per structural typology, so water-filling hands
            ``unclassified`` every seat the other eight cannot fill and it ends up at 46%
            of positives. The eight typologies are the axis the paper is about, so a
            single stratum is not allowed past this share.
        seed: The single RNG seed for all sampling.
    """

    n_cases: int = 15_000
    positive_fraction: float = 0.33
    hard_negative_fraction: float = 0.30
    hard_negative_oversample: float = 8.0
    hard_negative_min_score: float = 0.5
    window_pad_hours: float = 12.0
    max_window_hours: float | None = 96.0
    max_seeds_per_stream: int = 12
    max_stratum_share: float = 0.35
    seed: int = 1337

    def __post_init__(self) -> None:
        """Validate the sampling plan.

        Raises:
            ValueError: If a count or fraction is out of range, or
                ``hard_negative_fraction`` falls below the gate criterion.
        """
        if self.n_cases < 1:
            raise ValueError(f"n_cases must be >= 1, got {self.n_cases}")
        if not 0.0 < self.positive_fraction < 1.0:
            raise ValueError(f"positive_fraction must be in (0, 1), got {self.positive_fraction}")
        if self.hard_negative_fraction < MINIMUM_HARD_NEGATIVE_RATE:
            raise ValueError(
                f"hard_negative_fraction {self.hard_negative_fraction} is below the "
                f"{MINIMUM_HARD_NEGATIVE_RATE:.0%} gate criterion"
            )
        if self.hard_negative_oversample < 1.0:
            raise ValueError("hard_negative_oversample must be >= 1")
        if not 0.0 <= self.hard_negative_min_score <= 1.0:
            raise ValueError("hard_negative_min_score must be in [0, 1]")
        if self.max_seeds_per_stream < 1:
            raise ValueError("max_seeds_per_stream must be >= 1")
        if self.max_window_hours is not None and self.max_window_hours <= 0:
            raise ValueError("max_window_hours must be positive or None")
        if not 1.0 / len(POSITIVE_STRATA) <= self.max_stratum_share <= 1.0:
            raise ValueError(
                f"max_stratum_share must be between {1 / len(POSITIVE_STRATA):.3f} "
                f"(perfectly even) and 1.0, got {self.max_stratum_share}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the parameters as a plain dict.

        Returns:
            Field name to value.
        """
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class CaseRecord:
    """The index entry for one case: everything splitting and auditing need.

    Deliberately does *not* carry the case's node list. Node identity lives in
    ``case_nodes.parquet``, which the leakage auditor reads directly; duplicating it here
    would triple the index for no gain.

    Attributes:
        case_id: Deterministic identifier from
            :func:`~g2t_aml.data.case_extraction.case_id_for`.
        dataset: Substrate key.
        seed_node: The seed account.
        window_start: Inclusive window lower bound.
        window_end: Inclusive window upper bound.
        case_class: One of :data:`CASE_CLASSES`.
        label: ``"suspicious"`` or ``"licit"``. A hard negative is licit.
        typology: Stream typology carried onto a positive case, else None.
        pattern_ids: Every laundering stream the case touches. Sorted, so the split can
            keep a stream atomic.
        n_nodes: Node count after pruning.
        n_edges: Edge count after pruning.
        activity_bucket: Log-degree bucket of the seed, used for negative matching.
        motif_best: Highest-scoring structural motif, or None.
        motif_score: That motif's score.
        structural_hash: Digest of the sorted node list, for duplicate detection.
        provenance: The extraction provenance record, verbatim.
    """

    case_id: str
    dataset: str
    seed_node: str
    window_start: datetime
    window_end: datetime
    case_class: str
    label: str
    typology: str | None
    pattern_ids: tuple[str, ...]
    n_nodes: int
    n_edges: int
    activity_bucket: int
    motif_best: str | None
    motif_score: float
    structural_hash: str
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def window(self) -> TimeWindow:
        """Return the case's time window.

        Returns:
            The window as a :class:`~g2t_aml.data.case_extraction.TimeWindow`.
        """
        return TimeWindow(start=self.window_start, end=self.window_end)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record.

        Returns:
            The record with datetimes as ISO-8601 strings and ``pattern_ids`` as a list.
        """
        data = dataclasses.asdict(self)
        data["window_start"] = self.window_start.isoformat()
        data["window_end"] = self.window_end.isoformat()
        data["pattern_ids"] = list(self.pattern_ids)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaseRecord:
        """Rebuild a record written by :meth:`to_dict`.

        Args:
            data: The serialised record.

        Returns:
            The reconstructed record.

        Raises:
            KeyError: If a required field is missing.
            ValueError: If a timestamp is not parseable.
        """
        payload = dict(data)
        payload["window_start"] = datetime.fromisoformat(str(payload["window_start"]))
        payload["window_end"] = datetime.fromisoformat(str(payload["window_end"]))
        payload["pattern_ids"] = tuple(payload.get("pattern_ids") or ())
        return cls(**payload)


@dataclass
class CaseCollection:
    """A built case population: the index, plus node and edge membership by reference.

    Attributes:
        dataset: Substrate key.
        records: One :class:`CaseRecord` per case, in build order.
        node_membership: ``(case_id, node_index, node_id)``.
        edge_membership: ``(case_id, edge_index)``.
        source_manifest_hash: Digest identifying the interim graph the positions index
            into. :meth:`materialise` refuses to run against a different graph.
        extraction_params: The protocol used.
        sampling_params: The plan used.
        stratification: Observed counts by class, label and typology.
    """

    dataset: str
    records: list[CaseRecord]
    node_membership: pl.DataFrame
    edge_membership: pl.DataFrame
    source_manifest_hash: str
    extraction_params: dict[str, Any] = field(default_factory=dict)
    sampling_params: dict[str, Any] = field(default_factory=dict)
    stratification: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return the number of cases.

        Returns:
            Case count.
        """
        return len(self.records)

    @property
    def case_ids(self) -> list[str]:
        """Return every case id, in build order.

        Returns:
            The identifiers.
        """
        return [r.case_id for r in self.records]

    def by_id(self) -> dict[str, CaseRecord]:
        """Index the records by identifier.

        Returns:
            Case id to record.
        """
        return {r.case_id: r for r in self.records}

    def subset(self, case_ids: list[str]) -> CaseCollection:
        """Return a collection restricted to the given cases.

        Args:
            case_ids: Identifiers to keep. Unknown identifiers are ignored.

        Returns:
            A new collection sharing this one's parameters.
        """
        wanted = set(case_ids)
        return CaseCollection(
            dataset=self.dataset,
            records=[r for r in self.records if r.case_id in wanted],
            node_membership=self.node_membership.filter(pl.col("case_id").is_in(list(wanted))),
            edge_membership=self.edge_membership.filter(pl.col("case_id").is_in(list(wanted))),
            source_manifest_hash=self.source_manifest_hash,
            extraction_params=dict(self.extraction_params),
            sampling_params=dict(self.sampling_params),
            stratification=summarise_stratification(
                [r for r in self.records if r.case_id in wanted]
            ),
        )

    def materialise(self, case_id: str, index: GraphIndex) -> CanonicalGraph:
        """Rebuild one case as a full canonical graph.

        Args:
            case_id: The case to rebuild.
            index: An index over the *same* interim graph the cases were cut from.

        Returns:
            The case, with its recorded label, typology and extraction provenance.

        Raises:
            KeyError: If ``case_id`` is not in this collection.
            CaseSamplingError: If ``index`` is over a different graph than the one the
                positions were recorded against.
        """
        record = self.by_id()[case_id]
        if index.graph.dataset != self.dataset:
            raise CaseSamplingError(
                f"cases were cut from {self.dataset!r} but the index is over "
                f"{index.graph.dataset!r}; positions are not portable between graphs"
            )
        nodes = self.node_membership.filter(pl.col("case_id") == case_id)["node_index"].to_list()
        edges = self.edge_membership.filter(pl.col("case_id") == case_id)["edge_index"].to_list()
        return CanonicalGraph(
            graph_id=case_id,
            dataset=self.dataset,
            nodes=index.nodes[nodes],
            edges=index.edges[edges],
            node_feature_names=list(index.graph.node_feature_names),
            edge_feature_names=list(index.graph.edge_feature_names),
            availability=index.graph.availability,
            label=record.label,
            typology=record.typology,
            provenance=dict(record.provenance),
        )

    def save(self, directory: str | Path) -> Path:
        """Write the collection atomically.

        Args:
            directory: Destination, created if absent.

        Returns:
            The directory written to.

        Raises:
            OSError: If a write or rename fails.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        write_jsonl(out / "cases.jsonl", (r.to_dict() for r in self.records))
        self.node_membership.write_parquet(out / "case_nodes.parquet", compression="zstd")
        self.edge_membership.write_parquet(out / "case_edges.parquet", compression="zstd")
        write_json(
            out / "cases_manifest.json",
            {
                "sampling_schema_version": SAMPLING_SCHEMA_VERSION,
                "dataset": self.dataset,
                "n_cases": len(self.records),
                "source_manifest_hash": self.source_manifest_hash,
                "extraction_params": self.extraction_params,
                "sampling_params": self.sampling_params,
                "stratification": self.stratification,
                "case_id_list_sha256": hash_id_list(self.case_ids),
            },
            canonical=True,
        )
        return out

    @classmethod
    def load(cls, directory: str | Path) -> CaseCollection:
        """Read a collection written by :meth:`save`.

        Args:
            directory: Source directory.

        Returns:
            The reconstructed collection.

        Raises:
            FileNotFoundError: If any component file is absent.
            CaseSamplingError: If the manifest was written by an incompatible version.
        """
        src = Path(directory)
        manifest = read_json(src / "cases_manifest.json")
        if manifest.get("sampling_schema_version") != SAMPLING_SCHEMA_VERSION:
            raise CaseSamplingError(
                f"case corpus schema mismatch: file has "
                f"{manifest.get('sampling_schema_version')!r}, code expects "
                f"{SAMPLING_SCHEMA_VERSION!r}"
            )
        return cls(
            dataset=manifest["dataset"],
            records=[CaseRecord.from_dict(r) for r in read_jsonl(src / "cases.jsonl")],
            node_membership=pl.read_parquet(src / "case_nodes.parquet"),
            edge_membership=pl.read_parquet(src / "case_edges.parquet"),
            source_manifest_hash=manifest["source_manifest_hash"],
            extraction_params=dict(manifest.get("extraction_params") or {}),
            sampling_params=dict(manifest.get("sampling_params") or {}),
            stratification=dict(manifest.get("stratification") or {}),
        )


def summarise_stratification(records: list[CaseRecord]) -> dict[str, Any]:
    """Count a population by class, label and typology.

    Args:
        records: The cases to summarise.

    Returns:
        ``by_class``, ``by_label``, ``by_typology`` counts plus ``hard_negative_rate``, the
        share of the *negative* population that is hard. Rate is 0.0 when there are no
        negatives, which is a degenerate population rather than a division by zero.
    """
    by_class: dict[str, int] = defaultdict(int)
    by_label: dict[str, int] = defaultdict(int)
    by_typology: dict[str, int] = defaultdict(int)
    for record in records:
        by_class[record.case_class] += 1
        by_label[record.label] += 1
        by_typology[record.typology or "none"] += 1
    negatives = by_label.get("licit", 0)
    return {
        "by_class": dict(sorted(by_class.items())),
        "by_label": dict(sorted(by_label.items())),
        "by_typology": dict(sorted(by_typology.items())),
        "n_cases": len(records),
        "hard_negative_rate": (
            round(by_class.get("hard_negative", 0) / negatives, 6) if negatives else 0.0
        ),
    }


# --------------------------------------------------------------------- seeds ---


@dataclass(frozen=True)
class CaseSeed:
    """A decision to build one case: where to start, when to look, and what it is.

    Attributes:
        seed_node: The seed account.
        window: The case's time window.
        case_class: Intended population. A licit candidate is promoted from ``licit`` to
            ``hard_negative`` only after its motif score is known.
        typology: The typology carried from the seeding stream, or None.
        pattern_ids: Streams the seed participates in within the window.
    """

    seed_node: str
    window: TimeWindow
    case_class: str
    typology: str | None = None
    pattern_ids: tuple[str, ...] = ()


def _allocate_evenly(budget: int, capacities: dict[str, int]) -> dict[str, int]:
    """Split a budget as evenly as capacity allows, spilling shortfall onto the rest.

    Even allocation is what stops gather-scatter dominating the positive population purely
    because AMLworld generated more of it. Strata that cannot fill their equal share give
    the remainder back, and it is redistributed over the strata that still have room.

    Args:
        budget: Total to allocate.
        capacities: Stratum name to the most it can supply.

    Returns:
        Stratum name to allocation. The total is ``min(budget, sum(capacities))``.
    """
    allocation = {name: 0 for name in capacities}
    remaining = min(budget, sum(capacities.values()))
    open_strata = {n for n, cap in capacities.items() if cap > 0}
    while remaining > 0 and open_strata:
        share = max(1, remaining // len(open_strata))
        for name in sorted(open_strata):
            if remaining <= 0:
                break
            take = min(share, capacities[name] - allocation[name], remaining)
            allocation[name] += take
            remaining -= take
            if allocation[name] >= capacities[name]:
                open_strata.discard(name)
    return allocation


#: Public alias. Phase 6's Gold sampler needs exactly this allocation rule for the same
#: reason Phase 2 does — a typology stratum that cannot fill its equal share must give the
#: remainder back rather than shrinking the sample — and reimplementing it there would put
#: two subtly different "even" allocations in one repository.
allocate_evenly = _allocate_evenly


def bounded_window(
    start: datetime,
    end: datetime,
    centre: datetime,
    pad_hours: float,
    max_hours: float | None,
) -> TimeWindow:
    """Build a case window from an activity extent, capped at a maximum duration.

    Short activity gets its full extent plus padding. Activity too long to fit the cap
    gets a review window of exactly ``max_hours`` centred on ``centre`` — the median
    transaction time rather than the midpoint of the extent, because a stream's
    transactions are not uniformly spread and centring on the median keeps more of them.

    Args:
        start: First transaction in the activity.
        end: Last transaction in the activity.
        centre: Where to centre a capped window, normally the median transaction time.
        pad_hours: Padding applied on each side of the uncapped extent.
        max_hours: Duration ceiling, or None for no cap.

    Returns:
        The window.

    Raises:
        ValueError: If ``end`` precedes ``start`` or ``pad_hours`` is negative.
    """
    padded = TimeWindow(start=start, end=end).padded(pad_hours)
    if max_hours is None or padded.duration <= timedelta(hours=max_hours):
        return padded
    half = timedelta(hours=max_hours / 2)
    return TimeWindow(start=centre - half, end=centre + half)


def _stream_windows(
    graph: CanonicalGraph, pad_hours: float, max_hours: float | None
) -> dict[str, tuple[TimeWindow, str]]:
    """Compute each laundering stream's window and typology.

    Args:
        graph: The substrate graph.
        pad_hours: Padding applied on each side of the stream's temporal extent.
        max_hours: Duration ceiling. See :func:`bounded_window`.

    Returns:
        ``pattern_id`` to ``(window, typology)``. Empty when the graph carries no streams.
    """
    if "pattern_id" not in graph.edges.columns or "timestamp" not in graph.edges.columns:
        return {}
    grouped = (
        graph.edges.filter(pl.col("pattern_id").is_not_null())
        .group_by("pattern_id")
        .agg(
            pl.col("timestamp").min().alias("t0"),
            pl.col("timestamp").max().alias("t1"),
            pl.col("timestamp").median().alias("tm"),
            pl.col("typology").first().alias("typology"),
        )
        .sort("pattern_id")
    )
    return {
        str(row["pattern_id"]): (
            bounded_window(row["t0"], row["t1"], row["tm"], pad_hours, max_hours),
            str(row["typology"]),
        )
        for row in grouped.to_dicts()
    }


def _activity_buckets(index: GraphIndex) -> np.ndarray:
    """Assign every node a log-degree activity bucket.

    Degree spans 1 to 169,756 in HI-Small, so linear bucketing would put 99% of accounts in
    one bucket and make activity matching vacuous. Log spacing gives buckets that actually
    separate a two-transaction account from a two-thousand-transaction one.

    Args:
        index: The graph index.

    Returns:
        A bucket in ``[0, ACTIVITY_BUCKETS)`` per node.
    """
    cached = getattr(index, "activity_buckets", None)
    if cached is not None:
        return cached
    logged = np.log1p(index.degree.astype(np.float64))
    top = float(logged.max()) or 1.0
    buckets = np.clip(
        np.floor(logged / top * ACTIVITY_BUCKETS).astype(np.int64), 0, ACTIVITY_BUCKETS - 1
    )
    # Memoised on the index: without this it is recomputed once per candidate case, and
    # the array is 515,088 elements wide.
    index.activity_buckets = buckets
    return buckets


def positive_seeds(
    graph: CanonicalGraph,
    params: SamplingParams,
    rng: np.random.Generator,
) -> list[CaseSeed]:
    """Enumerate seeds for the suspicious population, stratified across typologies.

    Patterned streams contribute their participating accounts, windowed on the stream.
    Laundering-flagged transactions belonging to no stream contribute their accounts under
    the ``unclassified`` stratum, windowed on the account's own flagged activity — they are
    a third of HI-Small's flagged transactions and dropping them would bias the corpus
    toward exactly the neat structures the generator finds easiest.

    Args:
        graph: The substrate graph.
        params: The sampling plan.
        rng: Seeded generator; the only source of randomness.

    Returns:
        Seeds, deduplicated on ``(seed_node, window)`` and shuffled deterministically.

    Raises:
        CaseSamplingError: If the graph carries no laundering labels at all, which means
            the caller pointed positive sampling at a substrate that cannot support it.
    """
    if "is_laundering" not in graph.edges.columns:
        raise CaseSamplingError(
            f"{graph.dataset} carries no is_laundering column; positive sampling needs "
            "laundering ground truth"
        )
    windows = _stream_windows(graph, params.window_pad_hours, params.max_window_hours)
    by_stratum: dict[str, list[CaseSeed]] = {name: [] for name in POSITIVE_STRATA}

    # Patterned streams, one stratum per typology.
    if windows:
        streams = graph.edges.filter(pl.col("pattern_id").is_not_null()).select(
            "pattern_id", "src", "dst"
        )
        participants: dict[str, list[str]] = defaultdict(list)
        for row in streams.iter_rows(named=True):
            participants[str(row["pattern_id"])].extend((str(row["src"]), str(row["dst"])))
        for pattern_id in sorted(participants):
            window, typology = windows[pattern_id]
            accounts = sorted(set(participants[pattern_id]))
            if len(accounts) > params.max_seeds_per_stream:
                chosen = rng.choice(len(accounts), size=params.max_seeds_per_stream, replace=False)
                accounts = [accounts[i] for i in sorted(chosen)]
            by_stratum[typology].extend(
                CaseSeed(
                    seed_node=account,
                    window=window,
                    case_class="suspicious",
                    typology=typology,
                    pattern_ids=(pattern_id,),
                )
                for account in accounts
            )

    # Flagged-but-unpatterned transactions: the `unclassified` stratum.
    unpatterned = graph.edges.filter(pl.col("is_laundering"))
    if "pattern_id" in unpatterned.columns:
        unpatterned = unpatterned.filter(pl.col("pattern_id").is_null())
    if unpatterned.height and "timestamp" in unpatterned.columns:
        endpoints = pl.concat(
            [
                unpatterned.select(pl.col("src").alias("account"), "timestamp"),
                unpatterned.select(pl.col("dst").alias("account"), "timestamp"),
            ]
        )
        spans = (
            endpoints.group_by("account")
            .agg(
                pl.col("timestamp").min().alias("t0"),
                pl.col("timestamp").max().alias("t1"),
                pl.col("timestamp").median().alias("tm"),
            )
            .sort("account")
        )
        by_stratum["unclassified"].extend(
            CaseSeed(
                seed_node=str(row["account"]),
                window=bounded_window(
                    row["t0"],
                    row["t1"],
                    row["tm"],
                    params.window_pad_hours,
                    params.max_window_hours,
                ),
                case_class="suspicious",
                typology="unclassified",
                pattern_ids=(),
            )
            for row in spans.to_dicts()
        )

    seeds: list[CaseSeed] = []
    seen: set[tuple[str, datetime, datetime]] = set()
    for stratum in POSITIVE_STRATA:
        for candidate in by_stratum[stratum]:
            key = (candidate.seed_node, candidate.window.start, candidate.window.end)
            if key not in seen:
                seen.add(key)
                seeds.append(candidate)
    return seeds


def _licit_node_positions(index: GraphIndex) -> np.ndarray:
    """Return positions of accounts that appear in no laundering transaction.

    Args:
        index: The graph index.

    Returns:
        Node positions, ascending.
    """
    tainted = np.zeros(index.num_nodes, dtype=bool)
    flagged = np.flatnonzero(index.laundering)
    tainted[index.src_index[flagged]] = True
    tainted[index.dst_index[flagged]] = True
    return np.flatnonzero(~tainted)


def _window_active(index: GraphIndex, positions: np.ndarray, window: TimeWindow) -> np.ndarray:
    """Filter node positions to those with activity overlapping a window.

    Extraction refuses to build an empty case, so proposing seeds that cannot have one is
    wasted work. The node table's ``first_seen``/``last_seen`` make the check a vectorised
    comparison rather than a graph traversal.

    Args:
        index: The graph index.
        positions: Candidate node positions.
        window: The window to test against.

    Returns:
        The subset whose activity span overlaps the window.
    """
    if index.first_seen is None or index.last_seen is None:
        return positions
    lo, hi = to_micros(window.start), to_micros(window.end)
    return positions[(index.first_seen[positions] <= hi) & (index.last_seen[positions] >= lo)]


def _probe_active(
    index: GraphIndex,
    pool: np.ndarray,
    window: TimeWindow,
    rng: np.random.Generator,
    probes: int = 32,
) -> int | None:
    """Draw one pool member that was active in a window, by random probing.

    Filtering the whole pool would be the obvious implementation and is quadratic in
    practice: an activity bucket can hold a hundred thousand accounts and a corpus needs
    tens of thousands of negatives, so the scan dominates the entire build. Probing a
    handful of random members instead is O(1) per draw and, since most accounts in a
    bucket are active in most windows, usually succeeds on the first probe.

    Args:
        index: The graph index.
        pool: Candidate node positions.
        window: The window the seed must be active in.
        rng: Seeded generator.
        probes: How many members to try before giving up on this pool.

    Returns:
        An active node position, or None if no probe was active.
    """
    if pool.size == 0:
        return None
    if index.first_seen is None or index.last_seen is None:
        return int(pool[rng.integers(pool.size)])
    lo, hi = to_micros(window.start), to_micros(window.end)
    for candidate in pool[rng.integers(pool.size, size=min(probes, pool.size))]:
        position = int(candidate)
        if index.first_seen[position] <= hi and index.last_seen[position] >= lo:
            return position
    return None


def matched_licit_seeds(
    graph: CanonicalGraph,
    index: GraphIndex,
    positives: list[CaseRecord],
    n_wanted: int,
    rng: np.random.Generator,
) -> list[CaseSeed]:
    """Draw licit seeds matched to the positive population on activity and window.

    Each draw copies a real positive case's window verbatim and picks a laundering-free
    account from the same log-degree bucket that was active in it. Matching both channels
    is the point: an unmatched negative population lets a classifier win on transaction
    count or on the calendar, and a narrative model trained against that never has to learn
    anything about structure.

    Args:
        graph: The substrate graph.
        index: The graph index.
        positives: Built positive cases, supplying the target distribution.
        n_wanted: How many seeds to return.
        rng: Seeded generator.

    Returns:
        Up to ``n_wanted`` seeds, deduplicated on ``(seed_node, window)``.

    Raises:
        CaseSamplingError: If there are no positives to match against, or the index is
            over a different substrate than ``graph``.
    """
    if not positives:
        raise CaseSamplingError("cannot match negatives without a positive population")
    if index.graph.dataset != graph.dataset:
        raise CaseSamplingError(
            f"graph is {graph.dataset!r} but the index is over {index.graph.dataset!r}"
        )
    licit = _licit_node_positions(index)
    buckets = _activity_buckets(index)
    by_bucket: dict[int, np.ndarray] = {
        b: licit[buckets[licit] == b] for b in range(ACTIVITY_BUCKETS)
    }

    seeds: list[CaseSeed] = []
    seen: set[tuple[str, datetime, datetime]] = set()
    # Bounded rather than while-true: a bucket can legitimately hold no account active in
    # a given window, and an unbounded retry would spin forever on a narrow one.
    attempts = 0
    max_attempts = max(n_wanted * 8, 64)
    order = rng.permutation(len(positives))
    while len(seeds) < n_wanted and attempts < max_attempts:
        template = positives[int(order[attempts % len(order)])]
        attempts += 1
        pool = by_bucket.get(template.activity_bucket)
        if pool is None:
            continue
        position = _probe_active(index, pool, template.window, rng)
        if position is None:
            continue
        node_id = index.node_id_at(position)
        key = (node_id, template.window_start, template.window_end)
        if key in seen:
            continue
        seen.add(key)
        seeds.append(CaseSeed(seed_node=node_id, window=template.window, case_class="licit"))
    return seeds


# ----------------------------------------------------------------- building ---


def _build_one(
    graph: CanonicalGraph,
    index: GraphIndex,
    seed: CaseSeed,
    params: ExtractionParams,
    buckets: np.ndarray,
) -> tuple[CaseRecord, CaseCut] | None:
    """Cut and score a single case, without materialising its attribute columns.

    Args:
        graph: The substrate graph.
        index: The graph index.
        seed: The sampling decision.
        params: The extraction protocol.
        buckets: Per-node activity buckets, computed once by the caller.

    Returns:
        The record paired with its positional cut, or None when the seed has no activity
        in its window and therefore no case to build.
    """
    try:
        cut = cut_case(graph, seed.seed_node, seed.window, params, index=index)
    except CaseExtractionError:
        return None

    edges = index.edges[cut.edge_positions.tolist()]
    motifs = score_edges(edges)

    pattern_ids = tuple(seed.pattern_ids)
    if "pattern_id" in edges.columns:
        found = edges["pattern_id"].drop_nulls().unique().to_list()
        pattern_ids = tuple(sorted({*pattern_ids, *(str(p) for p in found)}))

    node_ids = index.nodes["node_id"].gather(cut.node_positions.tolist()).to_list()
    digest = hashlib.sha256(canonical_json(sorted(node_ids)).encode("utf-8")).hexdigest()
    record = CaseRecord(
        case_id=cut.case_id,
        dataset=graph.dataset,
        seed_node=seed.seed_node,
        window_start=seed.window.start,
        window_end=seed.window.end,
        case_class=seed.case_class,
        label=cut.label or "licit",
        typology=seed.typology if seed.case_class == "suspicious" else None,
        pattern_ids=pattern_ids,
        n_nodes=int(cut.node_positions.size),
        n_edges=int(cut.edge_positions.size),
        activity_bucket=int(buckets[index.position_of(seed.seed_node)]),
        motif_best=motifs.best,
        motif_score=motifs.best_score,
        structural_hash=short(digest, 16),
        provenance=dict(cut.provenance) | {"motifs": motifs.to_dict()},
    )
    return record, cut


def _membership_frames(
    built: list[tuple[CaseRecord, CaseCut]], index: GraphIndex
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Turn built cases into the two by-reference membership tables.

    Args:
        built: Records paired with their positional cuts.
        index: The graph index, supplying node identifiers for the node table.

    Returns:
        ``(node_membership, edge_membership)``.
    """
    node_case: list[str] = []
    node_pos: list[int] = []
    edge_case: list[str] = []
    edge_pos: list[int] = []
    for record, cut in built:
        node_case.extend([record.case_id] * cut.node_positions.size)
        node_pos.extend(int(p) for p in cut.node_positions)
        edge_case.extend([record.case_id] * cut.edge_positions.size)
        edge_pos.extend(int(p) for p in cut.edge_positions)
    nodes = pl.DataFrame(
        {"case_id": node_case, "node_index": node_pos},
        schema={"case_id": pl.Utf8, "node_index": pl.Int64},
    ).with_columns(index.nodes["node_id"].gather(node_pos).alias("node_id"))
    edges = pl.DataFrame(
        {"case_id": edge_case, "edge_index": edge_pos},
        schema={"case_id": pl.Utf8, "edge_index": pl.Int64},
    )
    return nodes, edges


def _partition_negatives(
    candidates: list[tuple[CaseRecord, CaseCut]],
    sampling: SamplingParams,
    wanted_negative: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[CaseRecord, CaseCut]], list[tuple[CaseRecord, CaseCut]]]:
    """Split the licit candidate pool into hard negatives and ordinary ones.

    Two things here are deliberate and neither is obvious.

    **Mining is per window, not global.** Taking the top scorers across the whole pool
    concentrates hard negatives wherever high-motif licit activity happens to fall in
    time, and the temporal split then inherits that clumping: measured on HI-Small, a
    globally-mined corpus at 29% hard negatives overall produced 25% in train and **64% in
    test**, which would make every test metric incomparable to validation. Mining the same
    share inside each window makes the rate uniform across time by construction, so any
    temporal boundary cuts a representative population.

    **Ordinary negatives are drawn at random, not taken from the top of what is left.**
    An "easy" negative selected for being the next-highest scorer is not easy, and the
    contrast between the two populations — which is the whole point of having both — would
    be softened by exactly the amount the mining threshold moved.

    Args:
        candidates: Built licit cases paired with their cuts.
        sampling: The sampling plan.
        wanted_negative: Total negatives required.
        rng: Seeded generator.

    Returns:
        ``(hard, easy)``. Both are sorted by case id so the corpus is reproducible.
    """
    by_window: dict[tuple[datetime, datetime], list[tuple[CaseRecord, CaseCut]]] = defaultdict(list)
    for item in candidates:
        by_window[(item[0].window_start, item[0].window_end)].append(item)

    hard: list[tuple[CaseRecord, CaseCut]] = []
    remainder: list[tuple[CaseRecord, CaseCut]] = []
    for window in sorted(by_window):
        group = sorted(by_window[window], key=lambda item: (-item[0].motif_score, item[0].case_id))
        quota = int(round(len(group) * sampling.hard_negative_fraction))
        taken = 0
        for item in group:
            if taken < quota and item[0].motif_score >= sampling.hard_negative_min_score:
                hard.append(item)
                taken += 1
            else:
                remainder.append(item)

    # Trim proportionally if the pool overshot, so the ratio survives the trim.
    wanted_hard = min(len(hard), int(round(wanted_negative * sampling.hard_negative_fraction)))
    hard = sorted(hard, key=lambda item: item[0].case_id)
    if len(hard) > wanted_hard:
        keep = rng.permutation(len(hard))[:wanted_hard]
        hard = [hard[int(i)] for i in sorted(keep)]

    wanted_easy = max(wanted_negative - len(hard), 0)
    remainder = sorted(remainder, key=lambda item: item[0].case_id)
    if len(remainder) > wanted_easy:
        keep = rng.permutation(len(remainder))[:wanted_easy]
        remainder = [remainder[int(i)] for i in sorted(keep)]
    return hard, remainder


def sample_cases(
    graph: CanonicalGraph,
    index: GraphIndex,
    extraction: ExtractionParams,
    sampling: SamplingParams,
    *,
    source_manifest_hash: str = "",
) -> CaseCollection:
    """Build the full balanced case population: positives, negatives, hard negatives.

    Args:
        graph: The substrate graph.
        index: An index over ``graph``.
        extraction: The case-construction protocol.
        sampling: The sampling plan.
        source_manifest_hash: Digest identifying ``graph``'s interim artifacts, recorded so
            the by-reference positions can never be applied to a different graph.

    Returns:
        The built collection, with stratification counts attached.

    Raises:
        CaseSamplingError: If the substrate has no laundering ground truth, or if hard
            negatives cannot reach :data:`MINIMUM_HARD_NEGATIVE_RATE` of the negative
            population — a gate criterion, so it fails rather than quietly shipping a
            corpus that cannot support the paper's central claim.
    """
    rng = np.random.default_rng(sampling.seed)

    # ------------------------------------------------------------ positives ---
    candidates = positive_seeds(graph, sampling, rng)
    by_stratum: dict[str, list[CaseSeed]] = {name: [] for name in POSITIVE_STRATA}
    for candidate in candidates:
        by_stratum[candidate.typology or "unclassified"].append(candidate)
    wanted_positive = int(round(sampling.n_cases * sampling.positive_fraction))
    ceiling = max(1, int(wanted_positive * sampling.max_stratum_share))
    allocation = _allocate_evenly(
        wanted_positive, {name: min(len(v), ceiling) for name, v in by_stratum.items()}
    )

    buckets = _activity_buckets(index)
    built: list[tuple[CaseRecord, CaseCut]] = []
    for stratum in POSITIVE_STRATA:
        pool = by_stratum[stratum]
        if not pool:
            continue
        order = rng.permutation(len(pool))
        taken = 0
        for position in order:
            if taken >= allocation[stratum]:
                break
            outcome = _build_one(graph, index, pool[int(position)], extraction, buckets)
            if outcome is not None:
                built.append(outcome)
                taken += 1

    positives = [record for record, _ in built]
    if not positives:
        raise CaseSamplingError("no positive case could be built; check the window padding")

    # ------------------------------------------------------------ negatives ---
    wanted_negative = sampling.n_cases - len(positives)
    wanted_hard = int(round(wanted_negative * sampling.hard_negative_fraction))
    pool_size = min(
        int(wanted_negative + wanted_hard * (sampling.hard_negative_oversample - 1.0)),
        wanted_negative * 8,
    )
    licit_seeds = matched_licit_seeds(graph, index, positives, pool_size, rng)

    licit_built: list[tuple[CaseRecord, CaseCut]] = []
    for seed in licit_seeds:
        outcome = _build_one(graph, index, seed, extraction, buckets)
        # A "licit" seed can still land in a case that touches flagged activity two hops
        # away. That case is not a negative and must not be labelled as one.
        if outcome is not None and outcome[0].label == "licit":
            licit_built.append(outcome)

    hard, easy = _partition_negatives(licit_built, sampling, wanted_negative, rng)

    negatives_total = len(hard) + len(easy)
    if negatives_total and len(hard) / negatives_total < MINIMUM_HARD_NEGATIVE_RATE:
        raise CaseSamplingError(
            f"hard negatives are {len(hard) / negatives_total:.1%} of the negative "
            f"population, below the {MINIMUM_HARD_NEGATIVE_RATE:.0%} gate criterion. "
            "Raise hard_negative_oversample or lower hard_negative_min_score, and record "
            "the change as a decision."
        )

    for record, cut in hard:
        built.append((dataclasses.replace(record, case_class="hard_negative"), cut))
    built.extend(easy)

    records = [record for record, _ in built]
    node_membership, edge_membership = _membership_frames(built, index)
    return CaseCollection(
        dataset=graph.dataset,
        records=records,
        node_membership=node_membership,
        edge_membership=edge_membership,
        source_manifest_hash=source_manifest_hash,
        extraction_params=extraction.to_dict(),
        sampling_params=sampling.to_dict(),
        stratification=summarise_stratification(records),
    )


def build_realistic_stream(
    graph: CanonicalGraph,
    index: GraphIndex,
    extraction: ExtractionParams,
    *,
    window: TimeWindow,
    n_cases: int,
    seed: int,
    target_prevalence: float | None = None,
    source_manifest_hash: str = "",
) -> CaseCollection:
    """Build a second test set at the substrate's true class balance.

    The balanced test set answers "can it write a good narrative for a suspicious case".
    It cannot answer "would this be useful in a real alert queue", because a real queue is
    overwhelmingly licit and the cost of an overclaiming narrative scales with how many
    licit cases the analyst reads. Without this stream the "validation in a realistic
    decision setting" claim is hollow.

    Seeds are drawn **uniformly** from accounts active in the window, with no matching, no
    stratification and no laundering filter, so whatever prevalence emerges is the
    substrate's own. That measured rate is the honest number, and it is recorded; forcing
    a textbook 1-in-several-hundred would be choosing the answer.

    Args:
        graph: The substrate graph.
        index: An index over ``graph``.
        extraction: The case-construction protocol, identical to the balanced corpus.
        window: The temporal window to draw from — normally the test split's.
        n_cases: How many cases to build.
        seed: RNG seed.
        target_prevalence: If given, suspicious cases are down-sampled to this rate. The
            observed rate is recorded either way, so the adjustment is never invisible.
        source_manifest_hash: Digest identifying ``graph``'s interim artifacts.

    Returns:
        The stream, with ``stratification`` carrying ``observed_prevalence`` and, when
        down-sampling occurred, ``target_prevalence``.

    Raises:
        CaseSamplingError: If no account is active in ``window``.
    """
    rng = np.random.default_rng(seed)
    active = _window_active(index, np.arange(index.num_nodes), window)
    if active.size == 0:
        raise CaseSamplingError(f"no account is active in {window.start}..{window.end}")

    buckets = _activity_buckets(index)
    built: list[tuple[CaseRecord, CaseCut]] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = max(n_cases * 6, 128)
    while len(built) < n_cases and attempts < max_attempts:
        attempts += 1
        node_id = index.node_id_at(int(active[rng.integers(active.size)]))
        if node_id in seen:
            continue
        seen.add(node_id)
        outcome = _build_one(
            graph,
            index,
            CaseSeed(seed_node=node_id, window=window, case_class="licit"),
            extraction,
            buckets,
        )
        if outcome is None:
            continue
        record, cut = outcome
        built.append((dataclasses.replace(record, case_class=record.label), cut))

    observed = sum(1 for record, _ in built if record.label == "suspicious")
    observed_rate = observed / len(built) if built else 0.0
    if target_prevalence is not None and observed_rate > target_prevalence:
        keep_positive = int(round(len(built) * target_prevalence))
        positives = [item for item in built if item[0].label == "suspicious"]
        negatives = [item for item in built if item[0].label != "suspicious"]
        chosen = rng.permutation(len(positives))[:keep_positive]
        built = negatives + [positives[int(i)] for i in sorted(chosen)]

    records = [record for record, _ in built]
    node_membership, edge_membership = _membership_frames(built, index)
    stratification = summarise_stratification(records)
    stratification["observed_prevalence"] = round(observed_rate, 6)
    stratification["target_prevalence"] = target_prevalence
    stratification["sampling"] = "uniform over accounts active in the window"
    return CaseCollection(
        dataset=graph.dataset,
        records=records,
        node_membership=node_membership,
        edge_membership=edge_membership,
        source_manifest_hash=source_manifest_hash,
        extraction_params=extraction.to_dict(),
        sampling_params={"n_cases": n_cases, "seed": seed, "window": window.to_dict()},
        stratification=stratification,
    )


def availability_of(collection: CaseCollection, index: GraphIndex) -> AvailabilityMask:
    """Return the availability mask governing every case in a collection.

    Args:
        collection: The case population.
        index: An index over the graph the cases were cut from.

    Returns:
        The substrate's mask. Invariant 4 travels with the cases, not just the substrate.

    Raises:
        CaseSamplingError: If the index is over a different substrate.
    """
    if index.graph.dataset != collection.dataset:
        raise CaseSamplingError(
            f"collection is over {collection.dataset!r}, index is over " f"{index.graph.dataset!r}"
        )
    return index.graph.availability
