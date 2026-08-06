"""Temporal splits and the frozen manifests every downstream phase loads them from.

A reviewer who spots a random split stops reading, and they are right to. Both substrates
are temporal, and a random split lets a model see the future: the same laundering stream
appears on both sides of the boundary, the same accounts appear in training and test, and
every number goes up for reasons that have nothing to do with the method.

Four rules, in the order they are applied.

**Time.** Cases are ordered by window start, and every test case begins after every train
case ends. Val sits between them.

**Natural boundaries, not percentiles.** Boundaries sit on a natural grid — midnight by
default — and are chosen by searching that grid for the pair whose *achieved* proportions
come closest to the requested ones. A quantile of window starts does not work on this
substrate: cases have duration, HI-Small's span is 17.7 days with 99.98% of transactions
in the first ten, and the 70th and 85th percentiles of case start fall about a day apart —
a one-day val band that cannot hold a four-day case. See :func:`temporal_boundaries`.

**Buffer gap.** A case whose window straddles a boundary has evidence on both sides. Those
are dropped outright, along with any case within ``buffer_hours`` of the boundary, so no
case can bleed across.

**Stream atomicity.** A laundering stream that appeared in two splits would be the single
most expensive leak in the corpus: the model would have seen the same laundering event
during training that it is being tested on. Each stream is assigned to the split holding
its earliest case, and its cases anywhere else are dropped.

Node-disjointness is then *measured*, not enforced by default. See
:data:`DEFAULT_OVERLAP_MODE` for why, and D-021 for the decision.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl

from g2t_aml.data.case_extraction import from_micros, to_micros
from g2t_aml.data.case_sampling import CaseCollection, CaseRecord, summarise_stratification
from g2t_aml.utils.hashing import hash_id_list
from g2t_aml.utils.io import read_json, write_json

#: Bumping this invalidates every committed split manifest.
MANIFEST_VERSION = "1.0.0"

#: The three splits, in temporal order. Order is meaningful: train precedes val precedes
#: test, and the audit checks exactly that.
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")

OverlapMode = Literal["report", "strict"]

#: Node overlap is reported, not enforced, by default. HI-Small's giant component holds
#: 72.2% of its 515,088 accounts, so *some* account recurrence between a train case and a
#: test case is a property of the substrate rather than a defect in the split — the same
#: correspondent bank appears in both, as it would in any real institution's data. Strict
#: mode drops every test case touching a train node, which on a graph this dense removes
#: most of the test set and biases what remains toward isolated, atypical activity.
#: Reporting the rate and publishing it is the honest choice; strict mode exists so the
#: claim can be checked. See D-021.
DEFAULT_OVERLAP_MODE: OverlapMode = "report"

#: Floating-point slack allowed when checking the proportions sum to one.
PROPORTION_TOLERANCE = 1e-9

#: Default proportions, by case count over time.
DEFAULT_PROPORTIONS: tuple[float, float, float] = (0.70, 0.15, 0.15)


class SplitError(RuntimeError):
    """Raised when a split cannot be constructed to the requested shape."""


@dataclass(frozen=True)
class SplitParams:
    """How the temporal split is cut.

    Attributes:
        proportions: Train/val/test shares of the case population, by time.
        buffer_hours: Cases within this many hours of a boundary are dropped.
        boundary_snap_hours: Boundaries are snapped to a multiple of this many hours from
            the substrate's first timestamp. 24 means midnight.
        min_split_fraction: A boundary pair is only a candidate if every split holds at
            least this share of the surviving cases. A four-case val split satisfies
            "non-empty" and is useless; this makes degeneracy a search constraint rather
            than something discovered downstream.
        retention_weight: How much a dropped case counts against a boundary pair, relative
            to a unit of proportion error. Without it the search buys an exact 70/15/15 by
            discarding most of the corpus, which is a worse split than a slightly uneven
            one built from twice as many cases.
        overlap_mode: ``"report"`` or ``"strict"``. See :data:`DEFAULT_OVERLAP_MODE`.
        mode: Recorded in the manifest. Only ``"temporal"`` is implemented, and no
            alternative will be added without a decision entry — invariant 2.
    """

    proportions: tuple[float, float, float] = DEFAULT_PROPORTIONS
    buffer_hours: float = 24.0
    boundary_snap_hours: float = 24.0
    min_split_fraction: float = 0.05
    retention_weight: float = 1.5
    overlap_mode: OverlapMode = DEFAULT_OVERLAP_MODE
    mode: str = "temporal"

    def __post_init__(self) -> None:
        """Validate the split plan.

        Raises:
            ValueError: If the proportions are not three positive numbers summing to 1,
                a duration is negative, or the overlap mode is unknown.
        """
        if len(self.proportions) != len(SPLIT_NAMES):
            raise ValueError(f"expected {len(SPLIT_NAMES)} proportions, got {self.proportions}")
        if any(p <= 0 for p in self.proportions):
            raise ValueError(f"every proportion must be positive, got {self.proportions}")
        if abs(sum(self.proportions) - 1.0) > PROPORTION_TOLERANCE:
            raise ValueError(f"proportions must sum to 1, got {sum(self.proportions)}")
        if self.buffer_hours < 0 or self.boundary_snap_hours <= 0:
            raise ValueError("buffer_hours must be >= 0 and boundary_snap_hours > 0")
        if self.retention_weight < 0:
            raise ValueError("retention_weight must be >= 0")
        if not 0.0 <= self.min_split_fraction < 1.0 / len(SPLIT_NAMES):
            raise ValueError(
                f"min_split_fraction must be in [0, {1 / len(SPLIT_NAMES):.3f}), "
                f"got {self.min_split_fraction}"
            )
        if self.overlap_mode not in ("report", "strict"):
            raise ValueError(f"unknown overlap_mode {self.overlap_mode!r}")
        if self.mode != "temporal":
            raise ValueError(
                f"only the temporal split is implemented, got {self.mode!r}. A random "
                "split inflates every number in the paper (invariant 2)."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the parameters as a plain dict.

        Returns:
            Field name to value, with ``proportions`` as a list.
        """
        data = dataclasses.asdict(self)
        data["proportions"] = list(self.proportions)
        return data


@dataclass
class SplitAssignment:
    """The outcome of splitting: who went where, and who was dropped and why.

    Attributes:
        splits: Split name to case ids, in temporal order within each split.
        boundaries: The two instants separating train/val and val/test.
        dropped: Case id to the reason it was excluded.
        params: The plan used.
    """

    splits: dict[str, list[str]]
    boundaries: tuple[datetime, datetime]
    dropped: dict[str, str] = field(default_factory=dict)
    params: SplitParams = field(default_factory=SplitParams)

    @property
    def counts(self) -> dict[str, int]:
        """Return the size of each split.

        Returns:
            Split name to case count.
        """
        return {name: len(ids) for name, ids in self.splits.items()}

    def drop_reasons(self) -> dict[str, int]:
        """Tally why cases were dropped.

        Returns:
            Reason to count, sorted by reason.
        """
        tally: dict[str, int] = defaultdict(int)
        for reason in self.dropped.values():
            tally[reason] += 1
        return dict(sorted(tally.items()))


def _assign_masks(
    starts: np.ndarray,
    ends: np.ndarray,
    first: float,
    second: float,
    buffer_micros: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply one candidate boundary pair to the whole population at once.

    Args:
        starts: Window starts, in microseconds (see ``case_extraction.to_micros``).
        ends: Window ends, in microseconds.
        first: Train/val boundary, in microseconds.
        second: Val/test boundary, in microseconds.
        buffer_micros: Buffer half-width, in microseconds.

    Returns:
        ``(train, val, test)`` boolean masks. A case excluded from all three either
        straddles a boundary or falls inside the buffer.
    """
    straddles = ((starts < first) & (ends > first)) | ((starts < second) & (ends > second))
    near = np.zeros(starts.shape, dtype=bool)
    for boundary in (first, second):
        near |= (np.abs(starts - boundary) < buffer_micros) | (
            np.abs(ends - boundary) < buffer_micros
        )
    keep = ~(straddles | near)
    return (
        keep & (ends <= first),
        keep & (starts > first) & (ends <= second),
        keep & (starts > second),
    )


def temporal_boundaries(
    records: list[CaseRecord], params: SplitParams
) -> tuple[datetime, datetime]:
    """Choose the two split boundaries by searching for the requested proportions.

    A quantile of window starts is the obvious rule and it does not work here. Cases have
    *duration* — HI-Small's median case window is four days against a ten-day substrate —
    so the number of cases that survive a boundary pair depends on how the whole
    population's intervals sit relative to both boundaries, not on where any one quantile
    falls. Placing boundaries at the 70th and 85th percentile of window starts puts them
    about a day apart, and a one-day val band cannot hold a four-day case: the naive rule
    yields an empty val split on this substrate.

    Every boundary pair on the snap grid is therefore evaluated against the actual case
    population and scored on how close the achieved proportions come to the requested
    ones, with the surviving fraction as a tie-break. That is both what "split at natural
    temporal boundaries, not exact percentiles" asks for and the only rule that adapts to a
    population whose geometry is not known in advance.

    Args:
        records: The case population.
        params: The split plan.

    Returns:
        ``(train_val_boundary, val_test_boundary)``.

    Raises:
        SplitError: If there are too few cases to place two boundaries, or no boundary
            pair on the grid leaves all three splits non-empty — which means the case
            windows are wide relative to the substrate's span, and either the window
            duration cap or the proportions have to move.
    """
    if len(records) < len(SPLIT_NAMES):
        raise SplitError(f"cannot split {len(records)} cases into {len(SPLIT_NAMES)}")

    # Microseconds read literally, never via datetime.timestamp(): see to_micros.
    starts = np.array([to_micros(r.window_start) for r in records], dtype=np.float64)
    ends = np.array([to_micros(r.window_end) for r in records], dtype=np.float64)
    origin = starts.min()
    step = params.boundary_snap_hours * 3_600_000_000.0
    grid = np.arange(origin, ends.max() + step, step)
    if grid.size < len(SPLIT_NAMES):
        raise SplitError(
            f"the case population spans less than {2 * params.boundary_snap_hours}h; "
            "there is no room for two boundaries"
        )

    target = np.array(params.proportions, dtype=np.float64)
    buffer_micros = params.buffer_hours * 3_600_000_000.0
    best: tuple[float, float, float] | None = None
    for i, first in enumerate(grid[:-1]):
        for second in grid[i + 1 :]:
            train, val, test = _assign_masks(starts, ends, first, second, buffer_micros)
            counts = np.array([train.sum(), val.sum(), test.sum()], dtype=np.float64)
            total = counts.sum()
            if total == 0 or counts.min() < max(1, params.min_split_fraction * total):
                continue
            error = float(np.abs(counts / total - target).sum())
            kept = total / len(records)
            # Proportion error and discarded cases are traded off explicitly. Scoring on
            # proportion error alone buys an exact 70/15/15 by throwing away most of the
            # corpus; the earlier boundary breaks any remaining tie, so the choice is
            # fully determined.
            score = error + params.retention_weight * (1.0 - kept)
            candidate = (score, float(first), float(second))
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise SplitError(
            "no boundary pair gives every split at least "
            f"{params.min_split_fraction:.0%} of the surviving cases. Case windows are "
            "wide relative to the substrate's temporal span: a split band has to be at "
            "least one case-window wide to hold anything. Lower "
            "sampling.max_window_hours, or lower split.min_split_fraction and accept a "
            "small split."
        )
    return from_micros(best[1]), from_micros(best[2])


def temporal_split(records: list[CaseRecord], params: SplitParams | None = None) -> SplitAssignment:
    """Assign cases to train, val and test by time, dropping anything that would bleed.

    Args:
        records: The case population.
        params: The split plan. Defaults are 70/15/15 with a 24-hour buffer.

    Returns:
        The assignment, carrying the boundaries and a reason for every dropped case.

    Raises:
        SplitError: If the population is too small to split, or a split comes out empty —
            an empty val or test split is a configuration error that would otherwise
            surface much later as a confusing evaluation failure.
    """
    plan = params or SplitParams()
    first, second = temporal_boundaries(records, plan)
    buffer = timedelta(hours=plan.buffer_hours)

    splits: dict[str, list[CaseRecord]] = {name: [] for name in SPLIT_NAMES}
    dropped: dict[str, str] = {}
    for record in records:
        window = record.window
        if window.straddles(first) or window.straddles(second):
            dropped[record.case_id] = "straddles_boundary"
            continue
        if any(
            abs((window.start - boundary).total_seconds()) < buffer.total_seconds()
            or abs((window.end - boundary).total_seconds()) < buffer.total_seconds()
            for boundary in (first, second)
        ):
            dropped[record.case_id] = "within_buffer"
            continue
        if window.end <= first:
            splits["train"].append(record)
        elif window.end <= second:
            splits["val"].append(record)
        else:
            splits["test"].append(record)

    _enforce_stream_atomicity(splits, dropped)

    if empty := [name for name, items in splits.items() if not items]:
        raise SplitError(
            f"split(s) {empty} came out empty at boundaries {first} / {second}. "
            f"Proportions {plan.proportions} and a {plan.buffer_hours}h buffer are too "
            "aggressive for this case population's temporal spread."
        )
    return SplitAssignment(
        splits={
            name: [r.case_id for r in sorted(items, key=lambda r: (r.window_start, r.case_id))]
            for name, items in splits.items()
        },
        boundaries=(first, second),
        dropped=dropped,
        params=plan,
    )


def _enforce_stream_atomicity(splits: dict[str, list[CaseRecord]], dropped: dict[str, str]) -> None:
    """Keep every laundering stream inside exactly one split.

    A stream spanning two splits means the model trains on the same laundering event it is
    later tested on. Each stream is kept in the split holding its earliest case; its cases
    elsewhere are dropped.

    Args:
        splits: Split name to records. Mutated in place.
        dropped: Case id to reason. Mutated in place.
    """
    home: dict[str, str] = {}
    order = {name: i for i, name in enumerate(SPLIT_NAMES)}
    for name in SPLIT_NAMES:
        for record in splits[name]:
            for pattern_id in record.pattern_ids:
                if pattern_id not in home or order[name] < order[home[pattern_id]]:
                    home.setdefault(pattern_id, name)
    for name in SPLIT_NAMES:
        keep: list[CaseRecord] = []
        for record in splits[name]:
            if any(home.get(p, name) != name for p in record.pattern_ids):
                dropped[record.case_id] = "stream_in_earlier_split"
            else:
                keep.append(record)
        splits[name] = keep


# ------------------------------------------------------------------ overlap ---


@dataclass(frozen=True)
class OverlapReport:
    """How much node and edge identity is shared between splits.

    Attributes:
        node_overlap_rate: Fraction of test cases containing at least one node that also
            appears in a train case.
        val_node_overlap_rate: The same measure for val against train.
        edge_overlap_rate: Fraction of test cases sharing at least one *transaction* with
            a train case. Far more alarming than node overlap: a shared account is
            ordinary, a shared transaction means the same event is on both sides.
        shared_nodes: Count of distinct nodes appearing in both train and test.
        shared_edges: Count of distinct transactions appearing in both train and test.
        overlapping_test_cases: Test case ids sharing a node with train, sorted.
        mode: The overlap mode in force.
    """

    node_overlap_rate: float
    val_node_overlap_rate: float
    edge_overlap_rate: float
    shared_nodes: int
    shared_edges: int
    overlapping_test_cases: tuple[str, ...]
    mode: OverlapMode

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable report, without the full case-id list.

        Returns:
            The rates and counts. The case-id list is omitted deliberately: it can run to
            thousands of entries and belongs in the audit report, not the manifest.
        """
        data = dataclasses.asdict(self)
        data.pop("overlapping_test_cases")
        data["n_overlapping_test_cases"] = len(self.overlapping_test_cases)
        return data


def measure_overlap(collection: CaseCollection, assignment: SplitAssignment) -> OverlapReport:
    """Measure node and edge identity shared between train and the later splits.

    Args:
        collection: The case population, supplying node and edge membership.
        assignment: The split assignment.

    Returns:
        The overlap report.
    """
    train = set(assignment.splits["train"])
    val = set(assignment.splits["val"])
    test = set(assignment.splits["test"])

    nodes = collection.node_membership
    train_nodes = set(nodes.filter(pl.col("case_id").is_in(list(train)))["node_id"].to_list())
    test_rows = nodes.filter(pl.col("case_id").is_in(list(test)))
    val_rows = nodes.filter(pl.col("case_id").is_in(list(val)))

    overlapping = sorted(
        test_rows.filter(pl.col("node_id").is_in(list(train_nodes)))["case_id"].unique().to_list()
    )
    val_overlapping = val_rows.filter(pl.col("node_id").is_in(list(train_nodes)))[
        "case_id"
    ].n_unique()
    shared_nodes = len(train_nodes & set(test_rows["node_id"].to_list()))

    edges = collection.edge_membership
    train_edges = set(edges.filter(pl.col("case_id").is_in(list(train)))["edge_index"].to_list())
    test_edge_rows = edges.filter(pl.col("case_id").is_in(list(test)))
    shared_edges = train_edges & set(test_edge_rows["edge_index"].to_list())
    edge_overlapping = (
        test_edge_rows.filter(pl.col("edge_index").is_in(list(shared_edges)))["case_id"].n_unique()
        if shared_edges
        else 0
    )

    return OverlapReport(
        node_overlap_rate=round(len(overlapping) / len(test), 6) if test else 0.0,
        val_node_overlap_rate=round(val_overlapping / len(val), 6) if val else 0.0,
        edge_overlap_rate=round(edge_overlapping / len(test), 6) if test else 0.0,
        shared_nodes=shared_nodes,
        shared_edges=len(shared_edges),
        overlapping_test_cases=tuple(overlapping),
        mode=assignment.params.overlap_mode,
    )


def apply_overlap_mode(assignment: SplitAssignment, overlap: OverlapReport) -> SplitAssignment:
    """Drop node-overlapping test cases when strict mode is in force.

    Args:
        assignment: The split assignment.
        overlap: The measured overlap.

    Returns:
        The assignment unchanged in ``report`` mode, or with the overlapping test cases
        removed in ``strict`` mode.
    """
    if assignment.params.overlap_mode != "strict":
        return assignment
    removed = set(overlap.overlapping_test_cases)
    return SplitAssignment(
        splits={
            name: [cid for cid in ids if not (name == "test" and cid in removed)]
            for name, ids in assignment.splits.items()
        },
        boundaries=assignment.boundaries,
        dropped=assignment.dropped | {cid: "node_overlap_strict" for cid in removed},
        params=assignment.params,
    )


# ----------------------------------------------------------------- manifest ---


def build_manifest(
    collection: CaseCollection,
    assignment: SplitAssignment,
    overlap: OverlapReport,
    *,
    leakage_audit: dict[str, Any] | None = None,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Assemble the frozen split manifest.

    Every downstream job loads splits from this file by id and never recomputes them
    (invariant 2, D-006). The per-split content hash is over the *set* of ids, so a
    reordering is not a change but a single added or removed case is.

    Args:
        collection: The case population.
        assignment: The split assignment.
        overlap: The measured overlap.
        leakage_audit: The auditor's summary, when one has been run.
        created_at: Creation timestamp. Defaults to now, in UTC.

    Returns:
        The manifest, ready to be written and committed.
    """
    by_id = collection.by_id()
    return {
        "manifest_version": MANIFEST_VERSION,
        "dataset": collection.dataset,
        "created_at": (created_at or datetime.now(UTC)).isoformat(),
        "source_manifest_hash": collection.source_manifest_hash,
        "extraction_params": dict(collection.extraction_params),
        "sampling_params": dict(collection.sampling_params),
        "split_params": assignment.params.to_dict()
        | {
            "boundaries": [b.isoformat() for b in assignment.boundaries],
        },
        "splits": {
            name: {
                "case_ids": ids,
                "n": len(ids),
                "id_list_sha256": hash_id_list(ids),
            }
            for name, ids in assignment.splits.items()
        },
        "dropped": {
            "n": len(assignment.dropped),
            "by_reason": assignment.drop_reasons(),
        },
        "stratification": {
            name: summarise_stratification([by_id[cid] for cid in ids if cid in by_id])
            for name, ids in assignment.splits.items()
        }
        | {
            "overall": summarise_stratification(
                [by_id[cid] for ids in assignment.splits.values() for cid in ids if cid in by_id]
            )
        },
        "overlap": overlap.to_dict(),
        "leakage_audit": leakage_audit or {},
    }


def write_split_manifest(manifest: dict[str, Any], manifest_dir: str | Path) -> Path:
    """Write the manifest and the committed per-split id lists.

    Two artifacts, deliberately. ``splits.json`` is the machine-readable record this
    module reads back; the three ``.txt`` files are the literal id lists D-006 specifies,
    which are what makes a split reviewable in a diff.

    Args:
        manifest: The manifest from :func:`build_manifest`.
        manifest_dir: Destination, normally ``schemas/splits/<substrate>``.

    Returns:
        The path of the written ``splits.json``.

    Raises:
        OSError: If a write or rename fails.
    """
    out = Path(manifest_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        ids = manifest["splits"][name]["case_ids"]
        (out / f"{name}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
        write_json(
            out / f"{name}.sha256.json",
            {"n": len(ids), "id_list_sha256": manifest["splits"][name]["id_list_sha256"]},
            canonical=True,
        )
    return write_json(out / "splits.json", manifest, canonical=True)


def load_split_manifest(manifest_dir: str | Path) -> dict[str, Any]:
    """Read a committed split manifest and verify its content hashes.

    Args:
        manifest_dir: Directory holding ``splits.json``.

    Returns:
        The manifest.

    Raises:
        FileNotFoundError: If ``splits.json`` is absent.
        SplitError: If the manifest version is unsupported, or a split's id list does not
            match its recorded hash — which means the file was edited by hand, and every
            result derived from it is suspect.
    """
    path = Path(manifest_dir) / "splits.json"
    manifest = read_json(path)
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise SplitError(
            f"split manifest version mismatch: file has "
            f"{manifest.get('manifest_version')!r}, code expects {MANIFEST_VERSION!r}"
        )
    for name in SPLIT_NAMES:
        block = manifest["splits"][name]
        if hash_id_list(block["case_ids"]) != block["id_list_sha256"]:
            raise SplitError(
                f"{path}: the {name} id list does not match its recorded sha256. The "
                "manifest has been edited; regenerate it rather than repairing it."
            )
    return manifest


def split_of(manifest: dict[str, Any]) -> dict[str, str]:
    """Invert a manifest into a case-id to split-name lookup.

    Args:
        manifest: A loaded manifest.

    Returns:
        Case id to split name. Dropped cases are absent.
    """
    return {
        case_id: name for name in SPLIT_NAMES for case_id in manifest["splits"][name]["case_ids"]
    }
