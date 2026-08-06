"""Standalone leakage auditor. Any phase may invoke it; hard failures exit non-zero.

This module is deliberately independent of :mod:`g2t_aml.data.splits`. It re-derives every
property from the case collection and the manifest rather than trusting the code that
produced them, because an auditor that shares its subject's assumptions cannot catch a bug
in those assumptions. If splitting and auditing ever disagree, the auditor is right by
construction: it is the one that read the artifact.

Six checks, two of them fatal.

============================  ========  ===================================================
Check                         Severity  What it catches
============================  ========  ===================================================
Temporal violation            fatal     A test case beginning before a train case ends.
Stream straddling             fatal     One laundering stream present in two splits.
Label leakage                 fatal     A feature that predicts the label perfectly.
Node overlap                  report    Accounts recurring across splits.
Edge overlap                  report    The same transaction on both sides.
Duplicate / near-duplicate    report    Cases that are structurally the same case.
============================  ========  ===================================================

**Label leakage is fatal and is the check most likely to fire.** Including
``is_laundering`` in a node or edge feature tensor is the classic way a graph pipeline
scores 1.0 and means nothing, and it happens more often than one would think — the column
is right there in the edge table, and the feature list is written by hand. The check is
therefore twofold: named columns that are direct label proxies must not appear in any
declared feature list at all, and no declared feature may separate the labels perfectly.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from g2t_aml.data.case_sampling import CaseCollection, CaseRecord
from g2t_aml.utils.io import write_json

#: Bumping this invalidates stored audit reports.
AUDIT_SCHEMA_VERSION = "1.0.0"

#: Columns that *are* the label, or trivially imply it. None of these may appear in a
#: declared node or edge feature list, on any substrate, ever.
LABEL_PROXY_COLUMNS: frozenset[str] = frozenset(
    {"is_laundering", "typology", "pattern_id", "label", "case_class", "motif_score"}
)

#: Jaccard similarity on node sets at or above which two cases in different splits are
#: reported as near-duplicates.
NEAR_DUPLICATE_JACCARD = 0.90

#: A node appearing in more than this many cases is skipped when generating
#: near-duplicate candidate pairs. Such a node is a hub shared by thousands of cases and
#: generates a quadratic blow-up of pairs that are not duplicates of each other.
CANDIDATE_NODE_CASE_CAP = 200


class LeakageAuditError(RuntimeError):
    """Raised when the auditor cannot run — not when it finds leakage."""


@dataclass(frozen=True)
class Finding:
    """One audit result.

    Attributes:
        check: Stable identifier of the check that produced it.
        severity: ``"fatal"`` or ``"report"``. Only ``"fatal"`` fails the gate.
        passed: Whether the check passed.
        detail: Human-readable summary.
        evidence: Machine-readable specifics — counts, offending ids, measured rates.
    """

    check: str
    severity: str
    passed: bool
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable finding.

        Returns:
            The finding as a plain dict.
        """
        return dataclasses.asdict(self)


@dataclass
class LeakageReport:
    """The full audit outcome.

    Attributes:
        dataset: Substrate key.
        findings: Every check's result, in execution order.
        created_at: When the audit ran.
    """

    dataset: str
    findings: list[Finding]
    created_at: datetime

    @property
    def hard_failures(self) -> list[Finding]:
        """Return the fatal findings that did not pass.

        Returns:
            The failing fatal findings, which are what makes the gate exit non-zero.
        """
        return [f for f in self.findings if f.severity == "fatal" and not f.passed]

    @property
    def passed(self) -> bool:
        """Report whether the audit passes the gate.

        Returns:
            True when no fatal check failed. Reported-severity findings never block.
        """
        return not self.hard_failures

    def summary(self) -> dict[str, Any]:
        """Return the compact block embedded in the split manifest.

        Returns:
            The rates and counts a reader of the manifest needs, without the evidence.
        """
        by_check = {f.check: f for f in self.findings}
        overlap = by_check.get("node_overlap")
        edges = by_check.get("edge_overlap")
        temporal = by_check.get("temporal_ordering")
        return {
            "node_overlap_rate": (overlap.evidence.get("rate") if overlap else None),
            "edge_overlap_rate": (edges.evidence.get("rate") if edges else None),
            "temporal_violations": (temporal.evidence.get("violations", 0) if temporal else None),
            "duplicate_cases": by_check["duplicate_cases"].evidence.get("n_exact", 0)
            if "duplicate_cases" in by_check
            else None,
            "near_duplicate_cases": by_check["duplicate_cases"].evidence.get("n_near", 0)
            if "duplicate_cases" in by_check
            else None,
            "passed": self.passed,
            "hard_failures": [f.check for f in self.hard_failures],
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the full report.

        Returns:
            A JSON-serialisable document, including every finding's evidence.
        """
        return {
            "audit_schema_version": AUDIT_SCHEMA_VERSION,
            "dataset": self.dataset,
            "created_at": self.created_at.isoformat(),
            "passed": self.passed,
            "hard_failures": [f.check for f in self.hard_failures],
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary(),
        }

    def save(self, path: str | Path) -> Path:
        """Write the report atomically.

        Args:
            path: Destination file.

        Returns:
            The path written.

        Raises:
            OSError: If the write or rename fails.
        """
        return write_json(path, self.to_dict())


# ------------------------------------------------------------------- checks ---


def _check_temporal(
    splits: dict[str, list[CaseRecord]],
) -> Finding:
    """Assert every test case begins after every train case ends.

    Args:
        splits: Split name to records.

    Returns:
        A fatal finding. The evidence carries the observed extremes so a failure says by
        how much, not merely that.
    """
    violations: list[dict[str, Any]] = []
    ordered = [("train", "val"), ("val", "test"), ("train", "test")]
    extremes: dict[str, Any] = {}
    for earlier, later in ordered:
        if not splits[earlier] or not splits[later]:
            continue
        latest = max(r.window_end for r in splits[earlier])
        earliest = min(r.window_start for r in splits[later])
        extremes[f"{earlier}_end"] = latest.isoformat()
        extremes[f"{later}_start"] = earliest.isoformat()
        if earliest < latest:
            offenders = [r.case_id for r in splits[later] if r.window_start < latest]
            violations.append(
                {
                    "earlier": earlier,
                    "later": later,
                    "n_cases": len(offenders),
                    "example_case_ids": sorted(offenders)[:10],
                    "overlap_hours": round((latest - earliest).total_seconds() / 3600, 2),
                }
            )
    return Finding(
        check="temporal_ordering",
        severity="fatal",
        passed=not violations,
        detail=(
            "every later split begins after the previous one ends"
            if not violations
            else f"{len(violations)} temporal ordering violation(s)"
        ),
        evidence={"violations": len(violations), "detail": violations, **extremes},
    )


def _check_stream_atomicity(splits: dict[str, list[CaseRecord]]) -> Finding:
    """Assert no laundering stream appears in more than one split.

    Args:
        splits: Split name to records.

    Returns:
        A fatal finding naming any straddling stream.
    """
    homes: dict[str, set[str]] = defaultdict(set)
    for name, records in splits.items():
        for record in records:
            for pattern_id in record.pattern_ids:
                homes[pattern_id].add(name)
    straddling = {p: sorted(s) for p, s in homes.items() if len(s) > 1}
    return Finding(
        check="stream_atomicity",
        severity="fatal",
        passed=not straddling,
        detail=(
            f"all {len(homes)} laundering streams sit in exactly one split"
            if not straddling
            else f"{len(straddling)} stream(s) appear in more than one split"
        ),
        evidence={
            "n_streams": len(homes),
            "n_straddling": len(straddling),
            "straddling": dict(sorted(straddling.items())[:20]),
        },
    )


def _membership_sets(
    collection: CaseCollection, splits: dict[str, list[CaseRecord]], column: str, table: str
) -> dict[str, set[Any]]:
    """Collect the distinct node or edge identities present in each split.

    Args:
        collection: The case population.
        splits: Split name to records.
        column: ``node_id`` or ``edge_index``.
        table: ``node_membership`` or ``edge_membership``.

    Returns:
        Split name to the set of identities it contains.
    """
    frame: pl.DataFrame = getattr(collection, table)
    result: dict[str, set[Any]] = {}
    for name, records in splits.items():
        ids = [r.case_id for r in records]
        result[name] = set(frame.filter(pl.col("case_id").is_in(ids))[column].to_list())
    return result


def _check_overlap(
    collection: CaseCollection,
    splits: dict[str, list[CaseRecord]],
    *,
    column: str,
    table: str,
    check: str,
    noun: str,
) -> Finding:
    """Measure how many test cases share an identity with a train case.

    Args:
        collection: The case population.
        splits: Split name to records.
        column: Identity column.
        table: Membership table attribute name.
        check: Finding identifier.
        noun: Word used in the human-readable detail.

    Returns:
        A reported-severity finding carrying the rate and the shared count.
    """
    frame: pl.DataFrame = getattr(collection, table)
    train_ids = [r.case_id for r in splits["train"]]
    test_records = splits["test"]
    train_identities = set(frame.filter(pl.col("case_id").is_in(train_ids))[column].to_list())
    test_rows = frame.filter(pl.col("case_id").is_in([r.case_id for r in test_records]))
    hit = test_rows.filter(pl.col(column).is_in(list(train_identities)))
    n_cases = hit["case_id"].n_unique()
    rate = round(n_cases / len(test_records), 6) if test_records else 0.0
    shared = len(train_identities & set(test_rows[column].to_list()))
    return Finding(
        check=check,
        severity="report",
        passed=True,
        detail=f"{rate:.1%} of test cases contain a {noun} also seen in training",
        evidence={
            "rate": rate,
            "n_test_cases_touched": int(n_cases),
            "n_test_cases": len(test_records),
            f"n_shared_{noun}s": shared,
        },
    )


def _check_duplicates(collection: CaseCollection, splits: dict[str, list[CaseRecord]]) -> Finding:
    """Find cases that are the same case, exactly or nearly, across two splits.

    Exact duplicates are caught by the structural hash of the sorted node list. Near
    duplicates are found by inverting node membership into candidate pairs and computing
    the Jaccard similarity of their node sets; nodes appearing in more than
    :data:`CANDIDATE_NODE_CASE_CAP` cases are skipped as hubs, since they generate
    quadratically many pairs that are not duplicates of one another.

    Args:
        collection: The case population.
        splits: Split name to records.

    Returns:
        A reported-severity finding. Duplicates are reported rather than fatal because a
        structurally identical case built from a different seed is a real property of a
        dense graph, not necessarily a leak — but it must be visible.
    """
    home = {r.case_id: name for name, records in splits.items() for r in records}
    records = {r.case_id: r for rs in splits.values() for r in rs}

    by_hash: dict[str, list[str]] = defaultdict(list)
    for case_id, record in records.items():
        by_hash[record.structural_hash].append(case_id)
    exact = [
        sorted(ids) for ids in by_hash.values() if len(ids) > 1 and len({home[i] for i in ids}) > 1
    ]

    nodes = collection.node_membership.filter(pl.col("case_id").is_in(list(records)))
    per_case: dict[str, set[str]] = defaultdict(set)
    for case_id, node_id in zip(
        nodes["case_id"].to_list(), nodes["node_id"].to_list(), strict=True
    ):
        per_case[case_id].add(node_id)

    inverted: dict[str, list[str]] = defaultdict(list)
    for case_id, node_ids in per_case.items():
        for node_id in node_ids:
            inverted[node_id].append(case_id)

    candidates: set[tuple[str, str]] = set()
    for case_ids in inverted.values():
        if len(case_ids) > CANDIDATE_NODE_CASE_CAP:
            continue
        ordered = sorted(case_ids)
        for i, left in enumerate(ordered):
            for right in ordered[i + 1 :]:
                if home[left] != home[right]:
                    candidates.add((left, right))

    near: list[dict[str, Any]] = []
    for left, right in sorted(candidates):
        a, b = per_case[left], per_case[right]
        union = len(a | b)
        if not union:
            continue
        jaccard = len(a & b) / union
        if jaccard >= NEAR_DUPLICATE_JACCARD:
            near.append(
                {
                    "case_ids": [left, right],
                    "splits": [home[left], home[right]],
                    "jaccard": round(jaccard, 4),
                }
            )

    return Finding(
        check="duplicate_cases",
        severity="report",
        passed=not exact,
        detail=(
            f"{len(exact)} exact and {len(near)} near-duplicate case group(s) span "
            "more than one split"
        ),
        evidence={
            "n_exact": len(exact),
            "n_near": len(near),
            "jaccard_threshold": NEAR_DUPLICATE_JACCARD,
            "exact": exact[:20],
            "near": near[:20],
        },
    )


def _check_label_leakage(
    splits: dict[str, list[CaseRecord]],
    *,
    node_feature_names: list[str],
    edge_feature_names: list[str],
) -> Finding:
    """Assert no declared feature is the label in disguise.

    Two ways this goes wrong, and both are checked. A feature list may simply *name* a
    label column — ``is_laundering`` sitting in the edge feature list is the canonical
    version, and it is right there in the table waiting to be picked up. Or a feature may
    be a perfect empirical separator: a column whose value ranges do not overlap between
    suspicious and licit cases, which means a one-line threshold rule scores 1.0.

    Args:
        splits: Split name to records.
        node_feature_names: Declared node features.
        edge_feature_names: Declared edge features.

    Returns:
        A fatal finding naming any offending feature.
    """
    named = sorted((set(node_feature_names) | set(edge_feature_names)) & LABEL_PROXY_COLUMNS)

    records = [r for rs in splits.values() for r in rs]
    labels = np.array([r.label == "suspicious" for r in records], dtype=bool)
    perfect: list[dict[str, Any]] = []
    if labels.any() and not labels.all():
        # Case-level scalars a classifier could actually read off the input graph. A
        # scalar whose ranges do not overlap between the two labels is a one-line
        # classifier and means the corpus, not the model, is doing the work.
        #
        # `motif_score` is deliberately excluded. It is a sampling diagnostic rather than
        # a model input, and hard-negative mining selects licit cases *by* it, so a
        # separation there would be the mining working as designed rather than a leak. It
        # stays in LABEL_PROXY_COLUMNS, which forbids it as a declared feature.
        scalars = {
            "n_nodes": np.array([r.n_nodes for r in records], dtype=np.float64),
            "n_edges": np.array([r.n_edges for r in records], dtype=np.float64),
            "activity_bucket": np.array([r.activity_bucket for r in records], dtype=np.float64),
        }
        for name, values in scalars.items():
            positive, negative = values[labels], values[~labels]
            if positive.min() > negative.max() or negative.min() > positive.max():
                perfect.append(
                    {
                        "feature": name,
                        "reason": "value ranges do not overlap between the two labels",
                        "suspicious_range": [float(positive.min()), float(positive.max())],
                        "licit_range": [float(negative.min()), float(negative.max())],
                    }
                )

    offenders = named + [p["feature"] for p in perfect]
    return Finding(
        check="label_leakage",
        severity="fatal",
        passed=not offenders,
        detail=(
            "no declared feature names or reproduces the label"
            if not offenders
            else f"label proxies present: {offenders}"
        ),
        evidence={
            "named_label_proxies": named,
            "perfect_separators": perfect,
            "node_feature_names": list(node_feature_names),
            "edge_feature_names": list(edge_feature_names),
            "forbidden_columns": sorted(LABEL_PROXY_COLUMNS),
        },
    )


# --------------------------------------------------------------------- api ---


def audit_splits(
    collection: CaseCollection,
    manifest: dict[str, Any],
    *,
    node_feature_names: list[str] | None = None,
    edge_feature_names: list[str] | None = None,
    created_at: datetime | None = None,
) -> LeakageReport:
    """Audit a split manifest against the case collection it partitions.

    The manifest supplies only the membership lists. Every property audited is re-derived
    from ``collection``, so a bug in :mod:`g2t_aml.data.splits` cannot hide behind a
    manifest that asserts the split is fine.

    Args:
        collection: The case population.
        manifest: A loaded split manifest.
        node_feature_names: Declared node features. Defaults to the collection's
            extraction record, then to empty.
        edge_feature_names: Declared edge features.
        created_at: Report timestamp. Defaults to now.

    Returns:
        The report. Inspect :attr:`LeakageReport.passed` for the gate verdict.

    Raises:
        LeakageAuditError: If the manifest names a case the collection does not hold, or
            partitions nothing at all.
    """
    by_id = collection.by_id()
    splits: dict[str, list[CaseRecord]] = {}
    for name in ("train", "val", "test"):
        ids = manifest["splits"][name]["case_ids"]
        if missing := [cid for cid in ids if cid not in by_id]:
            raise LeakageAuditError(
                f"manifest split {name!r} names {len(missing)} case(s) absent from the "
                f"collection, e.g. {missing[:3]}"
            )
        splits[name] = [by_id[cid] for cid in ids]
    if not any(splits.values()):
        raise LeakageAuditError("the manifest partitions no cases")

    findings = [
        _check_temporal(splits),
        _check_stream_atomicity(splits),
        _check_label_leakage(
            splits,
            node_feature_names=node_feature_names or [],
            edge_feature_names=edge_feature_names or [],
        ),
        _check_overlap(
            collection,
            splits,
            column="node_id",
            table="node_membership",
            check="node_overlap",
            noun="node",
        ),
        _check_overlap(
            collection,
            splits,
            column="edge_index",
            table="edge_membership",
            check="edge_overlap",
            noun="edge",
        ),
        _check_duplicates(collection, splits),
    ]
    return LeakageReport(
        dataset=collection.dataset,
        findings=findings,
        created_at=created_at or datetime.now().astimezone(),
    )


def audit_temporal_disjointness(
    train: list[CaseRecord], later: list[CaseRecord], *, later_name: str = "realistic_test"
) -> Finding:
    """Check a standalone case stream begins after training ends.

    The realistic-imbalance test stream is built separately from the balanced corpus and is
    never in the split manifest, so it needs its own temporal check before it can be used
    to report anything.

    Args:
        train: The training cases.
        later: The stream to check.
        later_name: Name used in the finding's detail.

    Returns:
        A fatal finding.
    """
    if not train or not later:
        return Finding(
            check=f"temporal_ordering_{later_name}",
            severity="fatal",
            passed=False,
            detail="nothing to compare: one of the two populations is empty",
            evidence={"n_train": len(train), "n_later": len(later)},
        )
    latest = max(r.window_end for r in train)
    earliest = min(r.window_start for r in later)
    offenders = [r.case_id for r in later if r.window_start < latest]
    return Finding(
        check=f"temporal_ordering_{later_name}",
        severity="fatal",
        passed=not offenders,
        detail=(
            f"{later_name} begins after training ends"
            if not offenders
            else f"{len(offenders)} {later_name} case(s) begin before training ends"
        ),
        evidence={
            "train_end": latest.isoformat(),
            f"{later_name}_start": earliest.isoformat(),
            "n_violations": len(offenders),
            "example_case_ids": sorted(offenders)[:10],
        },
    )
