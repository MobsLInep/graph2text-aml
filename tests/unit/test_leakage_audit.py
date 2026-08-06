"""The leakage auditor, tested by planting the leaks it exists to find.

An auditor that has never been shown a real leak is decoration. Every fatal check here is
exercised twice: once against a clean split, once against a split with the corresponding
leak deliberately injected.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from g2t_aml.data.case_sampling import CaseCollection, CaseRecord
from g2t_aml.data.leakage_audit import (
    AUDIT_SCHEMA_VERSION,
    LABEL_PROXY_COLUMNS,
    LeakageAuditError,
    audit_splits,
    audit_temporal_disjointness,
)
from g2t_aml.data.splits import build_manifest, measure_overlap, temporal_split

T0 = datetime(2022, 9, 1)


def _record(
    index,
    *,
    day,
    label="licit",
    case_class="licit",
    pattern_ids=(),
    n_nodes=5,
    motif=0.5,
    structural_hash=None,
) -> CaseRecord:
    start = T0 + timedelta(days=day)
    return CaseRecord(
        case_id=f"fixture-{index:05d}",
        dataset="fixture",
        seed_node=f"B1|ACCT-{index:05d}",
        window_start=start,
        window_end=start + timedelta(hours=6),
        case_class=case_class,
        label=label,
        typology="fan_out" if label == "suspicious" else None,
        pattern_ids=pattern_ids,
        n_nodes=n_nodes,
        n_edges=7,
        activity_bucket=3,
        motif_best="fan_out",
        motif_score=motif,
        structural_hash=structural_hash or f"hash{index:012d}",
    )


def _population(n=120):
    return [
        _record(
            i,
            day=i * 30 / n,
            label="suspicious" if i % 4 == 0 else "licit",
            case_class=("suspicious" if i % 4 == 0 else "hard_negative" if i % 4 == 1 else "licit"),
        )
        for i in range(n)
    ]


def _collection(records, node_map=None, edge_map=None):
    nodes = node_map or {r.case_id: [f"B1|{r.case_id}-{i}" for i in range(3)] for r in records}
    edges = edge_map or {r.case_id: [i] for i, r in enumerate(records)}
    node_rows = [(c, v) for c, vs in nodes.items() for v in vs]
    edge_rows = [(c, v) for c, vs in edges.items() for v in vs]
    return CaseCollection(
        dataset="fixture",
        records=records,
        node_membership=pl.DataFrame(
            {
                "case_id": [r[0] for r in node_rows],
                "node_index": list(range(len(node_rows))),
                "node_id": [r[1] for r in node_rows],
            }
        ),
        edge_membership=pl.DataFrame(
            {"case_id": [r[0] for r in edge_rows], "edge_index": [r[1] for r in edge_rows]}
        ),
        source_manifest_hash="deadbeef",
    )


def _audit(records, *, node_map=None, edge_map=None, **kwargs):
    assignment = temporal_split(records)
    collection = _collection(records, node_map, edge_map)
    manifest = build_manifest(collection, assignment, measure_overlap(collection, assignment))
    return audit_splits(collection, manifest, **kwargs)


def _finding(report, check):
    return next(f for f in report.findings if f.check == check)


# ------------------------------------------------------------ clean split ---


def test_a_clean_split_passes_with_no_hard_failures():
    report = _audit(_population())
    assert report.passed
    assert report.hard_failures == []


def test_the_report_serialises_with_every_finding():
    report = _audit(_population())
    payload = report.to_dict()
    assert payload["audit_schema_version"] == AUDIT_SCHEMA_VERSION
    assert payload["passed"] is True
    checks = {f["check"] for f in payload["findings"]}
    assert {
        "temporal_ordering",
        "stream_atomicity",
        "label_leakage",
        "node_overlap",
        "edge_overlap",
        "duplicate_cases",
    } <= checks


def test_the_summary_is_the_block_that_goes_into_the_manifest():
    summary = _audit(_population()).summary()
    assert summary["temporal_violations"] == 0
    assert 0.0 <= summary["node_overlap_rate"] <= 1.0
    assert summary["passed"] is True


def test_the_report_writes_to_disk(tmp_path):
    path = _audit(_population()).save(tmp_path / "audit.json")
    assert path.is_file()


# --------------------------------------------- injected temporal violation ---


def test_the_auditor_catches_an_injected_temporal_violation():
    """Plant a test case that starts before training ends, and assert it fires."""
    records = _population()
    assignment = temporal_split(records)
    collection = _collection(records)
    manifest = build_manifest(collection, assignment, measure_overlap(collection, assignment))

    # Move a train case into the test split without changing its timestamps.
    stolen = manifest["splits"]["train"].pop("case_ids").copy()
    manifest["splits"]["train"]["case_ids"] = stolen[:-1]
    manifest["splits"]["test"]["case_ids"] = [stolen[-1], *manifest["splits"]["test"]["case_ids"]]

    report = audit_splits(collection, manifest)
    assert not report.passed
    assert "temporal_ordering" in [f.check for f in report.hard_failures]
    assert _finding(report, "temporal_ordering").evidence["violations"] > 0


def test_a_temporal_violation_reports_how_far_it_reaches():
    records = _population()
    assignment = temporal_split(records)
    collection = _collection(records)
    manifest = build_manifest(collection, assignment, measure_overlap(collection, assignment))
    manifest["splits"]["test"]["case_ids"].append(manifest["splits"]["train"]["case_ids"][0])
    detail = _finding(audit_splits(collection, manifest), "temporal_ordering").evidence["detail"]
    assert detail[0]["overlap_hours"] > 0
    assert detail[0]["example_case_ids"]


# ------------------------------------------------ injected stream straddle ---


def test_the_auditor_catches_a_stream_in_two_splits():
    records = _population()
    assignment = temporal_split(records)
    collection = _collection(records)
    manifest = build_manifest(collection, assignment, measure_overlap(collection, assignment))

    # Attach the same stream to one train case and one test case, bypassing the splitter.
    by_id = {r.case_id: r for r in records}
    import dataclasses

    train_id = manifest["splits"]["train"]["case_ids"][0]
    test_id = manifest["splits"]["test"]["case_ids"][0]
    patched = [
        dataclasses.replace(r, pattern_ids=("fan_out_00001",))
        if r.case_id in (train_id, test_id)
        else r
        for r in records
    ]
    collection = _collection(patched)
    report = audit_splits(collection, manifest)
    assert not report.passed
    assert "stream_atomicity" in [f.check for f in report.hard_failures]
    assert _finding(report, "stream_atomicity").evidence["n_straddling"] == 1
    assert by_id  # the fixture is genuinely split, not empty


# ---------------------------------------------------- injected label leak ---


def test_the_auditor_refuses_is_laundering_as_a_feature():
    """The classic way a graph pipeline scores 1.0 and means nothing."""
    report = _audit(_population(), edge_feature_names=["amount_paid", "is_laundering"])
    assert not report.passed
    assert "label_leakage" in [f.check for f in report.hard_failures]
    assert "is_laundering" in _finding(report, "label_leakage").evidence["named_label_proxies"]


@pytest.mark.parametrize("column", sorted(LABEL_PROXY_COLUMNS))
def test_every_label_proxy_column_is_refused(column):
    report = _audit(_population(), node_feature_names=[column])
    assert not report.passed


def test_clean_feature_lists_pass():
    report = _audit(
        _population(),
        node_feature_names=["degree", "total_sent"],
        edge_feature_names=["amount_paid"],
    )
    assert _finding(report, "label_leakage").passed


def test_the_auditor_catches_a_feature_that_perfectly_separates_the_labels():
    """A scalar whose ranges do not overlap is a one-line classifier."""
    records = [
        _record(
            i,
            day=i * 30 / 120,
            label="suspicious" if i % 4 == 0 else "licit",
            n_nodes=999 if i % 4 == 0 else 5,
        )
        for i in range(120)
    ]
    report = _audit(records)
    assert not report.passed
    separators = _finding(report, "label_leakage").evidence["perfect_separators"]
    assert any(s["feature"] == "n_nodes" for s in separators)


# ------------------------------------------------------------- duplicates ---


def test_the_auditor_reports_an_exact_duplicate_across_splits():
    records = _population()
    assignment = temporal_split(records)
    train_id = assignment.splits["train"][0]
    test_id = assignment.splits["test"][0]
    import dataclasses

    patched = [
        dataclasses.replace(r, structural_hash="COLLISION")
        if r.case_id in (train_id, test_id)
        else r
        for r in records
    ]
    collection = _collection(patched)
    manifest = build_manifest(collection, assignment, measure_overlap(collection, assignment))
    report = audit_splits(collection, manifest)
    duplicates = _finding(report, "duplicate_cases")
    assert duplicates.evidence["n_exact"] == 1
    # Reported, not fatal: a structurally identical case on a dense graph is a real
    # property of the substrate. It must be visible, not silently blocking.
    assert duplicates.severity == "report"
    assert report.passed


def test_the_auditor_finds_near_duplicates_by_node_overlap():
    records = _population()
    assignment = temporal_split(records)
    train_id = assignment.splits["train"][0]
    test_id = assignment.splits["test"][0]
    shared = [f"B1|SHARED-{i}" for i in range(10)]
    node_map = {r.case_id: [f"B1|{r.case_id}-{i}" for i in range(3)] for r in records}
    node_map[train_id] = shared
    node_map[test_id] = shared
    collection = _collection(records, node_map)
    manifest = build_manifest(collection, assignment, measure_overlap(collection, assignment))
    assert _finding(audit_splits(collection, manifest), "duplicate_cases").evidence["n_near"] >= 1


# ---------------------------------------------------------------- overlap ---


def test_node_overlap_is_reported_and_never_fatal():
    records = _population()
    shared = {r.case_id: ["B1|SHARED"] for r in records}
    report = _audit(records, node_map=shared)
    overlap = _finding(report, "node_overlap")
    assert overlap.evidence["rate"] == 1.0
    assert overlap.severity == "report"
    assert report.passed


def test_edge_overlap_notices_the_same_transaction_on_both_sides():
    records = _population()
    edges = {r.case_id: [0] for r in records}
    report = _audit(records, edge_map=edges)
    assert _finding(report, "edge_overlap").evidence["rate"] == 1.0


# ------------------------------------------------------------- interfaces ---


def test_auditing_a_manifest_naming_unknown_cases_is_an_error():
    records = _population()
    assignment = temporal_split(records)
    collection = _collection(records)
    manifest = build_manifest(collection, assignment, measure_overlap(collection, assignment))
    manifest["splits"]["test"]["case_ids"].append("fixture-99999")
    with pytest.raises(LeakageAuditError, match="absent from the collection"):
        audit_splits(collection, manifest)


def test_the_standalone_stream_check_passes_when_it_starts_later():
    train = [_record(i, day=i * 0.1) for i in range(10)]
    later = [_record(100 + i, day=20 + i * 0.1) for i in range(10)]
    assert audit_temporal_disjointness(train, later).passed


def test_the_standalone_stream_check_fires_when_it_does_not():
    train = [_record(i, day=10 + i * 0.1) for i in range(10)]
    later = [_record(100 + i, day=i * 0.1) for i in range(10)]
    finding = audit_temporal_disjointness(train, later)
    assert not finding.passed
    assert finding.evidence["n_violations"] == 10


def test_the_standalone_stream_check_fails_on_an_empty_population():
    assert not audit_temporal_disjointness([], [_record(0, day=1.0)]).passed
