"""Temporal splits: no leakage forward in time, no stream on both sides, frozen manifests.

Every fixture here has a known ordering, so a violation is a definite failure rather than
a statistical one.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from g2t_aml.data.case_sampling import CaseCollection, CaseRecord
from g2t_aml.data.splits import (
    MANIFEST_VERSION,
    SPLIT_NAMES,
    SplitError,
    SplitParams,
    apply_overlap_mode,
    build_manifest,
    load_split_manifest,
    measure_overlap,
    split_of,
    temporal_split,
    write_split_manifest,
)

T0 = datetime(2022, 9, 1)


def _record(
    index: int,
    *,
    day: float,
    duration_hours: float = 6.0,
    label: str = "licit",
    case_class: str = "licit",
    pattern_ids: tuple[str, ...] = (),
    nodes: tuple[str, ...] = (),
) -> CaseRecord:
    start = T0 + timedelta(days=day)
    return CaseRecord(
        case_id=f"fixture-{index:05d}",
        dataset="fixture",
        seed_node=f"B1|ACCT-{index:05d}",
        window_start=start,
        window_end=start + timedelta(hours=duration_hours),
        case_class=case_class,
        label=label,
        typology="fan_out" if label == "suspicious" else None,
        pattern_ids=pattern_ids,
        n_nodes=len(nodes) or 5,
        n_edges=7,
        activity_bucket=3,
        motif_best="fan_out",
        motif_score=0.6,
        structural_hash=f"hash{index:012d}",
    )


def _population(n: int = 120, duration_hours: float = 6.0) -> list[CaseRecord]:
    """Cases spread evenly over 30 days, with a known temporal ordering."""
    return [
        _record(
            i,
            day=i * 30 / n,
            duration_hours=duration_hours,
            label="suspicious" if i % 4 == 0 else "licit",
            case_class=("suspicious" if i % 4 == 0 else "hard_negative" if i % 4 == 1 else "licit"),
        )
        for i in range(n)
    ]


def _collection(records: list[CaseRecord], node_map: dict[str, list[str]] | None = None):
    """Wrap records in a collection, with per-case node membership."""
    mapping = node_map or {
        r.case_id: [f"B1|N{r.case_id[-3:]}-{i}" for i in range(3)] for r in records
    }
    rows = [(cid, node) for cid, nodes in mapping.items() for node in nodes]
    return CaseCollection(
        dataset="fixture",
        records=records,
        node_membership=pl.DataFrame(
            {
                "case_id": [r[0] for r in rows],
                "node_index": list(range(len(rows))),
                "node_id": [r[1] for r in rows],
            }
        ),
        edge_membership=pl.DataFrame(
            {"case_id": [r.case_id for r in records], "edge_index": list(range(len(records)))}
        ),
        source_manifest_hash="deadbeef",
    )


# --------------------------------------------------------------- ordering ---


def test_the_split_has_zero_temporal_violations():
    """The gate: every test case begins after every train case ends."""
    assignment = temporal_split(_population())
    by_id = {r.case_id: r for r in _population()}
    train_end = max(by_id[c].window_end for c in assignment.splits["train"])
    val_start = min(by_id[c].window_start for c in assignment.splits["val"])
    val_end = max(by_id[c].window_end for c in assignment.splits["val"])
    test_start = min(by_id[c].window_start for c in assignment.splits["test"])
    assert val_start > train_end
    assert test_start > val_end


def test_every_split_is_populated_and_ordered():
    assignment = temporal_split(_population())
    assert set(assignment.splits) == set(SPLIT_NAMES)
    assert all(count > 0 for count in assignment.counts.values())
    assert assignment.boundaries[0] < assignment.boundaries[1]


def test_a_case_straddling_a_boundary_is_dropped_with_a_reason():
    records = _population()
    assignment = temporal_split(records, SplitParams(buffer_hours=0.0))
    first, second = assignment.boundaries
    by_id = {r.case_id: r for r in records}
    for case_id, reason in assignment.dropped.items():
        if reason == "straddles_boundary":
            window = by_id[case_id].window
            assert window.straddles(first) or window.straddles(second)


def test_no_retained_case_crosses_a_boundary():
    records = _population()
    assignment = temporal_split(records)
    first, second = assignment.boundaries
    by_id = {r.case_id: r for r in records}
    for ids in assignment.splits.values():
        for case_id in ids:
            window = by_id[case_id].window
            assert not window.straddles(first)
            assert not window.straddles(second)


def test_the_buffer_widens_the_gap_and_costs_cases():
    records = _population()
    narrow = temporal_split(records, SplitParams(buffer_hours=0.0))
    wide = temporal_split(records, SplitParams(buffer_hours=24.0))
    assert len(wide.dropped) > len(narrow.dropped)
    assert "within_buffer" in wide.drop_reasons()


def test_boundaries_land_on_the_snap_grid():
    assignment = temporal_split(_population(), SplitParams(boundary_snap_hours=24.0))
    origin = min(r.window_start for r in _population())
    for boundary in assignment.boundaries:
        offset = (boundary - origin).total_seconds()
        assert abs(offset % 86400) < 1


# ---------------------------------------------------------- stream atomicity ---


def test_a_stream_never_appears_in_two_splits():
    """The most expensive possible leak: the same laundering event on both sides."""
    records = _population()
    # Plant one stream at both ends of the timeline.
    records[2] = _record(2, day=0.5, label="suspicious", pattern_ids=("fan_out_00001",))
    records[-3] = _record(900, day=29.0, label="suspicious", pattern_ids=("fan_out_00001",))
    assignment = temporal_split(records)
    homes = {
        pattern: {
            name
            for name, ids in assignment.splits.items()
            for cid in ids
            for pattern in {r.case_id: r for r in records}[cid].pattern_ids
        }
        for pattern in ("fan_out_00001",)
    }
    assert all(len(names) <= 1 for names in homes.values())
    assert "stream_in_earlier_split" in assignment.drop_reasons()


# ------------------------------------------------------------- parameters ---


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"proportions": (0.5, 0.5)}, "proportions"),
        ({"proportions": (0.5, 0.5, 0.5)}, "sum to 1"),
        ({"proportions": (1.0, 0.0, 0.0)}, "positive"),
        ({"buffer_hours": -1.0}, "buffer_hours"),
        ({"overlap_mode": "loose"}, "overlap_mode"),
        ({"mode": "random"}, "only the temporal split"),
    ],
)
def test_split_params_reject_nonsense(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SplitParams(**kwargs)


def test_a_random_split_cannot_be_requested_at_all():
    """Invariant 2, enforced at the type level rather than by convention."""
    with pytest.raises(ValueError, match="invariant 2"):
        SplitParams(mode="random")


def test_too_few_cases_to_split_is_an_error():
    with pytest.raises(SplitError, match="cannot split"):
        temporal_split([_record(0, day=0.0)])


def test_windows_too_wide_for_the_span_fail_loudly():
    """A case window wider than a split band cannot be placed; say so rather than guess."""
    records = [_record(i, day=i * 0.1, duration_hours=24 * 20) for i in range(30)]
    with pytest.raises(SplitError, match="wide relative to"):
        temporal_split(records)


# ---------------------------------------------------------------- overlap ---


def test_node_overlap_is_measured_not_assumed():
    records = _population()
    assignment = temporal_split(records)
    shared = {r.case_id: ["B1|SHARED"] for r in records}
    overlap = measure_overlap(_collection(records, shared), assignment)
    assert overlap.node_overlap_rate == 1.0
    assert overlap.mode == "report"


def test_disjoint_cases_report_zero_overlap():
    records = _population()
    assignment = temporal_split(records)
    unique = {r.case_id: [f"B1|{r.case_id}"] for r in records}
    overlap = measure_overlap(_collection(records, unique), assignment)
    assert overlap.node_overlap_rate == 0.0


def test_report_mode_keeps_overlapping_test_cases():
    records = _population()
    assignment = temporal_split(records)
    collection = _collection(records, {r.case_id: ["B1|SHARED"] for r in records})
    overlap = measure_overlap(collection, assignment)
    after = apply_overlap_mode(assignment, overlap)
    assert after.counts == assignment.counts


def test_strict_mode_drops_them():
    records = _population()
    assignment = temporal_split(records, SplitParams(overlap_mode="strict"))
    collection = _collection(records, {r.case_id: ["B1|SHARED"] for r in records})
    overlap = measure_overlap(collection, assignment)
    after = apply_overlap_mode(assignment, overlap)
    assert after.counts["test"] == 0
    assert set(after.dropped.values()) >= {"node_overlap_strict"}


# --------------------------------------------------------------- manifest ---


def test_the_manifest_carries_everything_a_reviewer_needs(tmp_path):
    records = _population()
    assignment = temporal_split(records)
    collection = _collection(records)
    manifest = build_manifest(collection, assignment, measure_overlap(collection, assignment))

    assert manifest["manifest_version"] == MANIFEST_VERSION
    assert manifest["dataset"] == "fixture"
    assert manifest["split_params"]["mode"] == "temporal"
    assert "boundaries" in manifest["split_params"]
    for name in SPLIT_NAMES:
        block = manifest["splits"][name]
        assert block["n"] == len(block["case_ids"])
        assert len(block["id_list_sha256"]) == 64
    assert "hard_negative_rate" in manifest["stratification"]["overall"]
    assert manifest["dropped"]["n"] == len(assignment.dropped)


def test_the_manifest_round_trips_and_verifies_its_hashes(tmp_path):
    records = _population()
    assignment = temporal_split(records)
    collection = _collection(records)
    manifest = build_manifest(collection, assignment, measure_overlap(collection, assignment))
    write_split_manifest(manifest, tmp_path)

    loaded = load_split_manifest(tmp_path)
    assert loaded["splits"]["train"]["case_ids"] == manifest["splits"]["train"]["case_ids"]
    assert len(split_of(loaded)) == sum(assignment.counts.values())


def test_committed_id_lists_are_written_alongside_the_manifest(tmp_path):
    """D-006: the literal ID lists are what makes a split reviewable in a diff."""
    records = _population()
    assignment = temporal_split(records)
    collection = _collection(records)
    write_split_manifest(
        build_manifest(collection, assignment, measure_overlap(collection, assignment)),
        tmp_path,
    )
    for name in SPLIT_NAMES:
        listed = (tmp_path / f"{name}.txt").read_text().split()
        assert listed == assignment.splits[name]
        assert (tmp_path / f"{name}.sha256.json").is_file()


def test_a_hand_edited_manifest_is_rejected(tmp_path):
    """The hash is what makes a committed split a promise rather than a suggestion."""
    records = _population()
    assignment = temporal_split(records)
    collection = _collection(records)
    manifest = build_manifest(collection, assignment, measure_overlap(collection, assignment))
    manifest["splits"]["test"]["case_ids"].append("fixture-99999")
    write_split_manifest(manifest, tmp_path)
    with pytest.raises(SplitError, match="does not match its recorded sha256"):
        load_split_manifest(tmp_path)


def test_a_manifest_from_a_future_version_is_rejected(tmp_path):
    records = _population()
    assignment = temporal_split(records)
    collection = _collection(records)
    manifest = build_manifest(collection, assignment, measure_overlap(collection, assignment))
    manifest["manifest_version"] = "99.0.0"
    write_split_manifest(manifest, tmp_path)
    with pytest.raises(SplitError, match="version mismatch"):
        load_split_manifest(tmp_path)


# --------------------------------------------------------- stratification ---


def test_stratification_proportions_stay_within_tolerance_of_the_targets():
    """The population is 25% suspicious and 25% hard negative by construction."""
    records = _population(n=400)
    assignment = temporal_split(records)
    collection = _collection(records)
    manifest = build_manifest(collection, assignment, measure_overlap(collection, assignment))
    overall = manifest["stratification"]["overall"]
    suspicious = overall["by_label"].get("suspicious", 0) / overall["n_cases"]
    assert abs(suspicious - 0.25) < 0.10
    assert overall["hard_negative_rate"] > 0.20
