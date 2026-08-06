"""End-to-end Phase 2: extract, sample, split, audit, freeze -- over a synthetic substrate.

The unit tests check each stage in isolation. This one checks that the stages agree: that
the manifest the splitter writes is the manifest the auditor can read, that the audit
passes on a corpus the pipeline itself produced, and that a case can still be materialised
from the frozen artifacts afterwards.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from g2t_aml.data.canonical import AMLWORLD_AVAILABILITY, CanonicalGraph
from g2t_aml.data.case_extraction import ExtractionParams, GraphIndex, TimeWindow
from g2t_aml.data.case_sampling import (
    CaseCollection,
    SamplingParams,
    build_realistic_stream,
    sample_cases,
)
from g2t_aml.data.leakage_audit import audit_splits, audit_temporal_disjointness
from g2t_aml.data.splits import (
    SPLIT_NAMES,
    SplitParams,
    apply_overlap_mode,
    build_manifest,
    load_split_manifest,
    measure_overlap,
    temporal_split,
    write_split_manifest,
)

pytestmark = pytest.mark.integration

T0 = datetime(2022, 9, 1)
TYPOLOGIES = (
    "fan_out",
    "fan_in",
    "gather_scatter",
    "scatter_gather",
    "cycle",
    "random",
    "bipartite",
    "stack",
)


def _substrate() -> CanonicalGraph:
    """A substrate with streams spread across 40 days and varied licit structure."""
    rows: list[tuple[str, str, datetime, float, bool, str | None, str | None]] = []
    for s in range(40):
        typology = TYPOLOGIES[s % len(TYPOLOGIES)]
        pattern_id = f"{typology}_{s:05d}"
        hub = f"B1|LAU-{s:03d}-HUB"
        # Stream width varies over the same range as the licit fans below. A fixture where
        # every laundering stream is the same size makes case size a perfect classifier,
        # which the leakage auditor correctly refuses -- and which would be a corpus
        # defect, not a test-fixture convenience.
        for i in range(2 + (s % 16)):
            rows.append(
                (
                    hub,
                    f"B1|LAU-{s:03d}-{i:02d}",
                    T0 + timedelta(days=s, hours=i),
                    500.0 + i,
                    True,
                    pattern_id,
                    typology,
                )
            )
    for a in range(1500):
        hub = f"B1|LIC-{a:04d}"
        for i in range(2 + (a % 16)):
            rows.append(
                (
                    hub,
                    f"B1|CP-{a:04d}-{i:02d}",
                    T0 + timedelta(days=(a % 40), hours=i % 12),
                    100.0 * (i + 1),
                    False,
                    None,
                    None,
                )
            )

    edges = pl.DataFrame(
        {
            "src": [r[0] for r in rows],
            "dst": [r[1] for r in rows],
            "timestamp": [r[2] for r in rows],
            "amount_paid": [r[3] for r in rows],
            "is_laundering": [r[4] for r in rows],
            "pattern_id": [r[5] for r in rows],
            "typology": [r[6] for r in rows],
        }
    )
    spans = (
        pl.concat(
            [
                edges.select(pl.col("src").alias("node_id"), "timestamp"),
                edges.select(pl.col("dst").alias("node_id"), "timestamp"),
            ]
        )
        .group_by("node_id")
        .agg(
            pl.col("timestamp").min().alias("first_seen"),
            pl.col("timestamp").max().alias("last_seen"),
            pl.len().alias("degree"),
        )
    )
    nodes = (
        spans.with_columns(pl.lit("account").alias("node_type"))
        .select("node_id", "node_type", "degree", "first_seen", "last_seen")
        .sort("node_id")
    )
    return CanonicalGraph(
        graph_id="fixture_substrate",
        dataset="fixture_substrate",
        nodes=nodes,
        edges=edges,
        node_feature_names=["degree"],
        edge_feature_names=["amount_paid"],
        availability=AMLWORLD_AVAILABILITY,
    )


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Run the whole Phase 2 pipeline once, the way scripts/02_build_cases.py does."""
    graph = _substrate()
    index = GraphIndex(graph)
    extraction = ExtractionParams(k_hops=2, n_max=80)
    collection = sample_cases(
        graph,
        index,
        extraction,
        SamplingParams(
            n_cases=1200,
            positive_fraction=0.25,
            hard_negative_fraction=0.25,
            hard_negative_oversample=8.0,
            hard_negative_min_score=0.4,
            max_window_hours=48.0,
        ),
        source_manifest_hash="fixturehash",
    )
    assignment = temporal_split(collection.records, SplitParams(buffer_hours=6.0))
    overlap = measure_overlap(collection, assignment)
    assignment = apply_overlap_mode(assignment, overlap)
    manifest = build_manifest(collection, assignment, overlap)
    report = audit_splits(
        collection,
        manifest,
        node_feature_names=list(graph.node_feature_names),
        edge_feature_names=list(graph.edge_feature_names),
    )
    manifest["leakage_audit"] = report.summary()

    test_records = [collection.by_id()[c] for c in assignment.splits["test"]]
    window = TimeWindow(
        start=min(r.window_start for r in test_records),
        end=max(r.window_end for r in test_records),
    )
    realistic = build_realistic_stream(
        graph, index, extraction, window=window, n_cases=400, seed=4242
    )

    root = tmp_path_factory.mktemp("phase2")
    collection.save(root / "cases")
    realistic.save(root / "cases" / "realistic_test")
    write_split_manifest(manifest, root / "splits")
    report.save(root / "leakage_audit.json")
    return {
        "graph": graph,
        "index": index,
        "collection": collection,
        "assignment": assignment,
        "manifest": manifest,
        "report": report,
        "realistic": realistic,
        "root": root,
    }


# ------------------------------------------------------------------- gate ---


def test_the_leakage_audit_passes_with_no_hard_failures(pipeline):
    report = pipeline["report"]
    assert report.passed, [f"{f.check}: {f.detail}" for f in report.hard_failures]


def test_there_are_zero_temporal_violations(pipeline):
    assert pipeline["report"].summary()["temporal_violations"] == 0


def test_the_node_overlap_rate_is_quantified(pipeline):
    rate = pipeline["report"].summary()["node_overlap_rate"]
    assert rate is not None
    assert 0.0 <= rate <= 1.0


def test_hard_negatives_clear_twenty_percent_in_every_split(pipeline):
    for name in SPLIT_NAMES:
        assert pipeline["manifest"]["stratification"][name]["hard_negative_rate"] >= 0.20


def test_every_split_is_populated(pipeline):
    counts = pipeline["assignment"].counts
    assert all(n > 0 for n in counts.values()), counts


def test_the_realistic_stream_starts_after_training_ends(pipeline):
    train = [pipeline["collection"].by_id()[c] for c in pipeline["assignment"].splits["train"]]
    assert audit_temporal_disjointness(train, pipeline["realistic"].records).passed


def test_the_realistic_stream_is_much_more_licit_than_the_balanced_corpus(pipeline):
    realistic = pipeline["realistic"].stratification["observed_prevalence"]
    balanced = pipeline["collection"].stratification["by_label"].get("suspicious", 0) / len(
        pipeline["collection"]
    )
    assert realistic < balanced


# ---------------------------------------------------------------- artifacts ---


def test_the_frozen_manifest_reloads_and_verifies(pipeline):
    loaded = load_split_manifest(pipeline["root"] / "splits")
    for name in SPLIT_NAMES:
        assert loaded["splits"][name]["case_ids"] == pipeline["assignment"].splits[name]


def test_the_case_corpus_reloads_from_disk(pipeline):
    loaded = CaseCollection.load(pipeline["root"] / "cases")
    assert loaded.case_ids == pipeline["collection"].case_ids


def test_a_case_named_by_the_manifest_materialises_from_the_frozen_corpus(pipeline):
    """The contract every downstream phase relies on: load by id, get a graph back."""
    loaded = CaseCollection.load(pipeline["root"] / "cases")
    manifest = load_split_manifest(pipeline["root"] / "splits")
    case_id = manifest["splits"]["test"]["case_ids"][0]
    case = loaded.materialise(case_id, pipeline["index"])
    assert case.graph_id == case_id
    assert case.num_nodes > 0
    assert case.availability == AMLWORLD_AVAILABILITY


def test_no_label_column_reaches_a_declared_feature_list(pipeline):
    """Invariant: is_laundering must never be a model input."""
    graph = pipeline["graph"]
    assert "is_laundering" not in graph.node_feature_names
    assert "is_laundering" not in graph.edge_feature_names
    assert "typology" not in graph.edge_feature_names
    assert "pattern_id" not in graph.edge_feature_names


def test_the_realistic_stream_is_stored_separately(pipeline):
    assert (pipeline["root"] / "cases" / "realistic_test" / "cases.jsonl").is_file()
    assert (pipeline["root"] / "cases" / "cases.jsonl").is_file()


def test_the_manifest_records_the_extraction_protocol_in_full(pipeline):
    params = pipeline["manifest"]["extraction_params"]
    for key in ("k_hops", "n_max", "prune_rule", "seed", "max_neighbours_per_node"):
        assert key in params, key
