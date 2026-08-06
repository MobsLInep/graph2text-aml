"""Case sampling: stratification, activity matching, and the hard-negative population.

The hard-negative population is the one the paper's central claim rests on, so it gets the
most attention here: it must actually be hard (high motif score, asserted against the
detector), it must actually be negative, and it must clear the 20% gate.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from g2t_aml.data.canonical import AMLWORLD_AVAILABILITY, CanonicalGraph
from g2t_aml.data.case_extraction import ExtractionParams, GraphIndex, TimeWindow
from g2t_aml.data.case_sampling import (
    ACTIVITY_BUCKETS,
    MINIMUM_HARD_NEGATIVE_RATE,
    POSITIVE_STRATA,
    CaseCollection,
    CaseSamplingError,
    SamplingParams,
    _allocate_evenly,
    bounded_window,
    build_realistic_stream,
    positive_seeds,
    sample_cases,
    summarise_stratification,
)
from g2t_aml.data.motifs import score_edges

T0 = datetime(2022, 9, 1)


def _substrate(
    *,
    n_streams: int = 12,
    stream_size: int = 6,
    n_licit: int = 900,
    licit_fan: int = 16,
) -> CanonicalGraph:
    """A synthetic substrate with laundering streams and structured licit traffic.

    The licit side is deliberately *structured* — fans and chains — so hard-negative
    mining has something real to find. Invariant 8: synthetic identifiers only.
    """
    src: list[str] = []
    dst: list[str] = []
    stamps: list[datetime] = []
    amounts: list[float] = []
    flags: list[bool] = []
    patterns: list[str | None] = []
    typologies: list[str | None] = []

    for s in range(n_streams):
        typology = POSITIVE_STRATA[s % 8]
        pattern_id = f"{typology}_{s:05d}"
        hub = f"B1|LAU-{s:03d}-HUB"
        for i in range(stream_size):
            src.append(hub)
            dst.append(f"B1|LAU-{s:03d}-{i:02d}")
            stamps.append(T0 + timedelta(days=s * 2, hours=i))
            amounts.append(500.0 + i)
            flags.append(True)
            patterns.append(pattern_id)
            typologies.append(typology)

    for a in range(n_licit):
        hub = f"B1|LIC-{a:04d}"
        # Fan width varies so licit cases span the motif score range: a fixture where
        # every negative has the same shape cannot show that mining discriminates.
        for i in range(2 + (a % licit_fan)):
            src.append(hub)
            dst.append(f"B1|CP-{a:04d}-{i:02d}")
            stamps.append(T0 + timedelta(days=(a % 24), hours=i % 12))
            amounts.append(100.0 * (i + 1))
            flags.append(False)
            patterns.append(None)
            typologies.append(None)

    edges = pl.DataFrame(
        {
            "src": src,
            "dst": dst,
            "timestamp": stamps,
            "amount_paid": amounts,
            "is_laundering": flags,
            "pattern_id": patterns,
            "typology": typologies,
        }
    )
    node_ids = sorted(set(src) | set(dst))
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
        pl.DataFrame({"node_id": node_ids})
        .join(spans, on="node_id", how="left")
        .with_columns(pl.lit("account").alias("node_type"))
        .select("node_id", "node_type", "degree", "first_seen", "last_seen")
        .sort("node_id")
    )
    return CanonicalGraph(
        graph_id="fixture",
        dataset="fixture",
        nodes=nodes,
        edges=edges,
        node_feature_names=["degree"],
        edge_feature_names=["amount_paid"],
        availability=AMLWORLD_AVAILABILITY,
    )


@pytest.fixture(scope="module")
def substrate() -> CanonicalGraph:
    return _substrate()


@pytest.fixture(scope="module")
def index(substrate) -> GraphIndex:
    return GraphIndex(substrate)


@pytest.fixture(scope="module")
def collection(substrate, index) -> CaseCollection:
    return sample_cases(
        substrate,
        index,
        ExtractionParams(k_hops=2, n_max=60),
        SamplingParams(
            n_cases=600,
            positive_fraction=0.25,
            hard_negative_fraction=0.25,
            hard_negative_oversample=6.0,
            hard_negative_min_score=0.4,
            max_window_hours=48.0,
        ),
        source_manifest_hash="fixturehash",
    )


# ---------------------------------------------------------------- windows ---


def test_a_short_activity_span_keeps_its_full_extent():
    window = bounded_window(T0, T0 + timedelta(hours=4), T0 + timedelta(hours=2), 6.0, 96.0)
    assert window.start == T0 - timedelta(hours=6)
    assert window.end == T0 + timedelta(hours=10)


def test_a_long_activity_span_is_capped_and_centred():
    centre = T0 + timedelta(days=5)
    window = bounded_window(T0, T0 + timedelta(days=10), centre, 12.0, 48.0)
    assert window.duration == timedelta(hours=48)
    assert window.start == centre - timedelta(hours=24)


def test_no_cap_means_no_cap():
    window = bounded_window(T0, T0 + timedelta(days=10), T0, 0.0, None)
    assert window.duration == timedelta(days=10)


# ------------------------------------------------------------- allocation ---


def test_even_allocation_gives_every_stratum_the_same_share():
    allocation = _allocate_evenly(90, {"a": 100, "b": 100, "c": 100})
    assert allocation == {"a": 30, "b": 30, "c": 30}


def test_a_stratum_that_cannot_fill_its_share_spills_onto_the_others():
    allocation = _allocate_evenly(90, {"a": 5, "b": 100, "c": 100})
    assert allocation["a"] == 5
    assert sum(allocation.values()) == 90


def test_allocation_never_exceeds_capacity():
    allocation = _allocate_evenly(1000, {"a": 5, "b": 7})
    assert allocation == {"a": 5, "b": 7}


# ------------------------------------------------------------ positive seeds ---


def test_positive_seeds_come_from_every_typology_present(substrate):
    seeds = positive_seeds(substrate, SamplingParams(), np.random.default_rng(0))
    typologies = {s.typology for s in seeds}
    assert len(typologies) >= 8
    assert all(s.case_class == "suspicious" for s in seeds)


def test_positive_seeds_carry_their_stream(substrate):
    seeds = positive_seeds(substrate, SamplingParams(), np.random.default_rng(0))
    patterned = [s for s in seeds if s.pattern_ids]
    assert patterned
    assert all(s.pattern_ids[0].startswith(s.typology) for s in patterned)


def test_positive_sampling_needs_laundering_ground_truth():
    graph = _substrate(n_streams=1)
    graph.edges = graph.edges.drop("is_laundering")
    with pytest.raises(CaseSamplingError, match="is_laundering"):
        positive_seeds(graph, SamplingParams(), np.random.default_rng(0))


def test_seeds_per_stream_are_capped(substrate):
    seeds = positive_seeds(
        substrate, SamplingParams(max_seeds_per_stream=2), np.random.default_rng(0)
    )
    from collections import Counter

    counts = Counter(s.pattern_ids[0] for s in seeds if s.pattern_ids)
    assert max(counts.values()) <= 2


# ---------------------------------------------------------------- corpus ----


def test_the_corpus_holds_all_three_populations(collection):
    classes = collection.stratification["by_class"]
    assert classes["suspicious"] > 0
    assert classes["licit"] > 0
    assert classes["hard_negative"] > 0


def test_hard_negatives_clear_the_twenty_percent_gate(collection):
    assert collection.stratification["hard_negative_rate"] >= MINIMUM_HARD_NEGATIVE_RATE


def test_hard_negatives_are_negative(collection):
    """A hard negative is a *licit* case that looks suspicious, not a third label."""
    hard = [r for r in collection.records if r.case_class == "hard_negative"]
    assert hard
    assert all(r.label == "licit" for r in hard)
    assert all(r.typology is None for r in hard)


def test_hard_negatives_actually_score_high_on_the_motif_detector(collection, index):
    """The claim that makes them 'hard', asserted against the detector itself."""
    hard = [r for r in collection.records if r.case_class == "hard_negative"]
    easy = [r for r in collection.records if r.case_class == "licit"]
    assert min(r.motif_score for r in hard) >= 0.4
    assert np.mean([r.motif_score for r in hard]) > np.mean([r.motif_score for r in easy])

    # Re-scored from the stored subgraph, not merely trusting the recorded number.
    sample = hard[0]
    rescored = score_edges(collection.materialise(sample.case_id, index).edges)
    assert rescored.best_score == sample.motif_score


def test_positive_cases_carry_a_typology_from_the_vocabulary(collection):
    positives = [r for r in collection.records if r.case_class == "suspicious"]
    assert positives
    assert all(r.typology in POSITIVE_STRATA for r in positives)


def test_positives_are_spread_across_typologies_not_concentrated(collection):
    by_typology = collection.stratification["by_typology"]
    represented = {k: v for k, v in by_typology.items() if k != "none"}
    assert len(represented) >= 6
    largest = max(represented.values())
    total = sum(represented.values())
    assert largest / total < 0.5


def test_negatives_are_activity_matched_to_positives(collection):
    """Without this a classifier wins on transaction count alone."""
    positives = [r.activity_bucket for r in collection.records if r.label == "suspicious"]
    negatives = [r.activity_bucket for r in collection.records if r.label == "licit"]
    assert abs(np.mean(positives) - np.mean(negatives)) < 2.0


def test_negatives_reuse_positive_windows_so_time_carries_no_signal(collection):
    positive_windows = {
        (r.window_start, r.window_end) for r in collection.records if r.label == "suspicious"
    }
    negative_windows = {
        (r.window_start, r.window_end) for r in collection.records if r.label == "licit"
    }
    assert negative_windows <= positive_windows


def test_a_hard_negative_share_below_the_gate_is_refused(substrate, index):
    with pytest.raises(CaseSamplingError, match="gate criterion"):
        sample_cases(
            substrate,
            index,
            ExtractionParams(k_hops=2, n_max=60),
            SamplingParams(
                n_cases=400,
                positive_fraction=0.25,
                hard_negative_fraction=0.45,
                hard_negative_oversample=1.0,
                hard_negative_min_score=0.99,
                max_window_hours=48.0,
            ),
        )


def test_the_gate_cannot_be_configured_away():
    with pytest.raises(ValueError, match="gate criterion"):
        SamplingParams(hard_negative_fraction=0.05)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_cases": 0}, "n_cases"),
        ({"positive_fraction": 1.5}, "positive_fraction"),
        ({"hard_negative_oversample": 0.5}, "oversample"),
        ({"hard_negative_min_score": 2.0}, "min_score"),
        ({"max_window_hours": -1.0}, "max_window_hours"),
        ({"max_stratum_share": 0.01}, "max_stratum_share"),
    ],
)
def test_sampling_params_reject_nonsense(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SamplingParams(**kwargs)


# ------------------------------------------------------------ determinism ---


def test_sampling_is_reproducible_from_its_seed(substrate, index):
    params = SamplingParams(
        n_cases=300,
        positive_fraction=0.3,
        max_window_hours=48.0,
        hard_negative_min_score=0.4,
        hard_negative_oversample=6.0,
    )
    first = sample_cases(substrate, index, ExtractionParams(n_max=60), params)
    second = sample_cases(substrate, index, ExtractionParams(n_max=60), params)
    assert first.case_ids == second.case_ids


def test_a_different_seed_gives_a_different_corpus(substrate, index):
    base = SamplingParams(
        n_cases=300,
        positive_fraction=0.3,
        max_window_hours=48.0,
        hard_negative_min_score=0.4,
        hard_negative_oversample=6.0,
    )
    first = sample_cases(substrate, index, ExtractionParams(n_max=60), base)
    import dataclasses

    other = sample_cases(
        substrate, index, ExtractionParams(n_max=60), dataclasses.replace(base, seed=99)
    )
    assert first.case_ids != other.case_ids


# ------------------------------------------------------------------- i/o ----


def test_the_collection_round_trips_through_disk(collection, tmp_path):
    collection.save(tmp_path)
    loaded = CaseCollection.load(tmp_path)
    assert loaded.case_ids == collection.case_ids
    assert loaded.dataset == collection.dataset
    assert loaded.source_manifest_hash == collection.source_manifest_hash
    assert loaded.stratification == collection.stratification


def test_a_case_materialises_back_into_a_full_graph(collection, index):
    case_id = collection.case_ids[0]
    case = collection.materialise(case_id, index)
    record = collection.by_id()[case_id]
    assert case.num_nodes == record.n_nodes
    assert case.num_edges == record.n_edges
    assert case.availability == AMLWORLD_AVAILABILITY
    assert case.label == record.label


def test_materialising_against_the_wrong_graph_is_refused(collection, index):
    collection.dataset = "some_other_substrate"
    with pytest.raises(CaseSamplingError, match="not portable"):
        collection.materialise(collection.case_ids[0], index)
    collection.dataset = "fixture"


def test_a_subset_recomputes_its_own_stratification(collection):
    subset = collection.subset(collection.case_ids[:50])
    assert len(subset) == 50
    assert subset.stratification["n_cases"] == 50
    assert subset.node_membership["case_id"].n_unique() <= 50


# ------------------------------------------------------ realistic stream ----


def test_the_realistic_stream_records_the_prevalence_it_observed(substrate, index):
    stream = build_realistic_stream(
        substrate,
        index,
        ExtractionParams(k_hops=2, n_max=60),
        window=TimeWindow(T0, T0 + timedelta(days=25)),
        n_cases=300,
        seed=7,
    )
    assert len(stream) > 0
    assert 0.0 <= stream.stratification["observed_prevalence"] <= 1.0
    assert stream.stratification["target_prevalence"] is None
    assert "uniform" in stream.stratification["sampling"]


def test_the_realistic_stream_can_be_down_sampled_to_a_target(substrate, index):
    stream = build_realistic_stream(
        substrate,
        index,
        ExtractionParams(k_hops=2, n_max=60),
        window=TimeWindow(T0, T0 + timedelta(days=25)),
        n_cases=300,
        seed=7,
        target_prevalence=0.01,
    )
    observed = sum(1 for r in stream.records if r.label == "suspicious") / len(stream)
    assert observed <= 0.02
    assert stream.stratification["target_prevalence"] == 0.01


def test_the_realistic_stream_is_far_more_licit_than_the_balanced_corpus(
    substrate, index, collection
):
    stream = build_realistic_stream(
        substrate,
        index,
        ExtractionParams(k_hops=2, n_max=60),
        window=TimeWindow(T0, T0 + timedelta(days=25)),
        n_cases=400,
        seed=7,
    )
    balanced = collection.stratification["by_label"].get("suspicious", 0) / len(collection)
    assert stream.stratification["observed_prevalence"] < balanced


def test_an_empty_window_is_an_error(substrate, index):
    with pytest.raises(CaseSamplingError, match="no account is active"):
        build_realistic_stream(
            substrate,
            index,
            ExtractionParams(),
            window=TimeWindow(T0 + timedelta(days=900), T0 + timedelta(days=901)),
            n_cases=10,
            seed=1,
        )


# -------------------------------------------------------- stratification ----


def test_summarising_an_empty_population_does_not_divide_by_zero():
    assert summarise_stratification([])["hard_negative_rate"] == 0.0


def test_activity_buckets_are_bounded(index):
    from g2t_aml.data.case_sampling import _activity_buckets

    buckets = _activity_buckets(index)
    assert buckets.min() >= 0
    assert buckets.max() < ACTIVITY_BUCKETS
