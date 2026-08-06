"""The Gold sample: stratification, the hard-negative floor, and honest deficits."""

from __future__ import annotations

import pytest

from g2t_aml.human.sampling import (
    SIZE_BUCKETS,
    TYPED_TYPOLOGIES,
    GoldCandidate,
    GoldSamplingError,
    GoldSamplingParams,
    sample_gold_cases,
    size_bucket_of,
)


def candidate(
    n: int,
    typology: str = "unclassified",
    case_class: str = "licit",
    n_nodes: int = 6,
    dataset: str = "amlworld_hi_small",
    split: str = "test",
) -> GoldCandidate:
    return GoldCandidate(
        case_id=f"{dataset}-{n:08x}",
        dataset=dataset,
        split=split,
        typology=typology,
        case_class=case_class,
        n_nodes=n_nodes,
        n_edges=n_nodes * 2,
    )


def population(
    *, per_typology: int = 40, n_hard: int = 300, n_licit: int = 300
) -> list[GoldCandidate]:
    """A population resembling the real AMLworld test split's shape."""
    out: list[GoldCandidate] = []
    n = 0
    sizes = [3, 8, 30]
    for typology in TYPED_TYPOLOGIES:
        for i in range(per_typology):
            n += 1
            out.append(candidate(n, typology, "suspicious", sizes[i % 3]))
    for i in range(n_hard):
        n += 1
        out.append(candidate(n, "unclassified", "hard_negative", sizes[i % 3]))
    for i in range(n_licit):
        n += 1
        out.append(candidate(n, "unclassified", "licit", sizes[i % 3]))
    return out


# ------------------------------------------------------------------ buckets ---


def test_size_buckets_partition_everything_above_one_node():
    for n_nodes in range(2, 200):
        assert size_bucket_of(n_nodes) in {name for name, _, _ in SIZE_BUCKETS}


def test_a_single_account_case_has_no_bucket():
    """A one-account case has no counterparty in scope and nothing to narrate (D-038)."""
    with pytest.raises(GoldSamplingError, match="below the smallest"):
        size_bucket_of(1)


# ----------------------------------------------------------------- sampling ---


def test_sample_meets_its_requested_size():
    sample = sample_gold_cases(
        population(), GoldSamplingParams(n_cases=350, substrate_shares={"amlworld_hi_small": 1.0})
    )
    assert len(sample) == 350


def test_hard_negative_floor_is_met():
    params = GoldSamplingParams(n_cases=350, substrate_shares={"amlworld_hi_small": 1.0})
    sample = sample_gold_cases(population(), params)
    assert sample.hard_negative_rate >= params.hard_negative_floor


def test_every_typology_is_represented():
    sample = sample_gold_cases(
        population(), GoldSamplingParams(n_cases=350, substrate_shares={"amlworld_hi_small": 1.0})
    )
    for typology in (*TYPED_TYPOLOGIES, "unclassified"):
        assert sample.by_typology.get(typology, 0) > 0, typology


def test_typed_typologies_are_balanced_within_one_of_each_other():
    sample = sample_gold_cases(
        population(), GoldSamplingParams(n_cases=350, substrate_shares={"amlworld_hi_small": 1.0})
    )
    counts = [sample.by_typology[t] for t in TYPED_TYPOLOGIES]
    assert max(counts) - min(counts) <= 2, counts


def test_a_capped_typology_gives_its_remainder_back_rather_than_shrinking_the_sample():
    """`stack` has 19 cases in the real test split; the sample must still reach its size."""
    pool = [c for c in population() if not (c.typology == "stack" and int(c.case_id[-8:], 16) % 7)]
    scarce = [c for c in pool if c.typology == "stack"]
    assert len(scarce) < 40
    sample = sample_gold_cases(
        pool, GoldSamplingParams(n_cases=350, substrate_shares={"amlworld_hi_small": 1.0})
    )
    assert len(sample) == 350
    assert sample.by_typology["stack"] == len(scarce) or sample.by_typology["stack"] > 0


def test_every_size_bucket_is_represented():
    sample = sample_gold_cases(
        population(), GoldSamplingParams(n_cases=350, substrate_shares={"amlworld_hi_small": 1.0})
    )
    assert set(sample.by_size_bucket) == {name for name, _, _ in SIZE_BUCKETS}


def test_sampling_is_deterministic_under_a_seed():
    params = GoldSamplingParams(n_cases=200, substrate_shares={"amlworld_hi_small": 1.0}, seed=7)
    first = sample_gold_cases(population(), params)
    second = sample_gold_cases(list(reversed(population())), params)
    assert first.case_ids == second.case_ids


def test_a_different_seed_selects_a_different_sample():
    a = sample_gold_cases(
        population(),
        GoldSamplingParams(n_cases=200, substrate_shares={"amlworld_hi_small": 1.0}, seed=1),
    )
    b = sample_gold_cases(
        population(),
        GoldSamplingParams(n_cases=200, substrate_shares={"amlworld_hi_small": 1.0}, seed=2),
    )
    assert a.case_ids != b.case_ids


def test_cases_outside_the_split_are_never_selected():
    pool = [*population(), candidate(99001, split="train"), candidate(99002, split="val")]
    sample = sample_gold_cases(
        pool,
        GoldSamplingParams(
            n_cases=100, min_reserved=100, substrate_shares={"amlworld_hi_small": 1.0}
        ),
    )
    assert all(c.split == "test" for c in sample.selected)


def test_single_account_cases_are_excluded_and_the_exclusion_is_reported():
    pool = [*population(), candidate(99003, n_nodes=1)]
    sample = sample_gold_cases(
        pool,
        GoldSamplingParams(
            n_cases=100, min_reserved=100, substrate_shares={"amlworld_hi_small": 1.0}
        ),
    )
    assert all(c.n_nodes >= 2 for c in sample.selected)
    assert "excluded:below_min_nodes" in sample.deficits


# ----------------------------------------------------------------- deficits ---


def test_an_absent_substrate_is_reported_as_a_deficit_not_silently_dropped():
    """Elliptic2's 30% quota is unobtainable; the sample must say so."""
    sample = sample_gold_cases(
        population(),
        GoldSamplingParams(
            n_cases=350, substrate_shares={"amlworld_hi_small": 0.7, "elliptic2": 0.3}
        ),
    )
    assert "dataset:elliptic2" in sample.deficits
    assert sample.deficits["dataset:elliptic2"][1] == 0


def test_the_deficit_is_reallocated_and_the_reallocation_is_itself_recorded():
    sample = sample_gold_cases(
        population(),
        GoldSamplingParams(
            n_cases=350, substrate_shares={"amlworld_hi_small": 0.7, "elliptic2": 0.3}
        ),
    )
    assert len(sample) == 350
    assert "reallocated" in sample.deficits
    assert "dataset:elliptic2" in sample.deficits


def test_reallocation_can_be_refused_and_then_the_sample_is_simply_smaller():
    sample = sample_gold_cases(
        population(),
        GoldSamplingParams(
            n_cases=350,
            substrate_shares={"amlworld_hi_small": 0.7, "elliptic2": 0.3},
            reallocate_deficit=False,
        ),
    )
    assert len(sample) < 350
    assert "reallocated" not in sample.deficits


# ------------------------------------------------------------------ refusal ---


def test_a_population_without_hard_negatives_is_refused():
    """The floor is a refusal threshold, not a target to approximate."""
    pool = [c for c in population() if c.case_class != "hard_negative"]
    with pytest.raises(GoldSamplingError, match="hard negatives"):
        sample_gold_cases(
            pool, GoldSamplingParams(n_cases=350, substrate_shares={"amlworld_hi_small": 1.0})
        )


def test_a_population_too_small_for_the_reservation_is_refused():
    pool = population(per_typology=2, n_hard=20, n_licit=20)
    with pytest.raises(GoldSamplingError, match="fewer than"):
        sample_gold_cases(
            pool,
            GoldSamplingParams(
                n_cases=350, min_reserved=200, substrate_shares={"amlworld_hi_small": 1.0}
            ),
        )


def test_a_hard_negative_share_below_the_floor_is_refused_before_any_case_is_looked_at():
    with pytest.raises(GoldSamplingError, match="below the"):
        GoldSamplingParams(hard_negative_share=0.05, typed_share=0.5, unclassified_share=0.45)


def test_shares_are_normalised():
    params = GoldSamplingParams(hard_negative_share=2.8, typed_share=4.4, unclassified_share=2.8)
    total = params.hard_negative_share + params.typed_share + params.unclassified_share
    assert total == pytest.approx(1.0)


def test_report_round_trips_to_json_serialisable_content():
    import json

    sample = sample_gold_cases(
        population(),
        GoldSamplingParams(
            n_cases=100, min_reserved=100, substrate_shares={"amlworld_hi_small": 1.0}
        ),
    )
    json.dumps(sample.to_dict())
    assert "hard negatives" in sample.summary()
