"""The statistics, each checked against a published value or an independent implementation.

The rule this file follows is invariant 1's rule, applied to Phase 12: a statistic is
validated against a source outside this repository, never against a second implementation
written by the same author on the same afternoon. Two such implementations agree on their
shared misreading, and the agreement is then mistaken for evidence.

Where scipy or the ``krippendorff`` package is available it is used as the independent
check, and where a canonical worked example exists — Krippendorff's own four-observer
matrix — the published number is asserted directly, because a package can be wrong too.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from g2t_aml.human.study_analysis import (
    NEMENYI_Q,
    _alpha_ordinal,
    analyse_study,
    critical_difference_diagram,
    durbin_test,
    friedman_test,
    intra_rater_reliability,
    krippendorff_alpha_ordinal,
    load_blind_key,
    nemenyi_posthoc,
    normalised_levenshtein,
    pearson_with_ci,
    spearman_with_ci,
)
from g2t_aml.human.study_design import DesignError, build_design
from g2t_aml.human.study_ui import RatingResponse

scipy_stats = pytest.importorskip("scipy.stats", reason="scipy is in the `eval` extra")

#: Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability", the four-observer
#: worked example. Published: nominal 0.743, **ordinal 0.815**, interval 0.849.
KRIPPENDORFF_2011 = [
    [1, 1, None, 1],
    [2, 2, 3, 2],
    [3, 3, 3, 3],
    [3, 3, 3, 3],
    [2, 2, 2, 2],
    [1, 2, 3, 4],
    [4, 4, 4, 4],
    [1, 1, 2, 1],
    [2, 2, 2, 2],
    [None, 5, 5, 5],
    [None, None, 1, 1],
    [None, 3, None, None],
]


# ------------------------------------------------------------ edit distance ---


def test_levenshtein_matches_the_textbook_example():
    """kitten -> sitting is 3 edits over a length of 7."""
    assert normalised_levenshtein("kitten", "sitting") == pytest.approx(3 / 7)


def test_identical_text_is_zero_distance():
    assert normalised_levenshtein("abc", "abc") == 0.0
    assert normalised_levenshtein("", "") == 0.0


def test_wholly_replaced_text_is_distance_one():
    assert normalised_levenshtein("abc", "") == 1.0
    assert normalised_levenshtein("", "abc") == 1.0


def test_distance_is_symmetric_and_bounded():
    rng = random.Random(3)
    alphabet = "abcdefg "
    for _ in range(50):
        a = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        b = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        d = normalised_levenshtein(a, b)
        assert 0.0 <= d <= 1.0
        assert d == pytest.approx(normalised_levenshtein(b, a))


def test_a_one_character_correction_in_a_long_draft_is_small():
    """The reason the metric is character-level: numeric fixes are the ones that matter."""
    draft = "The account received 9,435 Canadian Dollar from six counterparties." * 3
    fixed = draft.replace("9,435", "9,434", 1)
    assert 0 < normalised_levenshtein(draft, fixed) < 0.02


# ------------------------------------------------------- Krippendorff alpha ---


def test_ordinal_alpha_reproduces_the_published_worked_example():
    assert _alpha_ordinal(KRIPPENDORFF_2011) == pytest.approx(0.815, abs=5e-4)


def test_ordinal_alpha_matches_the_independent_package():
    krippendorff = pytest.importorskip("krippendorff", reason="in the `eval` extra")
    numpy = pytest.importorskip("numpy")

    matrix = numpy.array(
        [
            [row[c] if row[c] is not None else numpy.nan for row in KRIPPENDORFF_2011]
            for c in range(4)
        ],
        dtype=float,
    )
    expected = krippendorff.alpha(reliability_data=matrix, level_of_measurement="ordinal")
    assert _alpha_ordinal(KRIPPENDORFF_2011) == pytest.approx(expected, abs=1e-9)


def test_perfect_agreement_is_alpha_one():
    assert _alpha_ordinal([[3, 3, 3], [5, 5, 5], [1, 1, 1]]) == pytest.approx(1.0)


def test_ordinal_alpha_is_not_the_nominal_one():
    """A 6-vs-7 disagreement is not the same event as a 1-vs-7."""
    near = _alpha_ordinal([[6, 7], [6, 7], [1, 1], [2, 2], [7, 7]])
    far = _alpha_ordinal([[1, 7], [1, 7], [1, 1], [2, 2], [7, 7]])
    assert near > far


def test_alpha_refuses_a_matrix_with_nothing_pairable():
    with pytest.raises(ValueError, match="no agreement to measure"):
        _alpha_ordinal([[1, None], [2, None]])


def test_the_bootstrap_interval_brackets_the_estimate():
    alpha, lo, hi = krippendorff_alpha_ordinal(KRIPPENDORFF_2011, n_bootstrap=500, seed=1)
    assert lo <= alpha <= hi
    assert lo >= -1.0 and hi <= 1.0


def test_the_bootstrap_is_reproducible():
    a = krippendorff_alpha_ordinal(KRIPPENDORFF_2011, n_bootstrap=300, seed=7)
    b = krippendorff_alpha_ordinal(KRIPPENDORFF_2011, n_bootstrap=300, seed=7)
    assert a == b


# ------------------------------------------------------------- Friedman ---


def _random_blocks(n: int, k: int, seed: int, shift: float = 0.5):
    rng = random.Random(seed)
    systems = [f"S{j}" for j in range(k)]
    rows = {
        f"b{i}": {s: rng.gauss(shift * j, 1.0) for j, s in enumerate(systems)} for i in range(n)
    }
    return rows, systems


def test_friedman_matches_scipy():
    blocks, systems = _random_blocks(12, 4, seed=0)
    mine = friedman_test(blocks, systems)
    theirs = scipy_stats.friedmanchisquare(
        *[[blocks[b][s] for b in sorted(blocks)] for s in systems]
    )
    assert mine.statistic == pytest.approx(theirs.statistic, rel=1e-10)
    assert mine.p_value == pytest.approx(theirs.pvalue, rel=1e-8)


def test_friedman_tie_correction_matches_scipy():
    """Without the correction a tied matrix reports an inflated statistic."""
    systems = ["A", "B", "C", "D"]
    blocks = {f"b{i}": {"A": 1.0, "B": 1.0, "C": 2.0, "D": 3.0} for i in range(8)}
    mine = friedman_test(blocks, systems)
    theirs = scipy_stats.friedmanchisquare(
        *[[blocks[b][s] for b in sorted(blocks)] for s in systems]
    )
    assert mine.statistic == pytest.approx(theirs.statistic, rel=1e-10)


@pytest.mark.parametrize("k", [3, 5, 6])
def test_friedman_matches_scipy_across_widths(k):
    blocks, systems = _random_blocks(10, k, seed=k)
    mine = friedman_test(blocks, systems)
    theirs = scipy_stats.friedmanchisquare(
        *[[blocks[b][s] for b in sorted(blocks)] for s in systems]
    )
    assert mine.statistic == pytest.approx(theirs.statistic, rel=1e-10)


def test_friedman_refuses_an_incomplete_block():
    blocks, systems = _random_blocks(5, 3, seed=1)
    del blocks["b0"]["S1"]
    with pytest.raises(ValueError, match="complete blocks"):
        friedman_test(blocks, systems)


# --------------------------------------------------------------- Durbin ---


def test_durbin_reduces_to_friedman_on_a_complete_design():
    """The algebra says so; this asserts it rather than trusting it."""
    blocks, systems = _random_blocks(12, 4, seed=5)
    assert durbin_test(blocks, systems).statistic == pytest.approx(
        friedman_test(blocks, systems).statistic, rel=1e-10
    )


def test_durbin_handles_a_genuinely_incomplete_balanced_design():
    """A classic BIBD: 7 blocks, 7 treatments, 3 per block, each treatment in 3 blocks."""
    bibd = [
        (0, 1, 3),
        (1, 2, 4),
        (2, 3, 5),
        (3, 4, 6),
        (4, 5, 0),
        (5, 6, 1),
        (6, 0, 2),
    ]
    values = {0: 5.0, 1: 4.0, 2: 3.0, 3: 2.0, 4: 1.0, 5: 6.0, 6: 7.0}
    blocks = {
        f"b{i}": {f"S{t}": values[t] + 0.01 * i for t in triple} for i, triple in enumerate(bibd)
    }
    result = durbin_test(blocks, [f"S{t}" for t in range(7)])
    assert result.n_blocks == 7
    assert result.k_per_block == 3
    assert result.r_per_treatment == 3
    assert 0.0 <= result.p_value <= 1.0


def test_durbin_refuses_an_unbalanced_design():
    blocks = {"b0": {"A": 1.0, "B": 2.0}, "b1": {"A": 1.0, "B": 2.0, "C": 3.0}}
    with pytest.raises(ValueError, match="differing sizes"):
        durbin_test(blocks, ["A", "B", "C"])


# --------------------------------------------------------------- Nemenyi ---


def test_nemenyi_critical_difference_matches_the_published_formula():
    """Demsar (2006): CD = q_alpha * sqrt(k(k+1) / 6N)."""
    ranks = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
    result = nemenyi_posthoc(ranks, n_blocks=10)
    expected = NEMENYI_Q["0.05"][4] * math.sqrt(4 * 5 / (6 * 10))
    assert result.critical_difference == pytest.approx(expected)


def test_nemenyi_flags_exactly_the_pairs_beyond_the_critical_difference():
    ranks = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
    result = nemenyi_posthoc(ranks, n_blocks=10)
    cd = result.critical_difference
    expected = {(a, b) for a in ranks for b in ranks if a < b and abs(ranks[a] - ranks[b]) >= cd}
    assert set(result.significant_pairs) == expected


def test_more_blocks_shrink_the_critical_difference():
    ranks = {"A": 1.0, "B": 2.0, "C": 3.0}
    assert nemenyi_posthoc(ranks, 100).critical_difference < (
        nemenyi_posthoc(ranks, 10).critical_difference
    )


def test_nemenyi_refuses_a_width_outside_the_published_table():
    ranks = {f"S{i}": float(i) for i in range(12)}
    with pytest.raises(ValueError, match="no tabulated"):
        nemenyi_posthoc(ranks, 10)


def test_the_critical_difference_diagram_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    result = nemenyi_posthoc({"S1": 1.1, "S2": 1.9, "B7": 3.2, "B3": 3.8, "Bronze": 5.0}, 10)
    path = critical_difference_diagram(result, tmp_path / "cd.png", title="Time")
    assert path.is_file()
    assert path.stat().st_size > 5_000


# ----------------------------------------------------------- correlations ---


def test_spearman_matches_scipy():
    rng = random.Random(11)
    x = [rng.gauss(0, 1) for _ in range(40)]
    y = [0.6 * v + rng.gauss(0, 1) for v in x]
    mine, theirs = spearman_with_ci(x, y), scipy_stats.spearmanr(x, y)
    assert mine.statistic == pytest.approx(theirs.statistic, rel=1e-10)
    assert mine.p_value == pytest.approx(theirs.pvalue, rel=1e-6)


def test_pearson_matches_scipy_including_the_interval():
    rng = random.Random(12)
    x = [rng.gauss(0, 1) for _ in range(40)]
    y = [0.6 * v + rng.gauss(0, 1) for v in x]
    mine, theirs = pearson_with_ci(x, y), scipy_stats.pearsonr(x, y)
    assert mine.statistic == pytest.approx(theirs.statistic, rel=1e-10)
    assert mine.p_value == pytest.approx(theirs.pvalue, rel=1e-6)
    lo, hi = theirs.confidence_interval()
    assert mine.ci_low == pytest.approx(lo, abs=1e-6)
    assert mine.ci_high == pytest.approx(hi, abs=1e-6)


def test_spearman_is_unmoved_by_a_monotone_transform():
    rng = random.Random(13)
    x = [abs(rng.gauss(0, 1)) + 0.1 for _ in range(30)]
    y = [rng.gauss(0, 1) for _ in range(30)]
    a = spearman_with_ci(x, y).statistic
    b = spearman_with_ci([math.log(v) for v in x], y).statistic
    assert a == pytest.approx(b, abs=1e-12)


def test_correlation_refuses_mismatched_inputs():
    with pytest.raises(ValueError, match="equal-length"):
        pearson_with_ci([1.0, 2.0], [1.0])


# ------------------------------------------------------- intra-rater ---


def test_intra_rater_reliability_pairs_repeats_with_their_originals():
    def response(position, is_repeat, score):
        return RatingResponse(
            item_id=f"i{position}",
            rater_id="rater-01",
            case_id="c1",
            position=position,
            is_repeat=is_repeat,
            factual_correctness=score,
            completeness=score,
            actionability=score,
            readability=score,
            regulatory_tone=score,
            would_file=score >= 5,
            seconds_to_usable_draft=10.0,
            presented_narrative="n",
            corrected_narrative="n",
        )

    report = intra_rater_reliability([response(0, False, 6), response(20, True, 6)])
    assert report["n_pairs"] == 1
    assert report["mean_absolute_difference"]["factual_correctness"] == 0.0
    assert report["would_file_agreement"] == 1.0


def test_no_repeats_reports_no_pairs():
    assert intra_rater_reliability([])["n_pairs"] == 0


# ------------------------------------------------------ the whole analysis ---


@pytest.fixture(scope="module")
def simulated():
    """A study whose truth is known, so the pipeline can be checked for recovering it."""
    systems = ["S1", "S2", "B7", "B3", "Bronze"]
    design, key = build_design(
        [f"c{i:03d}" for i in range(100)],
        systems,
        [f"rater-{i:02d}" for i in range(10)],
        dataset="amlworld_hi_small",
        items_per_rater=60,
    )
    rng = random.Random(7)
    quality = {"S1": 6.0, "S2": 5.5, "B7": 5.0, "B3": 4.5, "Bronze": 3.5}
    speed = {"S1": 180, "S2": 200, "B7": 240, "B3": 260, "Bronze": 400}
    base = "The subject received funds from three counterparties within 22 hours."
    responses, automatic = [], {}
    for item in design.items:
        system = key.system_for(item.item_id)
        score = max(1, min(7, round(rng.gauss(quality[system], 1.0))))
        responses.append(
            RatingResponse(
                item_id=item.item_id,
                rater_id=item.rater_id,
                case_id=item.case_id,
                position=item.position,
                is_repeat=item.is_repeat,
                factual_correctness=score,
                completeness=score,
                actionability=score,
                readability=score,
                regulatory_tone=score,
                would_file=score >= 5,
                seconds_to_usable_draft=max(10.0, rng.gauss(speed[system], 40)),
                presented_narrative=base,
                corrected_narrative=(
                    base if system == "S1" else base.replace("three", "several")[: 40 + 4 * score]
                ),
                timing_source="browser",
            )
        )
        automatic[item.item_id] = min(1.0, max(0.0, rng.gauss(quality[system] / 7.0, 0.12)))
    return analyse_study(responses, key, systems=systems, automatic_scores=automatic)


def test_the_analysis_recovers_the_simulated_time_ordering(simulated):
    means = {s: v["mean_seconds"] for s, v in simulated.timing["per_system"].items()}
    assert means["S1"] < means["S2"] < means["B7"] < means["B3"] < means["Bronze"]


def test_every_arm_beats_the_baseline_on_time(simulated):
    for system, entry in simulated.timing["paired_vs_baseline"].items():
        assert entry["mean_difference"] < 0, system
        assert entry["p_value"] is not None and entry["p_value"] < 0.05


def test_the_anchor_block_makes_agreement_computable(simulated):
    """The regression this design change exists to prevent."""
    entry = simulated.agreement["factual_correctness"]
    assert entry["n_units"] > 0
    assert entry["alpha_ordinal"] is not None


def test_repeats_yield_an_intra_rater_estimate(simulated):
    assert simulated.intra_rater["n_pairs"] > 0


def test_friedman_and_durbin_are_both_reported(simulated):
    entry = simulated.omnibus["time_to_usable_draft"]
    assert entry["friedman"]["p_value"] < 0.05
    assert entry["durbin"]["p_value"] < 0.05
    assert entry["durbin"]["n_blocks"] > entry["friedman"]["n_blocks"]


def test_durbin_records_the_subset_it_ran_on(simulated):
    """A subset analysis reported as though it were the whole is the failure mode."""
    assert "subset" in simulated.omnibus["time_to_usable_draft"]["durbin"]


def test_the_automatic_metric_correlation_is_computed(simulated):
    entry = simulated.metric_validation
    assert entry["n"] > 100
    assert entry["spearman"]["statistic"] > 0
    assert entry["spearman"]["ci_low"] < entry["spearman"]["statistic"]


def test_likert_means_always_carry_an_agreement_statistic(simulated):
    for name, dimension in simulated.likert.items():
        if name == "aggregate":
            continue
        assert "agreement" in dimension


def test_the_aggregate_is_reported_and_carries_its_caveat(simulated):
    """The brief asks for aggregate reporting; the caveat is what keeps it honest."""
    aggregate = simulated.likert["aggregate"]
    assert set(aggregate["per_system"]) == set(simulated.systems)
    assert "not commensurable" in aggregate["note"] or "not\n" in aggregate["note"]
    assert aggregate["per_system"]["S1"]["mean"] > aggregate["per_system"]["Bronze"]["mean"]


def test_a_missing_automatic_score_becomes_a_warning_not_a_silence():
    systems = ["S1", "Bronze"]
    design, key = build_design(
        [f"c{i:03d}" for i in range(40)],
        systems,
        ["rater-01", "rater-02"],
        dataset="d",
        items_per_rater=20,
        n_anchor_cases=4,
    )
    responses = [
        RatingResponse(
            item_id=i.item_id,
            rater_id=i.rater_id,
            case_id=i.case_id,
            position=i.position,
            is_repeat=i.is_repeat,
            factual_correctness=5,
            completeness=5,
            actionability=5,
            readability=5,
            regulatory_tone=5,
            would_file=True,
            seconds_to_usable_draft=100.0,
            presented_narrative="n",
            corrected_narrative="n",
        )
        for i in design.items
    ]
    result = analyse_study(responses, key, systems=systems)
    assert any("Phase 12 gate requires" in w for w in result.warnings)


def test_server_timed_responses_are_flagged_in_the_warnings():
    systems = ["S1", "Bronze"]
    design, key = build_design(
        [f"c{i:03d}" for i in range(40)],
        systems,
        ["rater-01", "rater-02"],
        dataset="d",
        items_per_rater=20,
        n_anchor_cases=4,
    )
    responses = [
        RatingResponse(
            item_id=i.item_id,
            rater_id=i.rater_id,
            case_id=i.case_id,
            position=i.position,
            is_repeat=i.is_repeat,
            factual_correctness=5,
            completeness=5,
            actionability=5,
            readability=5,
            regulatory_tone=5,
            would_file=True,
            seconds_to_usable_draft=100.0,
            presented_narrative="n",
            corrected_narrative="n",
            timing_source="server",
        )
        for i in design.items
    ]
    result = analyse_study(responses, key, systems=systems)
    assert any("server clock" in w for w in result.warnings)


def test_responses_from_a_different_build_are_refused(simulated):
    design, key = build_design(
        [f"c{i:03d}" for i in range(40)],
        ["S1", "Bronze"],
        ["rater-01", "rater-02"],
        dataset="d",
        items_per_rater=20,
        n_anchor_cases=4,
    )
    stray = RatingResponse(
        item_id="not-in-the-key",
        rater_id="rater-01",
        case_id="c000",
        position=0,
        is_repeat=False,
        factual_correctness=5,
        completeness=5,
        actionability=5,
        readability=5,
        regulatory_tone=5,
        would_file=True,
        seconds_to_usable_draft=1.0,
        presented_narrative="n",
        corrected_narrative="n",
    )
    with pytest.raises(DesignError, match="different builds"):
        analyse_study([stray], key, systems=["S1", "Bronze"])


def test_the_blind_key_loader_refuses_a_design_file(tmp_path):
    design, key = build_design(
        [f"c{i:03d}" for i in range(40)],
        ["S1", "Bronze"],
        ["rater-01", "rater-02"],
        dataset="d",
        items_per_rater=20,
        n_anchor_cases=4,
    )
    design.write(tmp_path / "d.json", key, tmp_path / "k.json")
    with pytest.raises(DesignError, match="not a blind key"):
        load_blind_key(tmp_path / "d.json")
    assert load_blind_key(tmp_path / "k.json").assignments == key.assignments
