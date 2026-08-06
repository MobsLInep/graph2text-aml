"""Reporting, metric validation, and the input types that join narratives to records.

The reporting tests are mostly about ordering and separation — Layer 2 before Layer 1,
the two test streams never pooled — because those are reporting decisions with reasons
behind them, and a decision nothing enforces is a decision that erodes.
"""

from __future__ import annotations

import json

import pytest
from tests.factories import as_laundering_stream, fan_out_case

from g2t_aml.eval.metric_validation import correlate, fisher_interval, validate_metrics
from g2t_aml.eval.report import HEADLINE_METRIC, PRIMARY_METRICS, evaluate, worst_cases
from g2t_aml.eval.types import (
    EvaluationInputError,
    SystemOutput,
    load_system_outputs,
    pair_outputs_with_facts,
)
from g2t_aml.facts.extractor import extract_facts

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


@pytest.fixture(scope="module")
def facts():
    return extract_facts(as_laundering_stream(fan_out_case(width=9), "fan_out"))


@pytest.fixture(scope="module")
def facts_index(facts):
    # One record, several case ids: the report needs more than one case before an
    # interval or a comparison means anything.
    import dataclasses

    return {f"case-{i}": dataclasses.replace(facts, case_id=f"case-{i}") for i in range(12)}


def outputs_for(facts_index, system: str, narrative, stream: str = "balanced", seed=None):
    return [
        SystemOutput(
            system=system,
            case_id=case_id,
            narrative=narrative(case_id) if callable(narrative) else narrative,
            stream=stream,
            seed=seed,
        )
        for case_id in facts_index
    ]


GOOD = "The subject account sent funds to 9 distinct counterparties."
BAD = "The subject account sent funds to 40 distinct counterparties."


# ------------------------------------------------------------ input types ---


def test_load_reads_a_corpus_row_a_generation_row_and_a_plain_row(tmp_path):
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"case_id": "a", "target_narrative": "corpus shape", "tier": "bronze"},
                {"case_id": "b", "texts": ["generation shape", "second candidate"]},
                {"case_id": "c", "narrative": "plain shape"},
            )
        ),
        encoding="utf-8",
    )

    loaded = load_system_outputs(path, system="x")

    assert [o.narrative for o in loaded] == ["corpus shape", "generation shape", "plain shape"]


def test_an_empty_narrative_is_refused_rather_than_scored_as_perfect(tmp_path):
    # An empty narrative trivially contains no contradicted claim, so a silent fallback
    # to "" would flatter a broken system with a perfect headline score.
    path = tmp_path / "empty.jsonl"
    path.write_text(json.dumps({"case_id": "a", "texts": []}), encoding="utf-8")
    with pytest.raises(EvaluationInputError, match="no narrative"):
        load_system_outputs(path, system="x")


def test_an_unnamed_system_is_refused(tmp_path):
    path = tmp_path / "unnamed.jsonl"
    path.write_text(json.dumps({"case_id": "a", "narrative": "x"}), encoding="utf-8")
    with pytest.raises(EvaluationInputError, match="names no system"):
        load_system_outputs(path)


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(EvaluationInputError, match="no system outputs"):
        load_system_outputs(tmp_path / "nothing.jsonl", system="x")


def test_pairing_refuses_a_narrative_with_no_fact_record(facts):
    outputs = [SystemOutput(system="x", case_id="missing", narrative="x")]
    with pytest.raises(EvaluationInputError, match="no fact record"):
        list(pair_outputs_with_facts(outputs, {facts.case_id: facts}))


def test_pairing_can_skip_instead_when_asked(facts):
    outputs = [SystemOutput(system="x", case_id="missing", narrative="x")]
    assert list(pair_outputs_with_facts(outputs, {}, on_missing="skip")) == []


def test_pairing_attaches_a_gold_reference_when_one_exists(facts):
    outputs = [SystemOutput(system="x", case_id=facts.case_id, narrative="x")]
    (paired,) = pair_outputs_with_facts(
        outputs, {facts.case_id: facts}, references={facts.case_id: "the gold text"}
    )
    assert paired.reference == "the gold text"


def test_pairing_rejects_an_unknown_on_missing_mode(facts):
    with pytest.raises(EvaluationInputError, match="on_missing"):
        list(pair_outputs_with_facts([], {}, on_missing="whatever"))


# ---------------------------------------------------------- the report ---


def test_a_two_system_report_ranks_by_the_headline_metric(facts_index):
    report = evaluate(
        [*outputs_for(facts_index, "good", GOOD), *outputs_for(facts_index, "bad", BAD)],
        {},
        facts_index,
        bertscore_model=None,
        n_resamples=100,
    )

    ranked = [r.system for r in report.systems_in("balanced")]
    assert ranked == ["good", "bad"]
    assert report.systems[("good", "balanced")].faithfulness.zero_hallucination_rate == 1.0
    assert report.systems[("bad", "balanced")].faithfulness.zero_hallucination_rate == 0.0


def test_the_headline_metric_leads_the_markdown_and_the_latex(facts_index):
    report = evaluate(
        outputs_for(facts_index, "good", GOOD),
        {},
        facts_index,
        bertscore_model=None,
        n_resamples=50,
    )
    markdown = report.to_markdown()
    latex = report.to_latex("balanced")

    # Layer 2 before Layer 1, in both. The order a results section is read in is the
    # order its numbers are believed in.
    assert markdown.index("Layer 2") < markdown.index("Layer 1")
    assert "Zero-Hallucination Rate is the headline" in markdown
    assert latex.index("faithfulness-balanced") < latex.index("taxonomy-balanced")


def test_the_two_streams_are_reported_separately_and_never_pooled(facts_index):
    report = evaluate(
        [
            *outputs_for(facts_index, "s1", GOOD, stream="balanced"),
            *outputs_for(facts_index, "s1", BAD, stream="realistic"),
        ],
        {},
        facts_index,
        bertscore_model=None,
        n_resamples=50,
    )

    assert set(report.streams) == {"balanced", "realistic"}
    assert report.streams[0] == "balanced"
    assert report.systems[("s1", "balanced")].faithfulness.zero_hallucination_rate == 1.0
    assert report.systems[("s1", "realistic")].faithfulness.zero_hallucination_rate == 0.0
    # No key anywhere combines them.
    assert not any(stream == "pooled" for _, stream in report.systems)


def test_comparisons_are_keyed_by_stream_so_the_family_is_visible(facts_index):
    report = evaluate(
        [*outputs_for(facts_index, "good", GOOD), *outputs_for(facts_index, "bad", BAD)],
        {},
        facts_index,
        bertscore_model=None,
        n_resamples=100,
    )

    key = f"balanced/{HEADLINE_METRIC}"
    assert key in report.comparisons
    (comparison,) = report.comparisons[key]
    assert comparison.p_adjusted is not None
    # Pairs are emitted in sorted system order, so "bad" is a and "good" is b, and the
    # sign of delta follows that ordering rather than the ranking.
    assert (comparison.system_a, comparison.system_b) == ("bad", "good")
    assert comparison.cliffs_delta == -1.0
    assert "1 comparisons" in comparison.family


def test_only_layer_2_metrics_get_intervals_and_tests(facts_index):
    # A corrected significance test on BLEU would dress an overlap metric in the same
    # statistics as the faithfulness metrics and invite the reader to weigh them equally.
    report = evaluate(
        outputs_for(facts_index, "s1", GOOD), {}, facts_index, bertscore_model=None, n_resamples=50
    )
    intervals = report.systems[("s1", "balanced")].intervals

    assert set(intervals) <= set(PRIMARY_METRICS)
    assert not any("bleu" in key or "rouge" in key for key in intervals)


def test_breakdowns_by_typology_and_substrate_are_produced(facts_index):
    report = evaluate(
        outputs_for(facts_index, "s1", GOOD), {}, facts_index, bertscore_model=None, n_resamples=50
    )
    entry = report.systems[("s1", "balanced")]

    assert set(entry.by_typology) == {"fan_out"}
    assert set(entry.by_dataset) == {"amlworld_hi_small"}


def test_seed_variance_is_reported_when_several_seeds_are_present(facts_index):
    report = evaluate(
        [
            *outputs_for(facts_index, "s1", GOOD, seed=1),
            *outputs_for(facts_index, "s1", BAD, seed=2),
        ],
        {},
        facts_index,
        bertscore_model=None,
        n_resamples=50,
    )
    summary = report.systems[("s1", "balanced")].seeds[HEADLINE_METRIC]

    assert summary.n_seeds == 2
    assert summary.std is not None
    assert not summary.single_seed


def test_worst_cases_are_selected_mechanically_with_their_violations(facts_index):
    report = evaluate(
        outputs_for(facts_index, "bad", BAD), {}, facts_index, bertscore_model=None, n_resamples=50
    )
    worst = report.systems[("bad", "balanced")].worst

    assert worst
    assert all(entry["violations"] for entry in worst)
    assert all(v["verdict"] != "supported" for entry in worst for v in entry["violations"])


def test_worst_cases_rank_by_contradictions_first():
    from g2t_aml.eval.layer2_faithfulness import CaseFaithfulness

    def case(case_id, contradicted, f1_driver):
        return CaseFaithfulness(
            case_id=case_id,
            system="s",
            typology="fan_out",
            dataset="d",
            seed=None,
            stream="balanced",
            n_claims=10,
            n_supported=10 - contradicted,
            n_contradicted=contradicted,
            n_unverifiable=0,
            n_salient_required=10,
            n_salient_covered=f1_driver,
            n_numeric=0,
            n_numeric_correct=0,
            typology_correct=None,
            ordering_correct=None,
        )

    ranked = worst_cases([case("a", 1, 10), case("b", 5, 10), case("c", 0, 1)])
    assert [entry["case_id"] for entry in ranked] == ["b", "a", "c"]


def test_the_report_writes_json_markdown_latex_and_errors(tmp_path, facts_index):
    report = evaluate(
        outputs_for(facts_index, "s1", BAD), {}, facts_index, bertscore_model=None, n_resamples=50
    )

    written = report.write_all(tmp_path / "out")

    assert set(written) >= {"json", "markdown", "latex_balanced", "errors"}
    payload = json.loads(written["json"].read_text(encoding="utf-8"))
    assert payload["headline_metric"] == HEADLINE_METRIC
    assert payload["streams"] == ["balanced"]
    errors = [json.loads(line) for line in written["errors"].read_text().splitlines()]
    assert errors and all("hallucination_class" in e for e in errors)


def test_the_json_is_valid_json_with_no_nan(tmp_path, facts_index):
    # A metric library can return NaN on a degenerate input, and NaN is not valid JSON:
    # a report carrying one is a report no downstream consumer can read back.
    report = evaluate(
        outputs_for(facts_index, "s1", GOOD), {}, facts_index, bertscore_model=None, n_resamples=50
    )
    text = report.to_json(tmp_path / "r.json").read_text(encoding="utf-8")
    assert "NaN" not in text
    json.loads(text)


def test_latex_escapes_a_system_name_that_would_break_the_table(facts_index):
    report = evaluate(
        outputs_for(facts_index, "s1_with_underscores", GOOD),
        {},
        facts_index,
        bertscore_model=None,
        n_resamples=50,
    )
    assert "s1\\_with\\_underscores" in report.to_latex("balanced")


def test_latex_for_an_absent_stream_says_so(facts_index):
    report = evaluate(
        outputs_for(facts_index, "s1", GOOD), {}, facts_index, bertscore_model=None, n_resamples=50
    )
    assert "no systems scored" in report.to_latex("realistic")


# ------------------------------------------------------ metric validation ---


def test_correlation_reports_both_coefficients_with_intervals():
    automatic = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    human = [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0]

    result = correlate("zero_hallucination", "factual_correctness", automatic, human)

    assert result.spearman > 0.9
    assert result.pearson > 0.9
    assert result.spearman_ci[0] < result.spearman < result.spearman_ci[1]
    assert not result.reliable  # eight pairs is below the flagging threshold


def test_a_constant_metric_column_is_zero_correlation_not_nan():
    # A real and likely outcome: Bronze is 100% supported by construction, so its
    # per-case faithfulness column is constant and both coefficients come back NaN.
    result = correlate(
        "fact_precision", "factual_correctness", [1.0] * 10, [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    )
    assert result.spearman == 0.0
    assert result.pearson == 0.0
    assert result.spearman_p == 1.0


def test_a_correlation_below_three_pairs_is_refused():
    with pytest.raises(ValueError, match="at least three"):
        correlate("m", "d", [0.1, 0.2], [1.0, 2.0])


def test_a_mismatched_correlation_is_refused():
    with pytest.raises(ValueError, match="paired values"):
        correlate("m", "d", [0.1, 0.2, 0.3], [1.0])


def test_fisher_interval_is_degenerate_where_the_transform_is():
    assert fisher_interval(0.5, 3) == (0.5, 0.5)
    assert fisher_interval(1.0, 100) == (1.0, 1.0)


def test_fisher_interval_narrows_as_n_grows():
    small = fisher_interval(0.6, 20)
    large = fisher_interval(0.6, 500)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_validate_metrics_finds_the_headline_and_records_what_it_skipped():
    cases = [f"c{i}" for i in range(10)]
    automatic = {
        "zero_hallucination": {c: float(i % 2) for i, c in enumerate(cases)},
        "bleu": {c: 0.3 for c in cases},
        "unshared": {"nobody": 1.0},
    }
    human = {"factual_correctness": {c: float(i % 2) * 4 + 1 for i, c in enumerate(cases)}}

    report = validate_metrics(automatic, human)

    assert report.headline is not None
    assert report.headline.metric == "zero_hallucination"
    assert report.headline.spearman == pytest.approx(1.0)
    assert "unshared/factual_correctness" in report.missing
    assert "| `zero_hallucination` |" in report.markdown()


def test_validate_metrics_over_no_human_data_says_it_has_not_run():
    report = validate_metrics({"zero_hallucination": {"c1": 1.0}}, {})
    assert report.results == ()
    assert "has not run" in report.markdown()
