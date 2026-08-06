"""Phase 11: aggregation is correct on synthetic data with known effects, and honest about gaps.

The statistical pipeline is checked against effects planted by hand, because a battery that
reports significance on data with no effect, or misses an effect of a known size, is a
battery that will do the same to the paper's numbers and nobody will be able to tell.

The other half of this file is about absence. A missing run must be reported by name and
never imputed; a single-seed row must print an em dash for its standard deviation and never
a zero. Both are ways a results table can lie without containing a wrong number.
"""

from __future__ import annotations

import json

import pytest

from g2t_aml.experiments.aggregate import (
    HEADLINE_METRIC,
    MetricRow,
    ablation_table_latex,
    aggregate_matrix,
    collect_rows,
    long_table,
    main_table_latex,
    missing_report,
    taxonomy_table_latex,
    write_outputs,
)
from g2t_aml.experiments.registry import SEEDS_CENTRAL, all_systems, get_system
from g2t_aml.experiments.runner import write_completion_marker
from g2t_aml.utils.io import write_json, write_jsonl


def _write_run(
    root,
    system,
    seed,
    *,
    metrics,
    per_case=None,
    taxonomy=None,
    stream="balanced",
    by_typology=None,
):
    """Materialise one completed run on disk, the way an executor plus Phase 10 would."""
    directory = root / system / f"seed{seed}" / "hash0000"
    directory.mkdir(parents=True, exist_ok=True)
    block = {
        "system": system,
        "stream": stream,
        "faithfulness": {"system": system, "n_cases": 100, **metrics},
        "taxonomy": {"rate_by_class": taxonomy or {}},
        "by_typology": by_typology or {},
        "by_dataset": {},
        "layer1": None,
        "worst_cases": [],
    }
    write_json(directory / "metrics.json", {"systems": {f"{system}/{stream}": block}})
    if per_case:
        write_jsonl(
            directory / "per_case.jsonl",
            [{"case_id": c, "stream": stream, **v} for c, v in per_case.items()],
        )
    write_completion_marker(
        directory,
        run_id=f"{system}_seed{seed}",
        system_id=system,
        seed=seed,
        config_hash="hash0000",
    )
    return directory


def test_long_table_is_empty_but_well_shaped_when_nothing_ran():
    """A downstream groupby must not raise on an empty matrix."""
    frame = long_table([])
    assert list(frame.columns) == [
        "system",
        "seed",
        "metric",
        "substrate",
        "stream",
        "test_set",
        "typology",
        "value",
        "n_cases",
    ]
    assert frame.empty


def test_long_table_ordering_is_deterministic():
    rows = [
        MetricRow("S1", 2024, "m", "sub", "balanced", "test", "all", 0.5, 10),
        MetricRow("A1", 42, "m", "sub", "balanced", "test", "all", 0.3, 10),
        MetricRow("S1", 42, "m", "sub", "balanced", "test", "all", 0.4, 10),
    ]
    first = long_table(rows).to_csv(index=False)
    second = long_table(list(reversed(rows))).to_csv(index=False)
    assert first == second


def test_collect_reports_every_declared_run_that_did_not_produce_metrics(tmp_path):
    rows, missing, _ = collect_rows(tmp_path)
    assert rows == []
    declared = sum(len(s.seeds) for s in all_systems())
    assert len(missing) == declared
    assert {m.system for m in missing} == {s.system_id for s in all_systems()}
    assert all(m.reason for m in missing)


def test_a_completed_run_with_no_metrics_file_is_a_distinct_absence(tmp_path):
    directory = tmp_path / "S1" / "seed42" / "hash0000"
    directory.mkdir(parents=True)
    write_completion_marker(
        directory, run_id="S1_seed42", system_id="S1", seed=42, config_hash="hash0000"
    )
    _rows, missing, _ = collect_rows(tmp_path, specs=[get_system("S1")])
    seed42 = next(m for m in missing if m.seed == 42)
    assert "wrote no metrics.json" in seed42.reason
    assert seed42.status == "completed"


def test_a_run_without_a_completion_marker_does_not_contribute(tmp_path):
    """A directory with metrics but no marker is an interrupted job, not a result."""
    directory = tmp_path / "S1" / "seed42" / "hash0000"
    directory.mkdir(parents=True)
    write_json(directory / "metrics.json", {"systems": {}})
    rows, missing, _ = collect_rows(tmp_path, specs=[get_system("S1")])
    assert rows == []
    assert len(missing) == len(SEEDS_CENTRAL)


def test_across_seed_summary_reports_a_standard_deviation(tmp_path):
    for seed, value in zip(SEEDS_CENTRAL, (0.80, 0.84, 0.82), strict=True):
        _write_run(tmp_path, "S1", seed, metrics={HEADLINE_METRIC: value})
    result = aggregate_matrix(tmp_path, specs=[get_system("S1")], n_resamples=200)
    summary = result.summaries[("balanced", HEADLINE_METRIC, "S1")]
    assert summary.n_seeds == len(SEEDS_CENTRAL)
    assert summary.mean == pytest.approx(0.82, abs=1e-9)
    assert summary.std is not None
    assert summary.std > 0


def test_a_single_seed_summary_has_no_standard_deviation(tmp_path):
    """`std is None` at one seed is what stops a single-seed row reading as zero-variance."""
    _write_run(tmp_path, "A2", 42, metrics={HEADLINE_METRIC: 0.7})
    result = aggregate_matrix(tmp_path, specs=[get_system("A2")], n_resamples=200)
    summary = result.summaries[("balanced", HEADLINE_METRIC, "A2")]
    assert summary.n_seeds == 1
    assert summary.std is None


def test_single_seed_rows_print_a_dagger_not_a_zero(tmp_path):
    _write_run(tmp_path, "A2", 42, metrics={HEADLINE_METRIC: 0.7})
    result = aggregate_matrix(tmp_path, specs=[get_system("A2")], n_resamples=200)
    latex = main_table_latex(result, metrics=(HEADLINE_METRIC,))
    a2_row = next(line for line in latex.splitlines() if line.startswith("A2 &"))
    assert r"\dagger" in a2_row
    assert "$\\pm$ 0.0000" not in a2_row


def test_a_system_with_no_numbers_still_gets_a_row_and_is_named(tmp_path):
    """A table with fewer rows than the matrix declares is how a null result disappears."""
    _write_run(tmp_path, "S1", 42, metrics={HEADLINE_METRIC: 0.9})
    result = aggregate_matrix(tmp_path, n_resamples=200)
    latex = main_table_latex(result, metrics=(HEADLINE_METRIC,))
    for spec in all_systems():
        escaped = spec.system_id.replace("_", r"\_")
        assert any(line.startswith(f"{escaped} &") for line in latex.splitlines()), spec
    assert "Systems with no numbers in this table" in latex
    assert r"A3\_F3" in latex


def test_planted_effect_is_detected_with_the_right_direction(tmp_path):
    """A known, large effect must come out significant after Holm correction."""
    treatment = {f"c{i}": {HEADLINE_METRIC: 1.0} for i in range(60)}
    control = {f"c{i}": {HEADLINE_METRIC: 0.0} for i in range(60)}
    _write_run(tmp_path, "S1", 42, metrics={HEADLINE_METRIC: 1.0}, per_case=treatment)
    _write_run(tmp_path, "A1", 42, metrics={HEADLINE_METRIC: 0.0}, per_case=control)

    result = aggregate_matrix(
        tmp_path,
        specs=[get_system("S1"), get_system("A1")],
        metrics=(HEADLINE_METRIC,),
        n_resamples=500,
    )
    family = result.comparisons[f"balanced/{HEADLINE_METRIC}"]
    comparison = next(c for c in family if {c.system_a, c.system_b} == {"S1", "A1"})
    assert comparison.p_adjusted is not None
    assert comparison.p_adjusted < 0.05
    assert comparison.difference_ci.excludes_zero
    assert abs(comparison.cliffs_delta) == pytest.approx(1.0)


def test_no_effect_is_not_reported_as_significant(tmp_path):
    """The verdict nobody wants must be reachable: identical arms are indistinguishable."""
    identical = {f"c{i}": {HEADLINE_METRIC: float(i % 2)} for i in range(60)}
    _write_run(tmp_path, "S1", 42, metrics={HEADLINE_METRIC: 0.5}, per_case=identical)
    _write_run(tmp_path, "A1", 42, metrics={HEADLINE_METRIC: 0.5}, per_case=dict(identical))

    result = aggregate_matrix(
        tmp_path,
        specs=[get_system("S1"), get_system("A1")],
        metrics=(HEADLINE_METRIC,),
        n_resamples=500,
    )
    family = result.comparisons[f"balanced/{HEADLINE_METRIC}"]
    comparison = next(c for c in family if {c.system_a, c.system_b} == {"S1", "A1"})
    assert comparison.p_adjusted is not None
    assert comparison.p_adjusted > 0.05
    assert comparison.marker == "ns"
    assert comparison.cliffs_delta == pytest.approx(0.0)


def test_the_correction_family_is_this_metric_on_this_slice(tmp_path):
    """D-079: correcting over 15 comparisons and over 120 is a different claim."""
    for system in ("S1", "A1", "B7"):
        per_case = {f"c{i}": {HEADLINE_METRIC: float(i % 3 == 0)} for i in range(30)}
        _write_run(tmp_path, system, 42, metrics={HEADLINE_METRIC: 0.33}, per_case=per_case)
    result = aggregate_matrix(
        tmp_path,
        specs=[get_system(s) for s in ("S1", "A1", "B7")],
        metrics=(HEADLINE_METRIC,),
        n_resamples=200,
    )
    family = result.comparisons[f"balanced/{HEADLINE_METRIC}"]
    assert len(family) == 3
    assert all("3 comparisons" in c.family for c in family)


def test_comparisons_run_on_the_cases_two_systems_share(tmp_path):
    """A system evaluated on a subset is compared fairly, not dropped."""
    _write_run(
        tmp_path,
        "S1",
        42,
        metrics={HEADLINE_METRIC: 1.0},
        per_case={f"c{i}": {HEADLINE_METRIC: 1.0} for i in range(40)},
    )
    _write_run(
        tmp_path,
        "A1",
        42,
        metrics={HEADLINE_METRIC: 0.0},
        per_case={f"c{i}": {HEADLINE_METRIC: 0.0} for i in range(20)},
    )
    result = aggregate_matrix(
        tmp_path,
        specs=[get_system("S1"), get_system("A1")],
        metrics=(HEADLINE_METRIC,),
        n_resamples=200,
    )
    comparison = result.comparisons[f"balanced/{HEADLINE_METRIC}"][0]
    assert comparison.n_cases == 20


def test_per_typology_rows_are_collected(tmp_path):
    _write_run(
        tmp_path,
        "S1",
        42,
        metrics={HEADLINE_METRIC: 0.8},
        by_typology={
            "fan_out": {"system": "S1", "n_cases": 40, HEADLINE_METRIC: 0.9},
            "cycle": {"system": "S1", "n_cases": 20, HEADLINE_METRIC: 0.6},
        },
    )
    rows, _missing, _tax = collect_rows(tmp_path, specs=[get_system("S1")])
    typologies = {r.typology: r.value for r in rows if r.metric == HEADLINE_METRIC}
    assert typologies["fan_out"] == pytest.approx(0.9)
    assert typologies["cycle"] == pytest.approx(0.6)
    assert typologies["all"] == pytest.approx(0.8)


def test_taxonomy_rates_come_from_rate_by_class(tmp_path):
    _write_run(
        tmp_path,
        "B1",
        42,
        metrics={HEADLINE_METRIC: 1.0},
        taxonomy={"H1": 0.0, "H9": 0.9179},
    )
    result = aggregate_matrix(tmp_path, specs=[get_system("B1")], n_resamples=200)
    assert result.taxonomy[("balanced", "B1")]["H9"] == pytest.approx(0.9179)
    latex = taxonomy_table_latex(result)
    assert "0.9179" in latex
    assert r"\textbf{H4}" in latex  # the Critical classes are marked


def test_missing_report_lists_every_gap_with_a_reason(tmp_path):
    result = aggregate_matrix(tmp_path, n_resamples=100)
    report = missing_report(result)
    assert "| System | Seed | Status | Reason |" in report
    assert "`S1`" in report and "`B5`" in report
    assert report.count("\n") >= sum(len(s.seeds) for s in all_systems())


def test_missing_report_says_so_when_the_matrix_is_complete():
    from g2t_aml.experiments.aggregate import AggregateResult

    assert missing_report(AggregateResult(rows=())) == "Every declared run produced metrics."


def test_write_outputs_emits_the_three_tables_and_the_tidy_table(tmp_path):
    _write_run(tmp_path, "S1", 42, metrics={HEADLINE_METRIC: 0.9})
    result = aggregate_matrix(tmp_path, specs=[get_system("S1")], n_resamples=200)
    written = write_outputs(result, tmp_path / "out")
    for key in (
        "metrics_long_csv",
        "aggregate_json",
        "main_balanced_tex",
        "ablation_balanced_tex",
        "taxonomy_balanced_tex",
    ):
        assert written[key].is_file(), key
    payload = json.loads(written["aggregate_json"].read_text())
    assert payload["headline_metric"] == HEADLINE_METRIC
    assert payload["metadata"]["n_missing_runs"] >= 0


def test_ablation_table_names_the_axis_each_row_varies(tmp_path):
    result = aggregate_matrix(tmp_path, n_resamples=100)
    latex = ablation_table_latex(result)
    assert "graph tokens deranged (control)" in latex
    assert "MLP encoder, no message passing" in latex
    assert "inference guard off" in latex
    assert "gate off (F1)" in latex


def test_latex_escapes_underscores_in_system_ids(tmp_path):
    result = aggregate_matrix(tmp_path, n_resamples=100)
    latex = main_table_latex(result)
    assert "A3_F3 &" not in latex
    assert r"A3\_F3 &" in latex


def test_latex_carries_no_stray_unicode_dashes(tmp_path):
    """`PairedComparison.marker` returns a U+2014 for the uncorrected case; pdflatex chokes."""
    result = aggregate_matrix(tmp_path, n_resamples=100)
    for latex in (
        main_table_latex(result),
        ablation_table_latex(result),
        taxonomy_table_latex(result),
    ):
        assert "—" not in latex


def test_aggregating_an_empty_matrix_succeeds_and_reports_everything_missing(tmp_path):
    """The reporting path must be exercisable before any arm has trained."""
    result = aggregate_matrix(tmp_path, n_resamples=100)
    assert result.rows == ()
    assert result.systems_present == ()
    assert len(result.missing) == sum(len(s.seeds) for s in all_systems())
    assert result.metadata["n_systems_present"] == 0
    # Every artifact still renders.
    assert main_table_latex(result)
    assert ablation_table_latex(result)
    assert taxonomy_table_latex(result)
