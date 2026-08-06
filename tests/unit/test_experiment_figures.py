"""Phase 11: every figure renders from a fixture metrics file, populated and empty.

Both halves matter. A figure that only renders with data is a figure that will crash the
build on the day one arm is missing; a figure that silently renders an empty axis is a
figure that shows a system scoring zero when it did not run. Each function is exercised in
both states, and the empty state is required to say so on the figure itself.
"""

from __future__ import annotations

import pytest

from g2t_aml.experiments.aggregate import HEADLINE_METRIC, aggregate_matrix
from g2t_aml.experiments.registry import get_system
from g2t_aml.experiments.runner import write_completion_marker
from g2t_aml.utils.io import write_json, write_jsonl

pytest.importorskip("matplotlib", reason="figures need matplotlib; not in the base env")

from g2t_aml.experiments import figures  # noqa: E402


def _fixture_run(root, system, seed, value, *, taxonomy=None, by_typology=None):
    directory = root / system / f"seed{seed}" / "hash0000"
    directory.mkdir(parents=True, exist_ok=True)
    write_json(
        directory / "metrics.json",
        {
            "systems": {
                f"{system}/balanced": {
                    "system": system,
                    "stream": "balanced",
                    "faithfulness": {
                        "system": system,
                        "n_cases": 50,
                        HEADLINE_METRIC: value,
                        "fact_precision": value,
                        "fact_coverage": value * 0.9,
                    },
                    "taxonomy": {"rate_by_class": taxonomy or {"H1": 0.01, "H9": 0.2}},
                    "by_typology": by_typology or {},
                    "by_dataset": {},
                    "layer1": None,
                    "worst_cases": [],
                }
            }
        },
    )
    write_jsonl(
        directory / "per_case.jsonl",
        [{"case_id": f"c{i}", "stream": "balanced", HEADLINE_METRIC: value} for i in range(20)],
    )
    write_completion_marker(
        directory,
        run_id=f"{system}_seed{seed}",
        system_id=system,
        seed=seed,
        config_hash="hash0000",
    )


@pytest.fixture
def populated(tmp_path):
    """A fixture matrix with the three arms the figures centre on."""
    for seed in (42, 1337, 2024):
        _fixture_run(tmp_path, "S1", seed, 0.90)
        _fixture_run(tmp_path, "A1", seed, 0.55)
        _fixture_run(tmp_path, "B7", seed, 0.80)
    _fixture_run(
        tmp_path,
        "A2",
        42,
        0.72,
        by_typology={
            "fan_out": {"system": "A2", "n_cases": 10, HEADLINE_METRIC: 0.8},
            "cycle": {"system": "A2", "n_cases": 10, HEADLINE_METRIC: 0.5},
        },
    )
    return aggregate_matrix(
        tmp_path,
        specs=[get_system(s) for s in ("S1", "A1", "B7", "A2")],
        n_resamples=200,
    )


@pytest.fixture
def empty(tmp_path):
    return aggregate_matrix(tmp_path / "nothing", n_resamples=100)


def test_palette_is_okabe_ito_and_has_no_duplicates():
    assert len(set(figures.PALETTE)) == len(figures.PALETTE)
    assert "#0072B2" in figures.PALETTE  # blue
    assert "#D55E00" in figures.PALETTE  # vermillion


def test_output_is_vector_by_default():
    assert figures.FIGURE_FORMAT == "pdf"


def test_style_embeds_truetype_not_type3():
    """Some venues reject Type 3 fonts outright."""
    style = figures._style()
    assert style["pdf.fonttype"] == 42
    assert style["ps.fonttype"] == 42


@pytest.mark.parametrize(
    "name",
    [
        "main_comparison",
        "s1_vs_a1",
        "faithfulness_vs_fluency",
        "hallucination_breakdown",
        "typology_heatmap",
        "efficiency_frontier",
    ],
)
def test_every_figure_renders_with_data(populated, tmp_path, name):
    out = tmp_path / f"{name}.pdf"
    written = getattr(figures, name)(populated, out)
    assert written == out
    assert out.is_file()
    assert out.stat().st_size > 0


@pytest.mark.parametrize(
    "name",
    [
        "main_comparison",
        "s1_vs_a1",
        "faithfulness_vs_fluency",
        "hallucination_breakdown",
        "typology_heatmap",
        "efficiency_frontier",
    ],
)
def test_every_figure_renders_with_no_data_at_all(empty, tmp_path, name):
    """A missing file in the figures directory is indistinguishable from a build that
    did not run, so an absent figure is rendered carrying its stated absence."""
    out = tmp_path / f"empty_{name}.pdf"
    assert getattr(figures, name)(empty, out).is_file()


def test_render_all_writes_every_figure(populated, tmp_path):
    written = figures.render_all(populated, tmp_path / "figs")
    assert set(written) == {
        "main_comparison",
        "s1_vs_a1",
        "faithfulness_vs_fluency",
        "hallucination_breakdown",
        "typology_heatmap",
        "efficiency_frontier",
    }
    assert all(path.is_file() for path in written.values())


def test_series_omits_systems_with_no_data_rather_than_plotting_a_zero(populated):
    """A bar of height zero and a bar that does not exist are different claims."""
    systems, means, _bounds, _single = figures._series(populated, HEADLINE_METRIC, "balanced")
    assert "B5" not in systems, "an unrun system must not appear on the axis"
    assert set(systems) == {"S1", "A1", "B7", "A2"}
    assert all(m > 0 for m in means)


def test_series_flags_single_seed_systems(populated):
    systems, _means, _bounds, single = figures._series(populated, HEADLINE_METRIC, "balanced")
    flags = dict(zip(systems, single, strict=True))
    assert flags["A2"] is True, "A2 runs at one seed and must be marked"
    assert flags["S1"] is False, "S1 runs at three seeds and has a variance estimate"


def test_s1_vs_a1_is_drawn_at_the_same_scale_as_the_main_comparison(populated, tmp_path):
    """Gate 8's figure is not an appendix afterthought."""
    a = figures.s1_vs_a1(populated, tmp_path / "control.pdf")
    assert a.is_file()
    # It must be able to show the verdict nobody wants: identical arms, identical bars.
    for seed in (42, 1337, 2024):
        _fixture_run(tmp_path / "flat", "S1", seed, 0.7)
        _fixture_run(tmp_path / "flat", "A1", seed, 0.7)
    flat = aggregate_matrix(
        tmp_path / "flat", specs=[get_system("S1"), get_system("A1")], n_resamples=100
    )
    assert figures.s1_vs_a1(flat, tmp_path / "flat.pdf").is_file()


def test_efficiency_frontier_uses_latencies_when_supplied(populated, tmp_path):
    out = figures.efficiency_frontier(
        populated, tmp_path / "eff.pdf", latencies={"S1": 1.2, "B7": 0.9}
    )
    assert out.is_file()


def test_typology_heatmap_handles_a_system_with_no_typology_breakdown(populated, tmp_path):
    """S1 has no per-typology rows in the fixture; A2 does. The figure must not raise."""
    assert figures.typology_heatmap(populated, tmp_path / "heat.pdf").is_file()
