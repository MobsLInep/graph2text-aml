"""Phase 11: the paper's figures, from the aggregated metrics and nothing else.

Every figure here is a projection of the tidy table :mod:`g2t_aml.experiments.aggregate`
writes. Nothing recomputes a metric, and nothing reads a run directory: a figure that
disagrees with the results table is a class of bug this module cannot have, because both
read the same rows.

Four conventions, applied everywhere:

**Colourblind-safe.** :data:`PALETTE` is Okabe-Ito, which is distinguishable under
deuteranopia, protanopia and tritanopia, and it is also legible in greyscale print --
*Expert Systems with Applications* is read on paper as well as on screens. Systems are
additionally distinguished by hatch pattern in the stacked chart, so the encoding never
rests on hue alone.

**Vector output.** PDF by default, at a fixed figure width matching a two-column layout, so
a figure is not rescaled by the typesetter into unreadable tick labels.

**A missing number is a gap, not a zero.** A system with no data is absent from its axis
and named in the figure's own annotation. A bar of height zero and a bar that does not
exist are different claims, and the second one is what an unrun arm is.

**Error bars are the bootstrap CI where one exists**, and are omitted with a marked tick
where a system ran at one seed. The seed asymmetry is visible in the figures for the same
reason it is visible in the tables.

matplotlib is imported lazily inside each function. The module is imported by the
aggregation script on machines with no display and, in CI, with no matplotlib at all;
importing it at module scope would make a missing plotting dependency break the metrics
pipeline, which does not need it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from g2t_aml.experiments.aggregate import (
    HEADLINE_METRIC,
    TAXONOMY_CLASSES,
    AggregateResult,
)
from g2t_aml.experiments.registry import all_systems
from g2t_aml.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover -- typing only
    from matplotlib.figure import Figure

__all__ = [
    "FIGURE_FORMAT",
    "PALETTE",
    "efficiency_frontier",
    "faithfulness_vs_fluency",
    "hallucination_breakdown",
    "main_comparison",
    "render_all",
    "s1_vs_a1",
    "typology_heatmap",
]

log = get_logger(__name__)

#: Okabe-Ito. Safe under all three common colour-vision deficiencies and legible in
#: greyscale. Not chosen for looks: a reviewer printing the paper in black and white must
#: still be able to tell the treatment from its control.
PALETTE: tuple[str, ...] = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
)

#: Vector. A raster figure in a submission is a figure a typesetter will resample.
FIGURE_FORMAT = "pdf"

#: Single-column width in inches for a two-column Elsevier layout.
_COL_WIDTH = 3.5
_FULL_WIDTH = 7.2

#: Systems drawn in a distinct role colour rather than the default cycle, because these
#: three carry the argument and a reader should find them without consulting a legend.
_ROLE_COLOURS: Mapping[str, str] = {
    "S1": PALETTE[0],
    "S2": PALETTE[2],
    "A1": PALETTE[1],
    "B7": PALETTE[4],
    "B1": PALETTE[7],
}

_DEFAULT_COLOUR = "#8C8C8C"


def _style() -> dict[str, Any]:
    """Return the rcParams every figure sets.

    Returns:
        A mapping suitable for ``matplotlib.rc_context``. Fixed here rather than in a
        stylesheet file so a figure rendered from a test fixture looks like a figure
        rendered from the real metrics.
    """
    return {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "pdf.fonttype": 42,  # embed TrueType, not Type 3: some venues reject Type 3
        "ps.fonttype": 42,
    }


def _colour_for(system: str, index: int) -> str:
    """Return the colour a system is drawn in.

    Args:
        system: The system id.
        index: Its position, for systems with no assigned role colour.

    Returns:
        A hex colour.
    """
    return _ROLE_COLOURS.get(system, PALETTE[index % len(PALETTE)] if index else _DEFAULT_COLOUR)


def _save(fig: Figure, path: Path | str) -> Path:
    """Write a figure and close it.

    Args:
        fig: The figure.
        path: Destination; the suffix decides the format.

    Returns:
        The path written.
    """
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    log.info("wrote %s", out)
    return out


def _series(
    result: AggregateResult, metric: str, stream: str
) -> tuple[list[str], list[float], list[tuple[float, float] | None], list[bool]]:
    """Pull one metric across systems, in registry order.

    Args:
        result: The aggregation.
        metric: The metric.
        stream: The stream.

    Returns:
        ``(systems, means, ci_bounds, single_seed_flags)``, covering only systems that
        produced a number. A system with no data is simply absent -- see the module
        docstring on why it is not a zero.
    """
    systems: list[str] = []
    means: list[float] = []
    bounds: list[tuple[float, float] | None] = []
    single: list[bool] = []
    for spec in all_systems():
        summary = result.summaries.get((stream, metric, spec.system_id))
        if summary is None:
            continue
        systems.append(spec.system_id)
        means.append(summary.mean)
        interval = result.intervals.get((stream, metric, spec.system_id))
        bounds.append((interval.lo, interval.hi) if interval is not None else None)
        single.append(summary.std is None)
    return systems, means, bounds, single


def _annotate_absences(ax: Any, shown: Sequence[str]) -> None:
    """Name the systems that produced no number for this figure.

    Absences are computed against the REGISTRY, not against the aggregation: a system that
    produced no metrics and a system nobody declared are different things, and only the
    registry knows which systems the matrix promised.

    Args:
        ax: The axis to annotate.
        shown: The systems that are drawn.

    Returns:
        None. The annotation is the point: a bar chart of eleven systems when the matrix
        declares sixteen must say which five are missing, on the figure, not in a caption
        that gets separated from it.
    """
    absent = [spec.system_id for spec in all_systems() if spec.system_id not in set(shown)]
    if not absent:
        return
    ax.text(
        0.99,
        0.02,
        "not run: " + ", ".join(absent),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        style="italic",
        color="#555555",
    )


def main_comparison(
    result: AggregateResult,
    path: Path | str,
    *,
    metric: str = HEADLINE_METRIC,
    stream: str = "balanced",
) -> Path:
    """Draw the main comparison bar chart with confidence intervals.

    Args:
        result: The aggregation.
        path: Destination.
        metric: Which metric; the headline by default.
        stream: Which stream.

    Returns:
        The path written.
    """
    import matplotlib.pyplot as plt

    systems, means, bounds, single = _series(result, metric, stream)
    with plt.rc_context(_style()):
        fig, ax = plt.subplots(figsize=(_FULL_WIDTH, 2.8))
        colours = [_colour_for(s, i) for i, s in enumerate(systems)]
        positions = range(len(systems))
        bars = ax.bar(positions, means, color=colours, width=0.68, edgecolor="none")

        # The bar height is the ACROSS-SEED mean; the interval is the PER-CASE bootstrap.
        # They answer different questions and are computed from different data, so the
        # mean can fall marginally outside its own interval -- most visibly on a
        # multi-seed system whose seeds disagree. Clamping to zero keeps the whisker
        # honest (it never points the wrong way) without silently moving either number.
        lower = [max(0.0, m - (b[0] if b else m)) for m, b in zip(means, bounds, strict=True)]
        upper = [max(0.0, (b[1] if b else m) - m) for m, b in zip(means, bounds, strict=True)]
        if any(b is not None for b in bounds):
            ax.errorbar(
                list(positions),
                means,
                yerr=[lower, upper],
                fmt="none",
                ecolor="#222222",
                elinewidth=0.9,
                capsize=2.5,
            )
        # A single-seed bar gets a hollow marker above it. The seed asymmetry is a
        # property of the result and belongs on the result, not only in the caption.
        for pos, bar, is_single in zip(positions, bars, single, strict=True):
            if is_single:
                ax.plot(
                    pos,
                    bar.get_height() + 0.02,
                    marker="o",
                    markersize=3,
                    markerfacecolor="none",
                    markeredgecolor="#222222",
                    markeredgewidth=0.7,
                )

        ax.set_xticks(list(positions))
        ax.set_xticklabels(systems, rotation=0)
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{metric.replace('_', ' ')} by system ({stream} stream)")
        ax.text(
            0.01,
            0.97,
            "○ = single seed (no variance estimate)",
            transform=ax.transAxes,
            va="top",
            fontsize=6,
            color="#555555",
        )
        _annotate_absences(ax, systems)
        return _save(fig, path)


def s1_vs_a1(
    result: AggregateResult,
    path: Path | str,
    *,
    metrics: Sequence[str] = ("zero_hallucination_rate", "fact_precision", "fact_coverage"),
    stream: str = "balanced",
    treatment: str = "S1",
    control: str = "A1",
) -> Path:
    """Draw the sanity control beside the treatment, prominently.

    **This is the figure Gate 8 lives in.** A1 is S1 with every narrative paired with a
    different case's graph, identical in every other respect. If the two bars are the same
    height, the fusion layer is decoration and the paper's contribution is the corpus and
    the evaluation framework -- and that is a finding this figure has to be capable of
    showing, which is why it is drawn at the same scale as the main comparison rather than
    tucked into an appendix at half size.

    Args:
        result: The aggregation.
        path: Destination.
        metrics: The axes to compare on.
        stream: Which stream.
        treatment: The treatment arm.
        control: The control arm.

    Returns:
        The path written.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    with plt.rc_context(_style()):
        fig, ax = plt.subplots(figsize=(_COL_WIDTH, 2.6))
        width = 0.36
        x = np.arange(len(metrics))
        drawn = False
        for offset, (system, colour) in enumerate(
            (
                (treatment, _ROLE_COLOURS.get(treatment, PALETTE[0])),
                (control, _ROLE_COLOURS.get(control, PALETTE[1])),
            )
        ):
            values: list[float] = []
            errs_lo: list[float] = []
            errs_hi: list[float] = []
            for metric in metrics:
                summary = result.summaries.get((stream, metric, system))
                interval = result.intervals.get((stream, metric, system))
                value = float("nan") if summary is None else summary.mean
                values.append(value)
                if interval is not None and summary is not None:
                    errs_lo.append(max(0.0, summary.mean - interval.lo))
                    errs_hi.append(max(0.0, interval.hi - summary.mean))
                else:
                    errs_lo.append(0.0)
                    errs_hi.append(0.0)
            drawn = drawn or any(v == v for v in values)  # noqa: PLR0124 -- NaN test
            ax.bar(
                x + (offset - 0.5) * width,
                values,
                width=width,
                label=system,
                color=colour,
                edgecolor="none",
            )
            ax.errorbar(
                x + (offset - 0.5) * width,
                values,
                yerr=[errs_lo, errs_hi],
                fmt="none",
                ecolor="#222222",
                elinewidth=0.9,
                capsize=2.0,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in metrics])
        ax.set_ylabel("score")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{treatment} vs {control}: the sanity control")
        ax.legend(frameon=False, loc="upper right")
        if not drawn:
            ax.text(
                0.5,
                0.5,
                f"{treatment} and {control} have not been run.\nGate 8 is open.",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color="#B00020",
            )
        return _save(fig, path)


def faithfulness_vs_fluency(
    result: AggregateResult,
    path: Path | str,
    *,
    faithfulness: str = HEADLINE_METRIC,
    fluency: str = "layer1.rouge_l",
    stream: str = "balanced",
) -> Path:
    """Scatter faithfulness against a fluency proxy: does the trade-off exist?

    The question is whether faithfulness is bought with fluency. A negative slope is a
    finding; no slope is also a finding, and the more useful one for an applications paper
    arguing that a verified generator is deployable.

    Args:
        result: The aggregation.
        path: Destination.
        faithfulness: The faithfulness metric for the y axis.
        fluency: The Layer 1 metric for the x axis. **Blocked on Gold**: Layer 1 is
            scored against human references only, so this axis is empty until Phase 6
            produces one. The figure renders with a stated absence rather than failing.
        stream: Which stream.

    Returns:
        The path written.
    """
    import matplotlib.pyplot as plt

    with plt.rc_context(_style()):
        fig, ax = plt.subplots(figsize=(_COL_WIDTH, 2.8))
        points = 0
        for index, spec in enumerate(all_systems()):
            y = result.summaries.get((stream, faithfulness, spec.system_id))
            x = result.summaries.get((stream, fluency, spec.system_id))
            if y is None or x is None:
                continue
            points += 1
            ax.scatter(
                x.mean,
                y.mean,
                s=32,
                color=_colour_for(spec.system_id, index),
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            ax.annotate(
                spec.system_id,
                (x.mean, y.mean),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=6,
            )
        ax.set_xlabel(fluency.replace("_", " "))
        ax.set_ylabel(faithfulness.replace("_", " "))
        ax.set_title("Faithfulness against fluency")
        if points == 0:
            ax.text(
                0.5,
                0.5,
                f"No system has both {faithfulness}\nand {fluency}.\n"
                "Layer 1 is scored against Gold,\nwhich does not yet exist.",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=7,
                color="#555555",
            )
        return _save(fig, path)


def hallucination_breakdown(
    result: AggregateResult,
    path: Path | str,
    *,
    classes: Sequence[str] = TAXONOMY_CLASSES,
    stream: str = "balanced",
) -> Path:
    """Draw the stacked per-class hallucination rate for every system.

    Args:
        result: The aggregation.
        path: Destination.
        classes: The classes, in taxonomy order.
        stream: Which stream.

    Returns:
        The path written.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    systems = [
        spec.system_id for spec in all_systems() if (stream, spec.system_id) in result.taxonomy
    ]
    # Hatches as well as hue: nine classes exceed any colourblind-safe palette, so the
    # stack is distinguished on two channels rather than one.
    hatches = ("", "///", "...", "\\\\\\", "xxx", "|||", "---", "+++", "ooo")

    with plt.rc_context(_style()):
        fig, ax = plt.subplots(figsize=(_FULL_WIDTH, 2.8))
        bottom = np.zeros(len(systems))
        for index, klass in enumerate(classes):
            values = np.array(
                [result.taxonomy[(stream, s)].get(klass, 0.0) for s in systems], dtype=float
            )
            ax.bar(
                systems,
                values,
                bottom=bottom,
                label=klass,
                color=PALETTE[index % len(PALETTE)],
                hatch=hatches[index % len(hatches)],
                edgecolor="white",
                linewidth=0.4,
            )
            bottom = bottom + values
        ax.set_ylabel("per-narrative rate")
        ax.set_title("Hallucination classes by system")
        ax.legend(frameon=False, ncol=len(classes), loc="upper center", bbox_to_anchor=(0.5, -0.12))
        if not systems:
            ax.text(
                0.5,
                0.5,
                "No system has a taxonomy breakdown.",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color="#555555",
            )
        _annotate_absences(ax, systems)
        return _save(fig, path)


def typology_heatmap(
    result: AggregateResult,
    path: Path | str,
    *,
    metric: str = HEADLINE_METRIC,
    stream: str = "balanced",
) -> Path:
    """Draw system by typology performance as a heatmap.

    Phase 7's linear probe reached 0.33 structural macro-F1 on the pooled tokens: fan_out,
    gather_scatter and cycle are recoverable and stack and random are not. This figure is
    where that forecast is checked against what the generator actually did, so the per-cell
    values matter more than the row means.

    Args:
        result: The aggregation.
        path: Destination.
        metric: Which metric.
        stream: Which stream.

    Returns:
        The path written.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [
        r for r in result.rows if r.metric == metric and r.stream == stream and r.typology != "all"
    ]
    systems = [s.system_id for s in all_systems() if any(r.system == s.system_id for r in rows)]
    typologies = sorted({r.typology for r in rows})

    grid = np.full((len(systems), len(typologies)), np.nan)
    for i, system in enumerate(systems):
        for j, typology in enumerate(typologies):
            values = [r.value for r in rows if r.system == system and r.typology == typology]
            if values:
                grid[i, j] = sum(values) / len(values)

    with plt.rc_context(_style()):
        fig, ax = plt.subplots(figsize=(_FULL_WIDTH, max(2.0, 0.28 * len(systems) + 1.2)))
        # A sequential, perceptually uniform map: viridis is the one colourblind-safe
        # default matplotlib ships, and NaN cells are drawn in a neutral grey so a missing
        # cell is visibly missing rather than reading as the low end of the scale.
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#DDDDDD")
        image = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(typologies)))
        ax.set_xticklabels(typologies, rotation=45, ha="right")
        ax.set_yticks(range(len(systems)))
        ax.set_yticklabels(systems)
        ax.grid(visible=False)
        for i in range(len(systems)):
            for j in range(len(typologies)):
                if not np.isnan(grid[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{grid[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if grid[i, j] < 0.6 else "black",  # noqa: PLR2004
                    )
        fig.colorbar(image, ax=ax, shrink=0.8, label=metric.replace("_", " "))
        ax.set_title(f"{metric.replace('_', ' ')} by typology")
        if grid.size == 0:
            ax.text(
                0.5,
                0.5,
                "No per-typology values.",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#555555",
            )
        return _save(fig, path)


def efficiency_frontier(
    result: AggregateResult,
    path: Path | str,
    *,
    latencies: Mapping[str, float] | None = None,
    efficiency: Mapping[str, Any] | None = None,
    faithfulness: Mapping[str, float] | None = None,
    metric: str = HEADLINE_METRIC,
    stream: str = "balanced",
) -> Path:
    """Draw faithfulness against latency, sized by VRAM, shaped by deployability.

    This is the figure that carries the deployment argument, and it carries it in four
    channels at once: faithfulness on y, end-to-end latency on x, marker area for the VRAM
    a row needs, and marker shape for whether it runs inside the institution's perimeter. A
    reader who takes nothing else from the paper should be able to take from this figure
    that the on-premise points are not the slow, weak corner of the plot.

    **Absences are drawn as absences.** A system with no latency measurement is not
    plotted, and the count of such systems is annotated on the axes -- a scatter that
    silently omits thirteen of seventeen rows is a scatter that overstates what was
    measured.

    Args:
        result: The aggregation. Supplies faithfulness where the matrix has run.
        path: Destination.
        latencies: System id to median seconds per narrative. Superseded by ``efficiency``
            when that is given; kept so the Phase 11 call site is unchanged.
        efficiency: The Phase 13 table as written by
            :meth:`~g2t_aml.eval.efficiency.EfficiencyTable.write_json`. Supplies latency,
            VRAM and the on-premise flag.
        faithfulness: System id to a faithfulness score, used where the aggregation has no
            summary for a system. This is how a Phase 10 measurement -- Bronze's
            Zero-Hallucination Rate -- reaches the figure while the matrix is unrun; it is
            the same metric computed by the same code, read from a different file.
        metric: The faithfulness metric.
        stream: Which stream.

    Returns:
        The path written.
    """
    import matplotlib.pyplot as plt

    latency = dict(latencies or {})
    vram: dict[str, float] = {}
    on_premise: dict[str, bool] = {}
    scores = dict(faithfulness or {})

    if efficiency is not None:
        for row in efficiency.get("rows", []):
            system_id = str(row.get("system_id", ""))
            guard_off = row.get("latency_guard_off")
            if row.get("measured") and isinstance(guard_off, dict):
                latency[system_id] = float(guard_off["p50_s"])
            memory = row.get("memory") or {}
            reserved = memory.get("inference_reserved_gb")
            if reserved is not None:
                vram[system_id] = float(reserved)
            deployment = row.get("deployment") or {}
            if "on_premise" in deployment:
                on_premise[system_id] = bool(deployment["on_premise"])

    # Fallback for the executors, used only where the Phase 13 table did not say.
    local_executors = {"trained_generator", "local_zero_shot", "template", "classifier_template"}

    # Marker area scales with VRAM, floored so a row that needs no accelerator is still
    # visible rather than vanishing. The floor is stated in the legend: a small marker
    # means "little or no device memory", not "unknown".
    min_area, max_area = 25.0, 260.0
    largest = max(vram.values(), default=0.0)

    def _area(system_id: str) -> float:
        needed = vram.get(system_id, 0.0)
        if largest <= 0:
            return min_area
        return min_area + (max_area - min_area) * (needed / largest)

    with plt.rc_context(_style()):
        fig, ax = plt.subplots(figsize=(_COL_WIDTH, 2.9))
        plotted = 0
        missing: list[str] = []
        for index, spec in enumerate(all_systems()):
            summary = result.summaries.get((stream, metric, spec.system_id))
            score = summary.mean if summary is not None else scores.get(spec.system_id)
            if score is None or spec.system_id not in latency:
                missing.append(spec.system_id)
                continue
            plotted += 1
            is_local = on_premise.get(spec.system_id, str(spec.executor) in local_executors)
            ax.scatter(
                latency[spec.system_id],
                score,
                s=_area(spec.system_id),
                marker="o" if is_local else "^",
                color=_colour_for(spec.system_id, index),
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            ax.annotate(
                spec.system_id,
                (latency[spec.system_id], score),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=6,
            )
        if plotted:
            ax.set_xscale("log")
        # A rate lives in [0, 1] and the axis says so. Autoscaling around a single point
        # produced a y-range of 0.96-1.05, which draws a 4-point band as if it were the
        # whole scale and makes one measurement look like a spread of them.
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("end-to-end seconds per narrative (p50, log scale)")
        ax.set_ylabel(metric.replace("_", " "))
        # The coverage note belongs in the title, not floating in the axes: it qualifies
        # the whole figure, and every other placement collided with either the title or
        # the legend.
        title = "Efficiency frontier"
        if missing:
            title += f"\n({len(missing)} of {len(missing) + plotted} systems unmeasured)"
        ax.set_title(title, fontsize=8)
        ax.text(
            0.01,
            0.04,
            "\u25cf on-premise   \u25b2 external API\nmarker area \u221d inference VRAM",
            transform=ax.transAxes,
            fontsize=6,
            color="#555555",
            va="bottom",
        )
        if plotted == 0:
            ax.text(
                0.5,
                0.5,
                "No latency measurements.\nPhase 13 has not run.",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=7,
                color="#555555",
            )
        return _save(fig, path)


def render_all(
    result: AggregateResult,
    out_dir: Path | str,
    *,
    stream: str = "balanced",
    latencies: Mapping[str, float] | None = None,
    suffix: str = FIGURE_FORMAT,
) -> dict[str, Path]:
    """Render every figure the paper needs.

    Args:
        result: The aggregation.
        out_dir: Destination directory.
        stream: Which stream to draw. Streams are never pooled.
        latencies: Phase 13's latency measurements, if available.
        suffix: Output format.

    Returns:
        Figure name to the path written. A figure whose data is entirely absent is still
        rendered, carrying its stated absence, because a missing file in the figures
        directory is indistinguishable from a build that did not run.
    """
    out = Path(out_dir)
    written: dict[str, Path] = {
        "main_comparison": main_comparison(
            result, out / f"fig_main_comparison.{suffix}", stream=stream
        ),
        "s1_vs_a1": s1_vs_a1(result, out / f"fig_s1_vs_a1.{suffix}", stream=stream),
        "faithfulness_vs_fluency": faithfulness_vs_fluency(
            result, out / f"fig_faithfulness_vs_fluency.{suffix}", stream=stream
        ),
        "hallucination_breakdown": hallucination_breakdown(
            result, out / f"fig_hallucination_breakdown.{suffix}", stream=stream
        ),
        "typology_heatmap": typology_heatmap(
            result, out / f"fig_typology_heatmap.{suffix}", stream=stream
        ),
        "efficiency_frontier": efficiency_frontier(
            result, out / f"fig_efficiency_frontier.{suffix}", latencies=latencies, stream=stream
        ),
    }
    return written
