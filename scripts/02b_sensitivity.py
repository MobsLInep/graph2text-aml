#!/usr/bin/env python
"""Phase 2 sensitivity analysis over the two case-boundary parameters that matter.

The case boundary is constructed, not given, so a reviewer is entitled to ask what the
results would look like under a different boundary. Running the grid now costs a few
minutes and produces the appendix table that answers the question; not running it costs a
revision round.

The grid is ``k in {1, 2, 3}`` x ``n_max in {50, 100, 150, 300}``, evaluated over a fixed
sample of seeds so every cell sees the same cases. Three quantities are recorded:

**Case size distribution** — how big cases get, and how often the node budget binds.

**Typology recoverability** — the share of a seeding stream's laundering transactions that
survive into the case. This is the number that decides whether a parameter setting is
usable at all: a case that has lost the laundering path cannot be narrated faithfully.

**Label balance** — the share of cases that come out suspicious. Half the seed sample is
drawn uniformly rather than from a stream precisely so this number means something: at k=3
a licit seed reaches flagged activity three hops away often enough to change what
"negative" means, and a stream-only sample would report 100% at every cell.

Usage:
    uv run python scripts/02b_sensitivity.py
    uv run python scripts/02b_sensitivity.py sensitivity.n_seeds=100
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig, OmegaConf

from g2t_aml.data.canonical import CanonicalGraph
from g2t_aml.data.case_extraction import (
    CaseExtractionError,
    ExtractionParams,
    GraphIndex,
    TimeWindow,
    cut_case,
)
from g2t_aml.utils.io import write_json
from g2t_aml.utils.logging import configure_logging, get_logger, stage
from g2t_aml.utils.run_context import RunContext
from g2t_aml.utils.seeding import seed_everything

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
REPO_ROOT = Path(__file__).resolve().parents[1]

#: The grid. Fixed in code rather than config: it is the appendix table's definition, and
#: a table whose axes move between runs is not a table.
K_HOPS_GRID: tuple[int, ...] = (1, 2, 3)
N_MAX_GRID: tuple[int, ...] = (50, 100, 150, 300)


def _seed_sample(
    graph: CanonicalGraph, n_seeds: int, pad_hours: float, rng: np.random.Generator
) -> list[tuple[str, TimeWindow, str, int]]:
    """Draw the fixed seed sample shared by every grid cell.

    Half the seeds come from laundering streams and half from accounts drawn uniformly.
    Both halves are needed and they answer different questions. Typology recoverability is
    only defined for a seed that has a stream to recover, so it is measured on the first
    half. Label balance is only informative if some seeds *could* come out licit, so it is
    measured on the second — and that is where the interesting effect lives: at k=3 a
    licit seed reaches flagged activity three hops away often enough to change what
    "negative" means, which a stream-only sample cannot show at all.

    Windows here are uncapped, deliberately. The window duration cap is a separate
    parameter with its own trade-off (D-019); leaving it out isolates the effect of k and
    n_max, which is what this table is about.

    Args:
        graph: The substrate graph.
        n_seeds: How many seeds to draw in total.
        pad_hours: Window padding around each seed's activity extent.
        rng: Seeded generator.

    Returns:
        ``(seed_node, window, pattern_id, stream_size)`` per seed. ``pattern_id`` is empty
        and ``stream_size`` zero for a uniformly drawn seed, which excludes it from the
        recovery statistic while keeping it in the size and label statistics.

    Raises:
        CaseExtractionError: If the substrate carries no laundering streams.
    """
    if "pattern_id" not in graph.edges.columns:
        raise CaseExtractionError(f"{graph.dataset} has no pattern_id column to sample from")
    streams = (
        graph.edges.filter(pl.col("pattern_id").is_not_null())
        .group_by("pattern_id")
        .agg(
            pl.col("timestamp").min().alias("t0"),
            pl.col("timestamp").max().alias("t1"),
            pl.col("src").first().alias("seed"),
            pl.len().alias("n_txns"),
        )
        .sort("pattern_id")
    )
    if streams.is_empty():
        raise CaseExtractionError(f"{graph.dataset} has no laundering streams")
    rows = streams.to_dicts()
    n_stream_seeds = max(1, n_seeds // 2)
    chosen = rng.permutation(len(rows))[:n_stream_seeds]
    sample = [
        (
            str(rows[int(i)]["seed"]),
            TimeWindow(start=rows[int(i)]["t0"], end=rows[int(i)]["t1"]).padded(pad_hours),
            str(rows[int(i)]["pattern_id"]),
            int(rows[int(i)]["n_txns"]),
        )
        for i in sorted(chosen)
    ]

    # Uniformly drawn seeds, windowed to match the stream half's duration distribution so
    # window width is not silently confounded with seed type.
    durations = [w.duration for _, w, _, _ in sample]
    node_ids = graph.nodes["node_id"]
    spans = (
        graph.nodes.select("first_seen", "last_seen")
        if "first_seen" in graph.nodes.columns
        else None
    )
    for draw in range(n_seeds - n_stream_seeds):
        position = int(rng.integers(node_ids.len()))
        duration = durations[draw % len(durations)]
        start = (
            spans["first_seen"][position] if spans is not None else graph.edges["timestamp"].min()
        )
        sample.append(
            (str(node_ids[position]), TimeWindow(start=start, end=start + duration), "", 0)
        )
    return sample


def _evaluate_cell(
    graph: CanonicalGraph,
    index: GraphIndex,
    seeds: list[tuple[str, TimeWindow, str, int]],
    params: ExtractionParams,
) -> dict[str, Any]:
    """Extract every sampled case under one parameter setting and summarise the result.

    Args:
        graph: The substrate graph.
        index: The graph index.
        seeds: The shared seed sample.
        params: The cell's extraction parameters.

    Returns:
        One row of the appendix table.
    """
    sizes: list[int] = []
    edge_counts: list[int] = []
    recovery: list[float] = []
    suspicious = 0
    pruned = 0
    exceeded = 0
    failed = 0

    pattern_column = "pattern_id" in index.edges.columns
    for seed_node, window, pattern_id, stream_size in seeds:
        try:
            cut = cut_case(graph, seed_node, window, params, index=index)
        except CaseExtractionError:
            failed += 1
            continue
        sizes.append(int(cut.node_positions.size))
        edge_counts.append(int(cut.edge_positions.size))
        suspicious += int(cut.label == "suspicious")
        pruned += int(bool(cut.provenance["pruning_triggered"]))
        exceeded += int(bool(cut.provenance["n_max_exceeded"]))
        if pattern_column and stream_size and pattern_id:
            retained = index.edges["pattern_id"].gather(cut.edge_positions.tolist())
            recovery.append(float((retained == pattern_id).sum()) / stream_size)

    n = len(sizes)
    array = np.array(sizes, dtype=np.float64) if n else np.zeros(1)
    edges = np.array(edge_counts, dtype=np.float64) if n else np.zeros(1)
    recovered = np.array(recovery, dtype=np.float64) if recovery else np.zeros(1)
    return {
        "k_hops": params.k_hops,
        "n_max": params.n_max,
        "n_cases": n,
        "n_failed": failed,
        "nodes_median": float(np.median(array)),
        "nodes_p90": float(np.percentile(array, 90)),
        "nodes_max": float(array.max()),
        "edges_median": float(np.median(edges)),
        "edges_p90": float(np.percentile(edges, 90)),
        "pruning_rate": round(pruned / n, 4) if n else 0.0,
        "n_max_exceeded_rate": round(exceeded / n, 4) if n else 0.0,
        "typology_recovery_mean": round(float(recovered.mean()), 4),
        "typology_recovery_min": round(float(recovered.min()), 4),
        "full_recovery_rate": round(float((recovered >= 1.0).mean()), 4),
        "suspicious_rate": round(suspicious / n, 4) if n else 0.0,
    }


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    """Render the grid as the appendix table.

    Args:
        rows: One row per grid cell.

    Returns:
        A GitHub-flavoured Markdown table.
    """
    header = (
        "| k | n_max | cases | nodes p50 | nodes p90 | edges p50 | pruned | "
        "n_max exceeded | typology recovery | full recovery | suspicious |"
    )
    rule = "|" + "---|" * 11
    lines = [header, rule]
    for row in rows:
        lines.append(
            f"| {row['k_hops']} | {row['n_max']} | {row['n_cases']} | "
            f"{row['nodes_median']:.0f} | {row['nodes_p90']:.0f} | "
            f"{row['edges_median']:.0f} | {row['pruning_rate']:.1%} | "
            f"{row['n_max_exceeded_rate']:.1%} | {row['typology_recovery_mean']:.3f} | "
            f"{row['full_recovery_rate']:.1%} | {row['suspicious_rate']:.1%} |"
        )
    return "\n".join(lines) + "\n"


#: Exit code captured out of the Hydra-decorated entrypoint.
#:
#: ``@hydra.main`` **discards its wrapped function's return value** — it returns None
#: regardless — so the long-standing ``sys.exit(main())`` always exited 0 and every
#: documented "exits non-zero when the gate fails" in this repository was silently untrue.
#: A failing gate looked identical to a passing one to CI, to ``make``, and to any caller
#: checking ``$?``. Capturing the code out of a module-level cell is the smallest fix that
#: keeps the Hydra entrypoint shape. Found in Phase 5; see PHASE_LOG.
_EXIT_CODE: list[int] = []


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="config")
def _run(cfg: DictConfig) -> None:
    """Run the k x n_max sensitivity grid and write the appendix table.

    Args:
        cfg: Composed Hydra configuration.

    Returns:
        0 on success, or 0 with a recorded skip when the substrate is absent.
    """
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    configure_logging(log_file=run_dir / "sensitivity.log")
    log = get_logger(__name__)

    interim_dir = Path(cfg.paths.interim_dir) / str(cfg.data.interim_name)
    n_seeds = int(cfg.get("sensitivity", {}).get("n_seeds", 200))

    with stage("sensitivity", log, dataset=str(cfg.data.name), n_seeds=n_seeds) as summary:
        seeds_record = seed_everything(cfg.seed, deterministic=cfg.deterministic)
        RunContext.capture(
            experiment_name=cfg.experiment.name,
            cfg=cfg,
            seeds=seeds_record,
            repo_root=REPO_ROOT,
            phase="2",
        ).save(run_dir)

        if not (interim_dir / "canonical.json").is_file():
            log.warning("no interim graph at %s, skipping", interim_dir)
            write_json(run_dir / "sensitivity_skipped.json", {"reason": "interim graph absent"})
            summary["status"] = "skipped"
            _EXIT_CODE.append(0)
            return

        graph = CanonicalGraph.load(interim_dir)
        index = GraphIndex(graph)
        base = OmegaConf.to_container(cfg.cases.extraction, resolve=True)
        rng = np.random.default_rng(int(cfg.cases.sampling.seed))
        sample = _seed_sample(graph, n_seeds, float(cfg.cases.sampling.window_pad_hours), rng)
        log.info(
            "sampled %d stream seeds shared across all %d cells",
            len(sample),
            len(K_HOPS_GRID) * len(N_MAX_GRID),
        )

        rows: list[dict[str, Any]] = []
        for k_hops in K_HOPS_GRID:
            for n_max in N_MAX_GRID:
                params = ExtractionParams(
                    **{**base, "k_hops": k_hops, "n_max": n_max}  # type: ignore[arg-type]
                )
                row = _evaluate_cell(graph, index, sample, params)
                rows.append(row)
                log.info(
                    "k=%d n_max=%-3d  nodes p50 %3.0f  recovery %.3f  suspicious %.1f%%",
                    k_hops,
                    n_max,
                    row["nodes_median"],
                    row["typology_recovery_mean"],
                    row["suspicious_rate"] * 100,
                )

        payload = {
            "dataset": str(cfg.data.interim_name),
            "n_seeds": len(sample),
            "base_params": base,
            "grid": {"k_hops": list(K_HOPS_GRID), "n_max": list(N_MAX_GRID)},
            "rows": rows,
        }
        out_dir = Path(cfg.paths.metrics_dir) / "sensitivity"
        write_json(out_dir / f"case_extraction_{cfg.data.interim_name}.json", payload)
        table = _markdown_table(rows)
        (out_dir / f"case_extraction_{cfg.data.interim_name}.md").write_text(
            table, encoding="utf-8"
        )
        write_json(run_dir / "sensitivity.json", payload)
        log.info("wrote sensitivity table to %s", out_dir)
        print(table)

        summary["cells"] = len(rows)
        summary["status"] = "ok"

    _EXIT_CODE.append(0)
    return


def main() -> int:
    """Run the Hydra entrypoint and return its exit code.

    Returns:
        The code ``_run`` produced, or 0 when it produced none.
    """
    _run()
    return _EXIT_CODE[-1] if _EXIT_CODE else 0


if __name__ == "__main__":
    sys.exit(main())
