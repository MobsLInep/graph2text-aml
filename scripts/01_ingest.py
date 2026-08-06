#!/usr/bin/env python
"""Phase 1 ingest: verify raw data, normalise it, and write the interim artifacts.

The pipeline is deliberately linear and fails closed at every step:

    verify checksums -> load -> build canonical -> compute statistics
                     -> write Parquet -> write manifest -> write statistics report

Nothing downstream is trustworthy if this stage is sloppy, so a checksum mismatch, a
changed CSV header, or a node/edge count that disagrees with the published figures all
abort the run rather than warn. The one exception is Elliptic2, which is access-gated:
when its files are absent the run reports that and exits cleanly, because Phase 1 does not
block on a dataset we may not have been granted yet.

Usage:
    uv run python scripts/01_ingest.py                    # AMLworld HI-Small
    uv run python scripts/01_ingest.py data=elliptic2
    uv run python scripts/01_ingest.py data.size=LI-Small
    uv run python scripts/01_ingest.py ingest.n_rows=50000   # subset, recorded in manifest
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from g2t_aml.data import canonical as canonical_module
from g2t_aml.data.canonical import CanonicalGraph
from g2t_aml.data.download import (
    REGISTRY,
    DataUnavailableError,
    verify,
)
from g2t_aml.data.loaders import amlworld, elliptic2
from g2t_aml.data.stats import compute_dataset_statistics
from g2t_aml.utils.hashing import hash_config, hash_file, short
from g2t_aml.utils.io import write_json
from g2t_aml.utils.logging import configure_logging, get_logger, stage
from g2t_aml.utils.run_context import RunContext
from g2t_aml.utils.seeding import seed_everything

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
REPO_ROOT = Path(__file__).resolve().parents[1]


class IngestError(RuntimeError):
    """Raised when ingest produces output that disagrees with the published figures."""


def _ingest_amlworld(cfg: DictConfig, log: Any) -> tuple[CanonicalGraph, dict[str, Any]]:
    """Load AMLworld and build its canonical account graph.

    Args:
        cfg: Composed Hydra configuration.
        log: Logger.

    Returns:
        The graph, and a dict of extra facts for the statistics report — the typology counts
        observed against those published, and the pattern-stream count.

    Raises:
        IngestError: If observed node/edge counts or typology counts disagree with the
            published figures and the run was not explicitly subsetted.
    """
    raw_dir = Path(cfg.paths.raw_dir)
    size = str(cfg.data.size)
    n_rows = cfg.ingest.n_rows

    log.info("loading transactions (%s)", size)
    txns = amlworld.load_transactions(size, raw_dir=raw_dir, n_rows=n_rows)
    log.info("loaded %d transactions", txns.height)

    log.info("parsing patterns file")
    patterns = amlworld.load_patterns(size, raw_dir=raw_dir)
    log.info(
        "parsed %d pattern transactions across %d streams",
        patterns.height,
        patterns["pattern_id"].n_unique(),
    )

    labelled = amlworld.attach_typologies(txns, patterns)
    graph = amlworld.build_account_graph(
        labelled,
        graph_id=str(cfg.data.interim_name),
        dataset=str(cfg.data.interim_name),
        provenance={
            "size": size,
            "loader": "g2t_aml.data.loaders.amlworld",
            "node_key": "bank|account",
            "subsetted_n_rows": n_rows,
        },
    )

    observed_typologies = amlworld.typology_counts(patterns, txns)
    published_typologies = amlworld.PUBLISHED_TYPOLOGY_COUNTS.get(size, {})
    typology_table = {
        key: {
            "published": value,
            "observed": observed_typologies.get(key),
            "matches": observed_typologies.get(key) == value,
        }
        for key, value in published_typologies.items()
    }
    counts_table = amlworld.verify_published_statistics(graph, size)

    extra = {
        "num_pattern_streams": int(patterns["pattern_id"].n_unique()),
        "num_pattern_transactions": int(patterns.height),
        "typology_counts_observed": observed_typologies,
        "published_comparison": {"counts": counts_table, "typologies": typology_table},
    }

    # A subsetted run, or a run over the fixture slice, cannot reproduce the published
    # figures and is not expected to. Both cases still compute and record the comparison
    # table above, so the mismatch is visible in the statistics report rather than hidden;
    # what is skipped is only the abort.
    if n_rows is not None:
        log.warning(
            "run is subsetted to %d rows: published-statistic checks are skipped and the "
            "manifest records the subset",
            n_rows,
        )
        return graph, extra
    if not cfg.data.get("verify_published", True):
        log.warning(
            "data.verify_published is false for %s: the observed-vs-published table is "
            "written to the statistics report but disagreement will not abort the run",
            cfg.data.name,
        )
        return graph, extra

    failures = [k for k, v in counts_table.items() if not v["matches"]]
    failures += [f"typology:{k}" for k, v in typology_table.items() if not v["matches"]]
    if failures:
        raise IngestError(
            f"observed statistics disagree with the published figures for {size}: "
            f"{failures}\n{OmegaConf.to_yaml(OmegaConf.create(extra['published_comparison']))}"
        )
    log.info("observed counts and all typology counts match the published figures")
    return graph, extra


def _ingest_elliptic2(cfg: DictConfig, log: Any) -> tuple[CanonicalGraph, dict[str, Any]]:
    """Load Elliptic2's labelled subgraph index and build one representative graph.

    The full 122K-subgraph materialisation is Phase 2 work (case extraction). Phase 1 only
    establishes that the substrate loads, that its schema is what the documentation says,
    and that its availability mask is correct.

    Args:
        cfg: Composed Hydra configuration.
        log: Logger.

    Returns:
        A canonical graph for the first labelled subgraph, and summary facts.

    Raises:
        elliptic2.Elliptic2UnavailableError: If the files are absent.
    """
    raw_dir = Path(cfg.paths.raw_dir)
    memberships = elliptic2.load_labelled_subgraphs(raw_dir)
    summary = elliptic2.subgraph_labels(memberships)
    log.info("loaded %d labelled subgraphs", summary.height)

    background = elliptic2.load_background_graph(raw_dir)
    first = str(summary["subgraph_id"][0])
    graph = elliptic2.build_subgraph(
        first, raw_dir=raw_dir, memberships=memberships, background=background
    )

    label_counts = {
        str(row["label"]): int(row["len"])
        for row in summary.group_by("label").len().sort("label").to_dicts()
    }
    return graph, {
        "num_labelled_subgraphs": int(summary.height),
        "label_distribution": label_counts,
        "representative_subgraph_id": first,
        "anonymised_feature_columns": list(background.feature_columns),
        "published_comparison": {
            "counts": {
                "num_labelled_subgraphs": {
                    "published": elliptic2.PUBLISHED_STATISTICS["num_labelled_subgraphs"],
                    "observed": int(summary.height),
                    "note": "published figure is rounded to 122K",
                }
            }
        },
    }


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
    """Run the Phase 1 ingest for the configured substrate.

    Args:
        cfg: Composed Hydra configuration.

    Returns:
        0 on success, or 0 with a recorded skip when an access-gated substrate is absent.

    Raises:
        IngestError: If observed statistics disagree with the published figures.
        ChecksumMismatchError: If a raw file does not match its registered digest.
    """
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    configure_logging(log_file=run_dir / "ingest.log")
    log = get_logger(__name__)

    dataset_key = str(cfg.data.registry_key)
    interim_dir = Path(cfg.paths.interim_dir) / str(cfg.data.interim_name)

    with stage(
        "ingest",
        log,
        dataset=dataset_key,
        size=cfg.data.get("size"),
        raw_dir=str(cfg.paths.raw_dir),
        interim_dir=str(interim_dir),
        run_dir=str(run_dir),
    ) as summary:
        seeds = seed_everything(cfg.seed, deterministic=cfg.deterministic)
        write_json(run_dir / "resolved_config.json", OmegaConf.to_container(cfg, resolve=True))
        RunContext.capture(
            experiment_name=cfg.experiment.name,
            cfg=cfg,
            seeds=seeds,
            repo_root=REPO_ROOT,
            phase="1",
        ).save(run_dir)

        # ---------------------------------------------------------- verify ---
        report = verify(
            dataset_key, cfg.paths.raw_dir, compute_checksums=cfg.ingest.verify_checksums
        )
        write_json(run_dir / "verification.json", report.to_dict())
        try:
            report.raise_for_status()
        except DataUnavailableError as exc:
            if REGISTRY[dataset_key].redistributable:
                raise
            # Access-gated and not granted yet. Record and exit cleanly (see module docs).
            log.warning("%s is unavailable, skipping: %s", dataset_key, exc)
            write_json(
                run_dir / "ingest_skipped.json",
                {"dataset": dataset_key, "reason": "access-gated data not present"},
            )
            summary["status"] = "skipped: access-gated data not present"
            _EXIT_CODE.append(0)
            return
        log.info("checksum verification passed for %s", dataset_key)

        # ------------------------------------------------------------ load ---
        if dataset_key.startswith("amlworld"):
            graph, extra = _ingest_amlworld(cfg, log)
        else:
            graph, extra = _ingest_elliptic2(cfg, log)

        if cfg.ingest.validate_referential_integrity:
            log.info("validating referential integrity")
            graph.validate_referential_integrity()

        # ------------------------------------------------------- statistics ---
        log.info("computing dataset statistics")
        statistics = compute_dataset_statistics(
            graph, include_components=cfg.ingest.compute_components
        )
        statistics["extra"] = extra
        write_json(run_dir / "statistics.json", statistics)

        # ------------------------------------------------------------ write ---
        graph.save(interim_dir)
        log.info("wrote canonical graph to %s", interim_dir)
        write_json(interim_dir / "statistics.json", statistics)

        # -------------------------------------------------------- manifest ---
        artifacts = sorted(
            p for p in interim_dir.iterdir() if p.is_file() and p.name != "manifest.json"
        )
        manifest = {
            "dataset": dataset_key,
            "interim_name": str(cfg.data.interim_name),
            "canonical_schema_version": canonical_module.CANONICAL_SCHEMA_VERSION,
            "config_hash": short(hash_config(cfg)),
            "size": cfg.data.get("size"),
            # Invariant: a subsetted ingest is never silent.
            "subsetted_n_rows": cfg.ingest.n_rows,
            "is_complete_dataset": cfg.ingest.n_rows is None,
            "raw_verification": report.to_dict(),
            "graph": graph.summary(),
            "artifacts": [
                {"name": p.name, "size_bytes": p.stat().st_size, "sha256": hash_file(p)}
                for p in artifacts
            ],
        }
        write_json(interim_dir / "manifest.json", manifest, canonical=True)
        write_json(run_dir / "manifest.json", manifest, canonical=True)

        summary["num_nodes"] = graph.num_nodes
        summary["num_edges"] = graph.num_edges
        summary["interim_dir"] = str(interim_dir)
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
