#!/usr/bin/env python
"""Phase 2: build cases, split them temporally, and audit the split for leakage.

The pipeline is linear and fails closed:

    load interim graph -> extract + sample cases -> temporal split -> measure overlap
                       -> leakage audit -> realistic-imbalance stream -> write manifests

A hard failure in the audit exits non-zero and writes nothing to ``schemas/splits/``,
because a committed manifest is a promise that the split was checked. Invariant 2: this
script is run deliberately, never as part of training, and its output is a committed file.

Usage:
    uv run python scripts/02_build_cases.py                     # AMLworld HI-Small
    uv run python scripts/02_build_cases.py cases=debug         # 400 cases, for smoke
    uv run python scripts/02_build_cases.py data=elliptic2      # skips if access gated
    uv run python scripts/02_build_cases.py cases.split.overlap_mode=strict
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from g2t_aml.data.canonical import CanonicalGraph
from g2t_aml.data.case_extraction import ExtractionParams, GraphIndex, TimeWindow
from g2t_aml.data.case_sampling import (
    CaseCollection,
    SamplingParams,
    build_realistic_stream,
    sample_cases,
)
from g2t_aml.data.leakage_audit import (
    LeakageReport,
    audit_splits,
    audit_temporal_disjointness,
)
from g2t_aml.data.splits import (
    SplitAssignment,
    SplitParams,
    apply_overlap_mode,
    build_manifest,
    measure_overlap,
    temporal_split,
    write_split_manifest,
)
from g2t_aml.utils.hashing import short
from g2t_aml.utils.io import read_json, write_json
from g2t_aml.utils.logging import configure_logging, get_logger, stage
from g2t_aml.utils.run_context import RunContext
from g2t_aml.utils.seeding import seed_everything

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
REPO_ROOT = Path(__file__).resolve().parents[1]


#: Hard negatives must be at least this share of the negative population in every split.
#: Mirrors ``case_sampling.MINIMUM_HARD_NEGATIVE_RATE``, applied per split rather than to
#: the corpus as a whole -- a split without hard negatives cannot support the claim they
#: exist to test, however healthy the corpus-wide rate looks.
MINIMUM_SPLIT_HARD_NEGATIVE_RATE = 0.20


class CaseBuildError(RuntimeError):
    """Raised when the built case population cannot satisfy the Phase 2 gate."""


def _source_manifest_hash(interim_dir: Path) -> str:
    """Return a digest identifying the interim graph cases are cut from.

    Cases are stored as positions into that graph, so applying them to a different one
    would silently produce different cases with the same identifiers.

    Args:
        interim_dir: The substrate's interim directory.

    Returns:
        The short digest of the interim manifest's artifact list, or ``"unknown"`` when no
        manifest is present — which only happens for a hand-built fixture.
    """
    manifest_path = interim_dir / "manifest.json"
    if not manifest_path.is_file():
        return "unknown"
    manifest = read_json(manifest_path)
    return short(str(manifest.get("artifacts", "")).encode("utf-8").hex()[:64], 16)


def _extraction_params(cfg: DictConfig) -> ExtractionParams:
    """Build the extraction protocol from config.

    Args:
        cfg: Composed Hydra configuration.

    Returns:
        The parameters.

    Raises:
        ValueError: If a parameter is out of range.
    """
    block = OmegaConf.to_container(cfg.cases.extraction, resolve=True)
    return ExtractionParams(**block)  # type: ignore[arg-type]


def _split_params(cfg: DictConfig) -> SplitParams:
    """Build the split plan from config.

    Args:
        cfg: Composed Hydra configuration.

    Returns:
        The parameters.

    Raises:
        ValueError: If the plan is invalid.
    """
    block = dict(OmegaConf.to_container(cfg.cases.split, resolve=True))  # type: ignore[arg-type]
    block["proportions"] = tuple(block["proportions"])
    return SplitParams(**block)  # type: ignore[arg-type]


def _check_gate(manifest: dict[str, Any], log: Any) -> None:
    """Assert the built population satisfies the Phase 2 acceptance criteria.

    Args:
        manifest: The split manifest.
        log: Logger.

    Raises:
        CaseBuildError: If hard negatives fall below 20% of the negative population in
            any split. The corpus's central claim is that hard negatives are where a
            generator overclaims; a split without them cannot support it.
    """
    for name in ("train", "val", "test"):
        rate = manifest["stratification"][name]["hard_negative_rate"]
        if rate < MINIMUM_SPLIT_HARD_NEGATIVE_RATE:
            raise CaseBuildError(
                f"{name} split has a hard-negative rate of {rate:.1%}, below the "
                f"{MINIMUM_SPLIT_HARD_NEGATIVE_RATE:.0%} gate"
            )
        log.info("%s hard-negative rate %.1f%%", name, rate * 100)


def _record_run_context(cfg: DictConfig, run_dir: Path) -> None:
    """Seed every RNG and write the run's provenance sidecars.

    Invariant 5: every run records its git SHA, resolved config, seeds and library
    versions before it does any work.

    Args:
        cfg: Composed Hydra configuration.
        run_dir: This run's output directory.
    """
    seeds = seed_everything(cfg.seed, deterministic=cfg.deterministic)
    write_json(run_dir / "resolved_config.json", OmegaConf.to_container(cfg, resolve=True))
    RunContext.capture(
        experiment_name=cfg.experiment.name,
        cfg=cfg,
        seeds=seeds,
        repo_root=REPO_ROOT,
        phase="2",
    ).save(run_dir)


def _build_realistic(
    cfg: DictConfig,
    graph: CanonicalGraph,
    index: GraphIndex,
    extraction: ExtractionParams,
    collection: CaseCollection,
    assignment: SplitAssignment,
    log: Any,
) -> tuple[CaseCollection, TimeWindow]:
    """Build the realistic-imbalance stream over the test split's window.

    Args:
        cfg: Composed Hydra configuration.
        graph: The substrate graph.
        index: An index over ``graph``.
        extraction: The case-construction protocol, identical to the balanced corpus.
        collection: The balanced corpus, supplying the test split's temporal extent.
        assignment: The split assignment.
        log: Logger.

    Returns:
        The stream and the window it was drawn from.

    Raises:
        CaseSamplingError: If no account is active in the test window.
    """
    test_records = [collection.by_id()[cid] for cid in assignment.splits["test"]]
    window = TimeWindow(
        start=min(r.window_start for r in test_records),
        end=max(r.window_end for r in test_records),
    )
    log.info("building realistic-imbalance stream over %s..%s", window.start, window.end)
    stream = build_realistic_stream(
        graph,
        index,
        extraction,
        window=window,
        n_cases=int(cfg.cases.realistic.n_cases),
        seed=int(cfg.cases.realistic.seed),
        target_prevalence=cfg.cases.realistic.target_prevalence,
        source_manifest_hash=collection.source_manifest_hash,
    )
    log.info(
        "realistic stream: %d cases at observed prevalence %.4f",
        len(stream),
        stream.stratification["observed_prevalence"],
    )
    return stream, window


def _write_artifacts(
    manifest: dict[str, Any],
    collection: CaseCollection,
    realistic: CaseCollection,
    report: LeakageReport,
    cases_dir: Path,
    processed_dir: Path,
    run_dir: Path,
) -> None:
    """Write the case corpus, the realistic stream, the audit and the split record.

    Everything here lands under gitignored ``data/`` and ``artifacts/``. The one artifact
    that gets committed — the split manifest under ``schemas/splits/`` — is written by the
    caller and only after the audit has passed.

    Args:
        manifest: The split manifest.
        collection: The balanced case corpus.
        realistic: The realistic-imbalance stream.
        report: The leakage audit.
        cases_dir: Destination for the case corpus.
        processed_dir: Destination for the audit and the split record.
        run_dir: This run's Hydra output directory.

    Raises:
        OSError: If a write or rename fails.
    """
    collection.save(cases_dir)
    realistic.save(cases_dir / "realistic_test")
    report.save(processed_dir / "leakage_audit.json")
    report.save(run_dir / "leakage_audit.json")
    write_json(processed_dir / "splits.json", manifest, canonical=True)


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
    """Build, split and audit the Phase 2 case corpus.

    Args:
        cfg: Composed Hydra configuration.

    Returns:
        0 on success, or 0 with a recorded skip when the substrate is absent. Non-zero when
        the leakage audit records a hard failure.

    Raises:
        CaseBuildError: If the built population misses a gate criterion.
    """
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    configure_logging(log_file=run_dir / "build_cases.log")
    log = get_logger(__name__)

    interim_dir = Path(cfg.paths.interim_dir) / str(cfg.data.interim_name)
    processed_dir = Path(cfg.paths.processed_dir) / str(cfg.data.interim_name)
    cases_dir = processed_dir / "cases"

    with stage(
        "build_cases",
        log,
        dataset=str(cfg.data.name),
        interim_dir=str(interim_dir),
        cases_dir=str(cases_dir),
        run_dir=str(run_dir),
    ) as summary:
        _record_run_context(cfg, run_dir)

        if not (interim_dir / "canonical.json").is_file():
            # Elliptic2 is access-gated; Phase 2 does not block on a substrate we may not
            # have been granted yet, exactly as Phase 1 does not.
            log.warning("no interim graph at %s, skipping", interim_dir)
            write_json(
                run_dir / "cases_skipped.json",
                {"dataset": str(cfg.data.name), "reason": "interim graph not present"},
            )
            summary["status"] = "skipped: interim graph not present"
            _EXIT_CODE.append(0)
            return

        # --------------------------------------------------------- extract ---
        log.info("loading interim graph")
        graph = CanonicalGraph.load(interim_dir)
        index = GraphIndex(graph)
        log.info("indexed %d nodes, %d edges", index.num_nodes, index.num_edges)

        extraction = _extraction_params(cfg)
        sampling = SamplingParams(
            **OmegaConf.to_container(cfg.cases.sampling, resolve=True)  # type: ignore[arg-type]
        )
        log.info("sampling %d cases", sampling.n_cases)
        collection = sample_cases(
            graph,
            index,
            extraction,
            sampling,
            source_manifest_hash=_source_manifest_hash(interim_dir),
        )
        log.info("built %d cases: %s", len(collection), collection.stratification["by_class"])

        # ----------------------------------------------------------- split ---
        assignment = temporal_split(collection.records, _split_params(cfg))
        overlap = measure_overlap(collection, assignment)
        assignment = apply_overlap_mode(assignment, overlap)
        if assignment.params.overlap_mode == "strict":
            overlap = measure_overlap(collection, assignment)
        log.info(
            "split %s, dropped %d (%s)",
            assignment.counts,
            len(assignment.dropped),
            assignment.drop_reasons(),
        )
        log.info("node overlap rate %.3f", overlap.node_overlap_rate)

        # ----------------------------------------------------------- audit ---
        manifest = build_manifest(collection, assignment, overlap)
        report = audit_splits(
            collection,
            manifest,
            node_feature_names=list(graph.node_feature_names),
            edge_feature_names=list(graph.edge_feature_names),
        )

        # ------------------------------------- realistic-imbalance stream ---
        realistic, realistic_window = _build_realistic(
            cfg, graph, index, extraction, collection, assignment, log
        )
        report.findings.append(
            audit_temporal_disjointness(
                [collection.by_id()[cid] for cid in assignment.splits["train"]],
                realistic.records,
            )
        )

        # ----------------------------------------------------------- write ---
        manifest["leakage_audit"] = report.summary()
        manifest["realistic_stream"] = {
            "n": len(realistic),
            "window": realistic_window.to_dict(),
            **realistic.stratification,
        }
        _write_artifacts(manifest, collection, realistic, report, cases_dir, processed_dir, run_dir)

        if not report.passed:
            # Nothing is committed. A manifest under schemas/ asserts the split was
            # checked and passed, so a failing audit must not produce one.
            log.error("leakage audit hard failures: %s", [f.check for f in report.hard_failures])
            for finding in report.hard_failures:
                log.error("  %s: %s", finding.check, finding.detail)
            summary["status"] = "failed: leakage audit"
            _EXIT_CODE.append(1)
            return

        _check_gate(manifest, log)
        manifest_path = write_split_manifest(manifest, Path(cfg.data.split.manifest_dir))
        log.info("wrote committed split manifest to %s", manifest_path)

        summary["n_cases"] = len(collection)
        summary["splits"] = assignment.counts
        summary["node_overlap_rate"] = overlap.node_overlap_rate
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
