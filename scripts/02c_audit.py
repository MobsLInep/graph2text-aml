#!/usr/bin/env python
"""Re-run the leakage audit over already-built cases and a committed split manifest.

The auditor is standalone by design, so any phase can invoke it — before training, before
evaluation, or in CI once the corpus is available. It reads only artifacts: it does not
re-derive the split, so it will notice if a manifest has drifted from the corpus it
partitions.

Exits non-zero on any hard failure, which makes it usable as a gate.

Usage:
    uv run python scripts/02c_audit.py
    uv run python scripts/02c_audit.py data=elliptic2
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from g2t_aml.data.canonical import CanonicalGraph
from g2t_aml.data.case_sampling import CaseCollection
from g2t_aml.data.leakage_audit import audit_splits
from g2t_aml.data.splits import load_split_manifest
from g2t_aml.utils.io import write_json
from g2t_aml.utils.logging import configure_logging, get_logger, stage

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


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
    """Audit the committed split manifest against the built case corpus.

    Args:
        cfg: Composed Hydra configuration.

    Returns:
        0 when the audit passes or the corpus is absent, 1 on any hard failure.
    """
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    configure_logging(log_file=run_dir / "audit.log")
    log = get_logger(__name__)

    processed_dir = Path(cfg.paths.processed_dir) / str(cfg.data.interim_name)
    cases_dir = processed_dir / "cases"
    manifest_dir = Path(cfg.data.split.manifest_dir)
    interim_dir = Path(cfg.paths.interim_dir) / str(cfg.data.interim_name)

    with stage("audit", log, cases_dir=str(cases_dir), manifest_dir=str(manifest_dir)) as summary:
        if not (cases_dir / "cases.jsonl").is_file():
            log.warning("no case corpus at %s; run `make cases` first", cases_dir)
            summary["status"] = "skipped: no case corpus"
            _EXIT_CODE.append(0)
            return
        if not (manifest_dir / "splits.json").is_file():
            log.warning("no split manifest at %s; run `make cases` first", manifest_dir)
            summary["status"] = "skipped: no split manifest"
            _EXIT_CODE.append(0)
            return

        collection = CaseCollection.load(cases_dir)
        manifest = load_split_manifest(manifest_dir)
        graph = CanonicalGraph.load(interim_dir)

        report = audit_splits(
            collection,
            manifest,
            node_feature_names=list(graph.node_feature_names),
            edge_feature_names=list(graph.edge_feature_names),
        )
        report.save(run_dir / "leakage_audit.json")
        write_json(run_dir / "audit_summary.json", report.summary())

        for finding in report.findings:
            level = log.error if (finding.severity == "fatal" and not finding.passed) else log.info
            level("[%s] %s: %s", finding.severity, finding.check, finding.detail)

        summary["passed"] = report.passed
        summary["status"] = "ok" if report.passed else "failed"
        if not report.passed:
            _EXIT_CODE.append(1)
            return

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
