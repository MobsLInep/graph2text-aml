#!/usr/bin/env python
"""Build the calibration set the project lead writes reference answers into.

Writes ``<processed>/<substrate>/gold/calibration.json`` — the chosen cases, each with an
empty ``reference_narrative`` and ``reference_typology`` prefilled from the fact record.
**The lead fills in the narratives and the commentary before anyone calibrates**;
:func:`~g2t_aml.human.calibration.score_annotator` refuses to score against a set whose
references are still blank, rather than passing everyone against nothing.

The set spans typologies rather than following the Gold sample's proportions. Ten cases
drawn in proportion would be eight ``unclassified`` ones and would calibrate nobody on the
eight typed shapes, which is exactly where the systematic errors live.

Usage:
    uv run python scripts/06d_build_calibration.py corpus=gold
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from g2t_aml.human.calibration import CalibrationSet, build_calibration_set
from g2t_aml.utils.io import read_json, write_json
from g2t_aml.utils.logging import configure_logging, get_logger, stage
from g2t_aml.utils.run_context import RunContext

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Exit code captured out of the Hydra-decorated entrypoint; see D-051.
_EXIT_CODE: list[int] = []


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="config")
def _run(cfg: DictConfig) -> None:
    """Build the calibration set from the drawn Gold sample.

    Args:
        cfg: Composed Hydra configuration.

    Returns:
        Nothing. The exit code is recorded in `_EXIT_CODE`; 0 on success, 1 when no Gold
        sample has been drawn.
    """
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    configure_logging(log_file=run_dir / "calibration.log")
    log = get_logger(__name__)

    if str(cfg.corpus.tier) != "gold":
        log.error("run this with `corpus=gold`; corpus.tier is %r", str(cfg.corpus.tier))
        _EXIT_CODE.append(1)
        return

    processed_dir = Path(cfg.paths.processed_dir) / str(cfg.data.interim_name)
    gold_dir = processed_dir / "gold"
    sample_path = gold_dir / "gold_sample.json"

    with stage("calibration", log, sample=str(sample_path)) as summary:
        if not sample_path.is_file():
            log.error("no Gold sample at %s; run `make gold-sample` first", sample_path)
            summary["status"] = "skipped: no sample"
            _EXIT_CODE.append(1)
            return

        payload = read_json(sample_path)
        candidates = [
            (str(c["case_id"]), str(c["typology"])) for c in payload.get("per_case") or ()
        ]
        log.info("%d cases in the Gold sample", len(candidates))

        destination = gold_dir / "calibration.json"
        if destination.is_file():
            existing = CalibrationSet.from_dict(read_json(destination))
            written = sum(1 for i in existing.items if i.reference_narrative.strip())
            log.warning(
                "a calibration set already exists at %s with %d of %d references written. "
                "Refusing to overwrite it: annotators may already have calibrated against "
                "it, and a set that changed underneath them scores them against two "
                "different standards. Delete it deliberately to rebuild.",
                destination,
                written,
                len(existing),
            )
            summary["status"] = "skipped: already exists"
            _EXIT_CODE.append(0)
            return

        built = build_calibration_set(
            candidates,
            n_cases=int(cfg.corpus.calibration.n_cases),
            seed=int(cfg.corpus.calibration.seed),
            drawn_from=str(sample_path.relative_to(REPO_ROOT)),
        )
        write_json(destination, built.to_dict(), canonical=True)
        log.info("wrote %d calibration items to %s", len(built), destination)
        log.warning(
            "every reference_narrative is EMPTY. The project lead writes them, with "
            "commentary, before any annotator calibrates; scoring against blank "
            "references is refused."
        )

        by_typology: dict[str, int] = {}
        for item in built.items:
            by_typology[item.reference_typology] = by_typology.get(item.reference_typology, 0) + 1
        for typology, n in sorted(by_typology.items()):
            log.info("  %-16s %d", typology, n)

        RunContext.capture(
            experiment_name=cfg.experiment.name,
            cfg=cfg,
            seeds={"global": int(cfg.seed), "calibration": int(cfg.corpus.calibration.seed)},
            repo_root=REPO_ROOT,
            phase="6",
        ).save(run_dir)

        summary["n_items"] = len(built)
        summary["n_typologies"] = len(by_typology)
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
