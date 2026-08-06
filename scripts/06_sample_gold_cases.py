#!/usr/bin/env python
"""Select the Gold annotation sample and reserve it test-only in the split manifest.

Writes three things:

- ``schemas/splits/<substrate>/gold_reserved.txt`` — the committed id list, reviewable in
  a diff exactly like ``test.txt``.
- ``schemas/splits/<substrate>/gold_reservation.json`` — the machine-readable record, with
  the id-list hash and the sampling provenance.
- ``<processed>/<substrate>/gold/gold_sample.json`` — the stratification report: what was
  asked for, what was drawn, and every shortfall.

**Nothing here regenerates a split** (invariant 2). The reservation is a subset of the
existing, frozen test split recorded beside it; ``test.txt`` and its sha256 are untouched,
and the loader asserts that every reserved id really is a test-split member.

**Exits non-zero** when the sample cannot meet the hard-negative floor or the minimum
reservation size. Both are reasons to fix the population or the parameters rather than to
proceed with a Gold set that cannot do its job.

Usage:
    uv run python scripts/06_sample_gold_cases.py corpus=gold
    uv run python scripts/06_sample_gold_cases.py corpus=gold corpus.sampling.n_cases=300
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import hydra
import polars as pl
from omegaconf import DictConfig

from g2t_aml.corpus.validate import load_split_manifest
from g2t_aml.data.case_sampling import CaseCollection
from g2t_aml.human.reservation import GoldReservation, write_reservation
from g2t_aml.human.sampling import (
    GoldCandidate,
    GoldSamplingError,
    GoldSamplingParams,
    sample_gold_cases,
)
from g2t_aml.utils.io import write_json
from g2t_aml.utils.logging import configure_logging, get_logger, stage
from g2t_aml.utils.run_context import RunContext

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
REPO_ROOT = Path(__file__).resolve().parents[1]

#: The columns the sampler needs out of the Phase 3 fact aggregate.
_FACT_COLUMNS = ("case_id", "typology", "structure.n_nodes", "structure.n_edges")


def _candidates_for(
    processed_dir: Path, manifest_dir: Path, interim_name: str, split: str
) -> list[GoldCandidate]:
    """Build the candidate pool for one substrate.

    Three sources, deliberately: the frozen manifest decides the split, ``facts.parquet``
    supplies the typology the salience list keys on, and the Phase 2 case index supplies
    the case class. The typology comes from the *fact record* rather than the case index
    because the record is what an annotator and the automated metric both read (D-036).

    Args:
        processed_dir: The substrate's processed directory.
        manifest_dir: Its frozen split manifest directory.
        interim_name: The substrate key.
        split: The split candidates must belong to.

    Returns:
        The candidates, empty when the substrate has not been ingested.
    """
    facts_parquet = processed_dir / "facts.parquet"
    cases_dir = processed_dir / "cases"
    if not facts_parquet.is_file() or not (cases_dir / "cases.jsonl").is_file():
        return []
    if not (manifest_dir / f"{split}.txt").is_file():
        return []

    assignment = load_split_manifest(manifest_dir)
    wanted = {cid for cid, name in assignment.items() if name == split}

    frame = pl.read_parquet(facts_parquet).select(_FACT_COLUMNS)
    classes = {r.case_id: r.case_class for r in CaseCollection.load(cases_dir).records}

    candidates: list[GoldCandidate] = []
    for row in frame.iter_rows(named=True):
        case_id = str(row["case_id"])
        if case_id not in wanted or case_id not in classes:
            continue
        candidates.append(
            GoldCandidate(
                case_id=case_id,
                dataset=interim_name,
                split=split,
                typology=str(row["typology"]),
                case_class=str(classes[case_id]),
                n_nodes=int(row["structure.n_nodes"]),
                n_edges=int(row["structure.n_edges"]),
            )
        )
    return candidates


def _params_from(cfg: DictConfig) -> GoldSamplingParams:
    """Build the sampling parameters from the composed configuration.

    Args:
        cfg: The composed configuration.

    Returns:
        The parameters.
    """
    sampling = cfg.corpus.sampling
    return GoldSamplingParams(
        n_cases=int(sampling.n_cases),
        min_reserved=int(sampling.min_reserved),
        hard_negative_share=float(sampling.hard_negative_share),
        typed_share=float(sampling.typed_share),
        unclassified_share=float(sampling.unclassified_share),
        hard_negative_floor=float(sampling.hard_negative_floor),
        substrate_shares={str(s.interim_name): float(s.share) for s in sampling.substrates},
        split=str(sampling.split),
        seed=int(sampling.seed),
    )


#: Exit code captured out of the Hydra-decorated entrypoint. `@hydra.main` discards the
#: wrapped function's return value; see D-051.
_EXIT_CODE: list[int] = []


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="config")
def _run(cfg: DictConfig) -> None:
    """Draw the Gold sample and write its reservation.

    Args:
        cfg: Composed Hydra configuration.

    Returns:
        Nothing. The exit code is recorded in `_EXIT_CODE`; 0 on success, 1 when no
        substrate has data or the sample cannot meet its floors.
    """
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    configure_logging(log_file=run_dir / "gold_sample.log")
    log = get_logger(__name__)

    if str(cfg.corpus.tier) != "gold":
        log.error(
            "this script samples the Gold tier but corpus.tier is %r; run it with " "`corpus=gold`",
            str(cfg.corpus.tier),
        )
        _EXIT_CODE.append(1)
        return

    params = _params_from(cfg)
    processed_root = Path(cfg.paths.processed_dir)

    with stage("gold-sample", log, n_cases=params.n_cases, split=params.split) as summary:
        candidates: list[GoldCandidate] = []
        manifest_dirs: dict[str, Path] = {}
        for substrate in cfg.corpus.sampling.substrates:
            interim_name = str(substrate.interim_name)
            manifest_dir = Path(substrate.manifest_dir)
            manifest_dirs[interim_name] = manifest_dir
            found = _candidates_for(
                processed_root / interim_name, manifest_dir, interim_name, params.split
            )
            log.info("%s: %d candidates in the %s split", interim_name, len(found), params.split)
            if not found:
                log.warning(
                    "%s supplied no candidates; its %.0f%% quota will be reported as a "
                    "deficit rather than reallocated",
                    interim_name,
                    100 * params.substrate_shares.get(interim_name, 0.0),
                )
            candidates += found

        if not candidates:
            log.error(
                "no substrate supplied a candidate. Run `make facts` and `make cases` "
                "first; Gold cannot be sampled from a population that is not on disk."
            )
            summary["status"] = "skipped: no candidates"
            _EXIT_CODE.append(1)
            return

        try:
            sample = sample_gold_cases(candidates, params)
        except GoldSamplingError as exc:
            log.error("the Gold sample was refused: %s", exc)
            summary["status"] = "sampling refused"
            _EXIT_CODE.append(1)
            return

        print(sample.summary())

        created_at = datetime.now(UTC).isoformat()
        provenance: dict[str, Any] = {
            "script": "scripts/06_sample_gold_cases.py",
            "params": params.to_dict(),
            "deficits": {
                name: {"requested": req, "supplied": got}
                for name, (req, got) in sorted(sample.deficits.items())
            },
        }

        for dataset in sorted(sample.by_dataset):
            ids = [c.case_id for c in sample.selected if c.dataset == dataset]
            reservation = GoldReservation(
                dataset=dataset,
                case_ids=tuple(ids),
                split=params.split,
                created_at=created_at,
                provenance=provenance | {"n_for_this_substrate": len(ids)},
            )
            path = write_reservation(reservation, manifest_dirs[dataset])
            log.info("reserved %d %s cases test-only: %s", len(ids), dataset, path)

        report = sample.to_dict() | {
            "created_at": created_at,
            "candidates_considered": len(candidates),
            "per_case": [c.to_dict() for c in sample.selected],
        }
        for dataset in sorted(sample.by_dataset):
            gold_dir = processed_root / dataset / "gold"
            write_json(gold_dir / "gold_sample.json", report, canonical=True)
        write_json(run_dir / "gold_sample.json", report, canonical=True)

        RunContext.capture(
            experiment_name=cfg.experiment.name,
            cfg=cfg,
            seeds={"global": int(cfg.seed), "sampling": params.seed},
            repo_root=REPO_ROOT,
            phase="6",
        ).save(run_dir)

        summary["n_selected"] = len(sample)
        summary["hard_negative_rate"] = round(sample.hard_negative_rate, 4)
        summary["n_deficits"] = len(sample.deficits)
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
