#!/usr/bin/env python
"""Ingest reviewed annotations into ``gold.jsonl``, gated by the ten-point harness.

Reads the annotation store and the review log, builds ``training_record_v1`` records with
``tier="gold"``, gates them with the **same** ten checks that gate Bronze and Silver, and
writes:

- ``corpus/gold.jsonl`` — the records, schema-validated.
- ``corpus/gold_validation.json`` — the ten-point harness report.
- ``corpus/gold_quality.json`` — ingestion's own report: what was held and why, per-rule
  flag override rates, salience coverage, adjudication counts.
- ``corpus/gold_agreement.json`` — inter-annotator agreement over the double-annotated
  items.

**Exits non-zero when the gate fails.** A human-authored record failing the harness is a
record to revise, not a threshold to lower — and unlike Bronze, where a failure is a
renderer bug, here it usually means an annotator stated a number the record does not carry,
which is exactly what the phase exists to catch before it becomes the reference standard.

Usage:
    uv run python scripts/06b_ingest_gold.py corpus=gold
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from g2t_aml.corpus.factsio import load_case_facts_file
from g2t_aml.corpus.record import BronzeNarrative
from g2t_aml.corpus.tokenization import get_token_counter
from g2t_aml.corpus.validate import load_split_manifest, validate_corpus, write_report
from g2t_aml.facts.schema import CASE_FACTS_SCHEMA_VERSION, CaseFacts
from g2t_aml.facts.vocab import load_vocabulary
from g2t_aml.human.agreement import measure_agreement
from g2t_aml.human.gold_ingest import (
    GoldIngestError,
    bronze_narrative_from_record,
    ingest_annotations,
)
from g2t_aml.human.reservation import load_reservation
from g2t_aml.human.review import ReviewLog
from g2t_aml.human.store import AnnotationStore
from g2t_aml.utils.io import read_jsonl, write_json, write_jsonl
from g2t_aml.utils.logging import configure_logging, get_logger, stage
from g2t_aml.utils.run_context import RunContext

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_supporting(
    facts_dir: Path, bronze_path: Path, case_ids: set[str]
) -> tuple[dict[str, CaseFacts], dict[str, BronzeNarrative]]:
    """Load the fact records and Bronze slot alignments the ingestion needs.

    Args:
        facts_dir: Directory holding one fact record per case.
        bronze_path: The Bronze corpus file, read **only** for its slot annotations.
        case_ids: The cases to load, so the whole corpus is not held in memory.

    Returns:
        ``(facts_by_case, bronze_by_case)``. A case missing from either is simply absent,
        and ingestion holds it with a stated reason rather than failing the whole run.
    """
    facts: dict[str, CaseFacts] = {}
    for case_id in sorted(case_ids):
        path = facts_dir / f"{case_id}.json"
        if path.is_file():
            facts[case_id] = load_case_facts_file(path)

    bronze: dict[str, BronzeNarrative] = {}
    if bronze_path.is_file():
        for payload in read_jsonl(bronze_path):
            if not isinstance(payload, dict) or str(payload.get("case_id")) not in case_ids:
                continue
            try:
                bronze[str(payload["case_id"])] = bronze_narrative_from_record(payload)
            except GoldIngestError:
                continue
    return facts, bronze


#: Exit code captured out of the Hydra-decorated entrypoint; see D-051.
_EXIT_CODE: list[int] = []


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="config")
def _run(cfg: DictConfig) -> None:  # noqa: PLR0915 - one linear ingest-gate-write pass
    """Ingest, gate and write the Gold corpus.

    Args:
        cfg: Composed Hydra configuration.

    Returns:
        Nothing. The exit code is recorded in `_EXIT_CODE`; 0 on success, 1 when the
        schema version disagrees or the ten-point gate fails.
    """
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    configure_logging(log_file=run_dir / "gold.log")
    log = get_logger(__name__)

    if str(cfg.corpus.tier) != "gold":
        log.error("run this with `corpus=gold`; corpus.tier is %r", str(cfg.corpus.tier))
        _EXIT_CODE.append(1)
        return

    declared = str(cfg.schema_version.case_facts)
    if declared != CASE_FACTS_SCHEMA_VERSION:
        log.error(
            "config declares case_facts schema %s but the code is frozen at %s (invariant 3)",
            declared,
            CASE_FACTS_SCHEMA_VERSION,
        )
        _EXIT_CODE.append(1)
        return

    processed_dir = Path(cfg.paths.processed_dir) / str(cfg.data.interim_name)
    gold_dir = processed_dir / "gold"
    corpus_dir = processed_dir / "corpus"
    manifest_dir = Path(cfg.data.split.manifest_dir)

    with stage("gold-ingest", log, gold=str(gold_dir), out=str(corpus_dir)) as summary:
        store = AnnotationStore(root=gold_dir / "annotations")
        annotations = store.read_all()
        if not annotations:
            log.warning(
                "no annotations under %s. Run the interface first: "
                "`uv run streamlit run src/g2t_aml/human/annotation_ui.py -- "
                "--annotator annotator-01`",
                store.root,
            )
            summary["status"] = "skipped: no annotations"
            _EXIT_CODE.append(0)
            return

        reviews = ReviewLog(path=gold_dir / "reviews.jsonl").read()
        log.info("%d annotations, %d reviews", len(annotations), len(reviews))

        split_assignment = load_split_manifest(manifest_dir)
        reservation = load_reservation(manifest_dir, split_assignment=split_assignment)
        if reservation is not None:
            outside = sorted({a.case_id for a in annotations} - reservation.as_set)
            if outside:
                log.warning(
                    "%d annotated cases are outside the Gold reservation and are not "
                    "held out from training; first few: %s",
                    len(outside),
                    outside[:5],
                )

        case_ids = {a.case_id for a in annotations}
        facts_by_case, bronze_by_case = _load_supporting(
            processed_dir / "facts", corpus_dir / "bronze.jsonl", case_ids
        )
        log.info(
            "loaded %d fact records and %d Bronze alignments for %d cases",
            len(facts_by_case),
            len(bronze_by_case),
            len(case_ids),
        )

        vocabulary = load_vocabulary()
        counter = get_token_counter(str(cfg.corpus.ingestion.tokenizer))
        report = ingest_annotations(
            annotations,
            reviews,
            facts_by_case,
            bronze_by_case,
            split_assignment=split_assignment,
            case_store=processed_dir / "cases",
            repo_root=REPO_ROOT,
            vocabulary=vocabulary,
            token_counter=counter,
        )
        print(report.summary())

        mentioned_by = {
            (item.record.case_id, item.annotation.annotator_id): tuple(
                item.record.salience.get("mentioned", ())
            )
            for item in report.items
        }
        agreement = measure_agreement(
            annotations,
            double_annotation_rate=float(cfg.corpus.agreement.double_annotation_rate),
            mentioned_by=mentioned_by,
        )
        print()
        print(agreement.summary())

        payloads: list[dict[str, Any]] = report.payloads()
        gate = validate_corpus(
            payloads,
            repo_root=REPO_ROOT,
            split_manifest=split_assignment,
            vocabulary=vocabulary,
            token_counter=counter,
            dedup_threshold=float(cfg.corpus.ingestion.validation.dedup_jaccard),
        )
        print()
        print(gate.summary())

        corpus_dir.mkdir(parents=True, exist_ok=True)
        write_report(gate, corpus_dir / "gold_validation.json")
        write_json(corpus_dir / "gold_quality.json", report.to_dict(), canonical=True)
        write_json(corpus_dir / "gold_agreement.json", agreement.to_dict(), canonical=True)
        write_json(run_dir / "gold_quality.json", report.to_dict(), canonical=True)
        write_json(run_dir / "gold_agreement.json", agreement.to_dict(), canonical=True)

        if not gate.gate_passed and bool(cfg.corpus.ingestion.validation.fail_on_gate):
            log.error(
                "ten-point gate FAILED: %d of %d Gold records. A human-authored record "
                "that fails the harness is a record to revise; do not weaken the gate.",
                gate.failed,
                gate.total,
            )
            summary["status"] = "gate failed"
            _EXIT_CODE.append(1)
            return

        write_jsonl(corpus_dir / "gold.jsonl", payloads, canonical=True)

        RunContext.capture(
            experiment_name=cfg.experiment.name,
            cfg=cfg,
            seeds={"global": int(cfg.seed)},
            repo_root=REPO_ROOT,
            phase="6",
        ).save(run_dir)

        summary["n_ingested"] = report.n_ingested
        summary["n_held"] = len(report.held)
        summary["kappa"] = round(agreement.kappa, 4)
        summary["gate_passed"] = gate.gate_passed
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
