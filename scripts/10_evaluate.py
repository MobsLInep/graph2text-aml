#!/usr/bin/env python
"""Score one or many systems and write the evaluation report.

Reads the fact records, the Bronze corpus (for Method A's alignment reference and as the
template baseline arm), the Gold corpus (for Layer 1 references), and whatever generation
files the config names. Writes to ``artifacts/metrics/{run_id}/``:

- ``evaluation.json`` — the machine-readable report.
- ``evaluation.md`` — the human summary, Layer 2 first.
- ``tables_<stream>.tex`` — the paper's faithfulness and taxonomy tables.
- ``errors.jsonl`` — every classified error, for the qualitative section and for the
  hand-labelling sample.

**Runs on Bronze alone with no arguments**, which is the CI gate: Bronze scored against
its own fact records must come out at 100% supported and 100% Zero-Hallucination, and any
drift in the extractor or the checker shows up there before it reaches a real system.

Usage:
    uv run python scripts/10_evaluate.py
    uv run python scripts/10_evaluate.py eval.systems.s1=artifacts/runs/.../generations.jsonl
    uv run python scripts/10_evaluate.py eval.surface.bertscore=false eval.limit=200
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from g2t_aml.corpus.factsio import facts_from_dict
from g2t_aml.corpus.record import BronzeNarrative, SlotAnnotation
from g2t_aml.eval.report import evaluate
from g2t_aml.eval.types import SystemOutput, load_system_outputs
from g2t_aml.facts.schema import CASE_FACTS_SCHEMA_VERSION, CaseFacts
from g2t_aml.facts.vocab import load_vocabulary
from g2t_aml.utils.io import read_jsonl
from g2t_aml.utils.logging import configure_logging, get_logger, stage
from g2t_aml.utils.run_context import RunContext

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
REPO_ROOT = Path(__file__).resolve().parents[1]

#: See D-051 and the note in `04_build_bronze.py`: `@hydra.main` discards its wrapped
#: function's return value, so the exit code is carried out of a module-level cell.
_EXIT_CODE: list[int] = []


def _load_corpus(
    path: Path, *, limit: int = 0
) -> tuple[dict[str, str], dict[str, BronzeNarrative], dict[str, CaseFacts]]:
    """Read a corpus JSONL into narratives, slot alignments and embedded fact records.

    The fact record embedded on a training record is the record the narrative was
    *written from*, which is the one faithfulness must be scored against. Re-deriving it
    from the case store would be a second extraction, and a disagreement between the two
    would show up as a hallucination in every narrative rather than as the extractor bug
    it would be.

    Args:
        path: The corpus file.
        limit: Stop after this many records, or 0 for all.

    Returns:
        ``(narratives, bronze index, facts)``, each keyed by case id.
    """
    narratives: dict[str, str] = {}
    alignments: dict[str, BronzeNarrative] = {}
    facts: dict[str, CaseFacts] = {}
    for row in read_jsonl(path):
        if not isinstance(row, dict):
            continue
        case_id = str(row["case_id"])
        narratives[case_id] = str(row["target_narrative"])
        slots = tuple(
            SlotAnnotation.from_dict(s)
            for s in row.get("target_slots") or ()
            if isinstance(s, dict)
        )
        generator = row.get("generator") or {}
        alignments[case_id] = BronzeNarrative(
            case_id=case_id,
            text=narratives[case_id],
            annotated="",
            slots=slots,
            family=str(generator.get("family", "unknown")),
            variant=int(generator.get("variant", 0)),
        )
        facts[case_id] = _facts_from_row(row)
        if limit and len(narratives) >= limit:
            break
    return narratives, alignments, facts


def _facts_from_row(row: dict[str, Any]) -> CaseFacts:
    """Rebuild the fact record embedded in a training record.

    Args:
        row: A parsed training record.

    Returns:
        The fact record.
    """
    return facts_from_dict(dict(row["facts"]))


def _named_systems(cfg: DictConfig) -> dict[str, str]:
    """Read the system-name-to-generation-file mapping out of the config.

    Args:
        cfg: The composed config.

    Returns:
        System name to path. Empty when none are configured, which is the Bronze-only
        CI gate rather than an error.
    """
    block = OmegaConf.select(cfg, "eval.systems")
    if block is None:
        return {}
    resolved = OmegaConf.to_container(block, resolve=True)
    if not isinstance(resolved, dict):
        return {}
    return {str(k): str(v) for k, v in resolved.items() if v}


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="config")
def _run(cfg: DictConfig) -> None:  # noqa: PLR0915 -- one linear pipeline
    """Score the configured systems and write the report.

    Args:
        cfg: Composed Hydra configuration.

    Returns:
        Nothing. The exit code is recorded in `_EXIT_CODE`; 0 on success, 1 when the
        inputs are missing or the Bronze self-consistency gate fails.
    """
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    configure_logging(log_file=run_dir / "evaluate.log")
    log = get_logger(__name__)

    declared = str(cfg.schema_version.case_facts)
    if declared != CASE_FACTS_SCHEMA_VERSION:
        log.error(
            "config declares case_facts schema %s but the code is frozen at %s (invariant 3)",
            declared,
            CASE_FACTS_SCHEMA_VERSION,
        )
        _EXIT_CODE.append(1)
        return

    processed = Path(cfg.paths.processed_dir) / str(cfg.data.interim_name)
    corpus_dir = processed / "corpus"
    bronze_path = corpus_dir / "bronze.jsonl"
    gold_path = corpus_dir / "gold.jsonl"
    run_id = run_dir.name
    out_dir = Path(cfg.paths.metrics_dir) / "eval" / run_id

    with stage("evaluate", log, corpus=str(corpus_dir), out=str(out_dir)) as summary:
        if not bronze_path.is_file():
            log.warning("no Bronze corpus at %s; run `make bronze` first", bronze_path)
            summary["status"] = "skipped: no Bronze corpus"
            _EXIT_CODE.append(0)
            return

        limit = int(OmegaConf.select(cfg, "eval.limit") or 0)
        bronze_texts, bronze_index, facts = _load_corpus(bronze_path, limit=limit)
        log.info("loaded %d Bronze records", len(bronze_texts))

        references: dict[str, str] = {}
        if gold_path.is_file():
            gold_texts, _, gold_facts = _load_corpus(gold_path)
            references.update(gold_texts)
            # A Gold record carries its own fact record; prefer it, because a Gold
            # narrative was written against that one.
            facts.update(gold_facts)
            log.info("loaded %d Gold references", len(references))
        else:
            log.warning(
                "no Gold corpus at %s — Layer 1 will report every overlap metric as "
                "unavailable, and Layer 2 is unaffected",
                gold_path,
            )

        outputs: list[SystemOutput] = [
            SystemOutput(
                system="bronze",
                case_id=case_id,
                narrative=text,
                split=None,
                stream="balanced",
                slots=bronze_index[case_id].slots,
            )
            for case_id, text in bronze_texts.items()
        ]

        for name, path in sorted(_named_systems(cfg).items()):
            source = Path(path)
            if not source.is_file():
                log.error("system %r names %s, which does not exist", name, source)
                _EXIT_CODE.append(1)
                return
            loaded = load_system_outputs(source, system=name, limit=limit or None)
            outputs.extend(loaded)
            log.info("loaded %d narratives for system %r", len(loaded), name)

        vocabulary = load_vocabulary()
        bertscore_model = (
            str(cfg.eval.surface.bertscore_model)
            if bool(OmegaConf.select(cfg, "eval.surface.bertscore"))
            else None
        )
        report = evaluate(
            outputs,
            references,
            facts,
            bronze=bronze_index,
            run_id=run_id,
            vocabulary=vocabulary,
            bertscore_model=bertscore_model,
            metadata={
                "dataset": str(cfg.data.interim_name),
                "schema_version": CASE_FACTS_SCHEMA_VERSION,
                "n_systems": len({o.system for o in outputs}),
                "n_narratives": len(outputs),
            },
            n_resamples=int(OmegaConf.select(cfg, "eval.stats.bootstrap_samples") or 10_000),
            seed=int(cfg.seed),
        )

        written = report.write_all(out_dir)
        for name, path in sorted(written.items()):
            log.info("wrote %s -> %s", name, path)

        RunContext.capture(
            experiment_name=cfg.experiment.name,
            cfg=cfg,
            seeds={"global": int(cfg.seed)},
            repo_root=REPO_ROOT,
            phase="10",
        ).save(run_dir)

        # The gate. Bronze is faithful by construction: it renders from the fact record
        # and every formatter ships with its inverse, so scoring it against its own
        # records must come out perfect. Anything less is a bug in the extractor, the
        # checker or the renderer -- never a property of Bronze -- and it has to fail the
        # run rather than appear as a slightly-imperfect baseline row in a table.
        bronze_report = report.systems.get(("bronze", "balanced"))
        if bronze_report is None:
            log.error("Bronze was not scored; the gate cannot run")
            summary["status"] = "gate failed: no Bronze"
            _EXIT_CODE.append(1)
            return

        zero_hallucination = bronze_report.faithfulness.zero_hallucination_rate
        summary["bronze_zero_hallucination"] = round(zero_hallucination, 6)
        summary["bronze_fact_precision"] = round(bronze_report.faithfulness.fact_precision, 6)
        summary["n_systems"] = len({o.system for o in outputs})

        if report.template_finding is not None:
            log.info("template baseline: %s", report.template_finding.headline)
            summary["template_non_discriminative"] = report.template_finding.non_discriminative

        if zero_hallucination < 1.0 and bool(
            OmegaConf.select(cfg, "eval.fail_on_bronze_gate") is not False
        ):
            log.error(
                "Bronze self-consistency gate FAILED: Zero-Hallucination %.4f, expected "
                "1.0. Bronze is faithful by construction, so this is a bug in the "
                "extractor or the checker.",
                zero_hallucination,
            )
            summary["status"] = "gate failed"
            _EXIT_CODE.append(1)
            return

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
