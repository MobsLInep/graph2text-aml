#!/usr/bin/env python
"""Build the Bronze corpus: load facts, render, validate, dedupe, write.

Writes four things:

- ``corpus/bronze.jsonl`` — one training record per case, schema-validated.
- ``corpus/bronze_validation.json`` — the ten-point harness report.
- ``corpus/bronze_diversity.json`` — distinct-n, self-BLEU, per-family breakdowns.
- ``corpus/bronze_samples.md`` — one rendered narrative per family, plain and annotated,
  for a human to read. A corpus nobody has looked at is a corpus nobody has checked.

**Exits non-zero when the gate fails**, and the gate is not negotiable from the command
line: ``corpus.validation.fail_on_gate=false`` exists for inspecting a broken build, and
using it to ship one would be a decision, not a configuration.

Usage:
    uv run python scripts/04_build_bronze.py
    uv run python scripts/04_build_bronze.py corpus.limit=500
    uv run python scripts/04_build_bronze.py data=elliptic2
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from g2t_aml.corpus.bronze.renderer import RENDERER_VERSION, RenderError, render_bronze
from g2t_aml.corpus.claims import claims_from_slots
from g2t_aml.corpus.diversity import measure_diversity
from g2t_aml.corpus.factsio import load_case_facts_file
from g2t_aml.corpus.graphref import build_graph_ref
from g2t_aml.corpus.record import TrainingRecord
from g2t_aml.corpus.tokenization import get_token_counter, word_count
from g2t_aml.corpus.validate import load_split_manifest, validate_corpus
from g2t_aml.facts.checkers import CheckContext, check_claim, check_narrative_text, summarise
from g2t_aml.facts.salience import salience_report
from g2t_aml.facts.schema import CASE_FACTS_SCHEMA_VERSION, CaseFacts
from g2t_aml.facts.serialiser import serialise_facts
from g2t_aml.facts.vocab import load_vocabulary
from g2t_aml.utils.io import write_json, write_jsonl
from g2t_aml.utils.logging import configure_logging, get_logger, stage
from g2t_aml.utils.run_context import RunContext

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_record(
    facts: CaseFacts,
    split: str,
    tier: str,
    case_store: Path,
    vocabulary: Any,
    counter: Any,
    seed: int,
) -> tuple[TrainingRecord, str, int]:
    """Render one case and wrap it in a training record.

    The verification block is computed here from the narrative's own slots, parsed back
    out of the rendered text rather than read from the record — see
    :mod:`g2t_aml.corpus.claims` for why that direction is the whole point. The validation
    harness recomputes it independently and does not trust what is written here.

    Args:
        facts: The fact record.
        split: The split from the frozen manifest.
        tier: The corpus tier.
        case_store: Directory holding the case membership tables.
        vocabulary: The controlled vocabulary.
        counter: The token counter.
        seed: The run seed, recorded for provenance.

    Returns:
        ``(record, family, variant)``.

    Raises:
        RenderError: If the case cannot be rendered.
    """
    narrative = render_bronze(facts, vocabulary=vocabulary, seed=seed, token_counter=counter)
    context = CheckContext(facts=facts, vocabulary=vocabulary)
    results = [check_claim(c, context) for c in claims_from_slots(narrative.slots, narrative.text)]
    results += check_narrative_text(narrative.text, context)
    verdicts = summarise(results)

    salience = salience_report(facts, vocabulary)
    mentioned = [p for p in salience.required if p in narrative.slot_paths()]

    record = TrainingRecord(
        case_id=facts.case_id,
        dataset=facts.dataset,
        split=split,
        tier=tier,
        facts=facts,
        graph_ref=build_graph_ref(case_store, facts.case_id, REPO_ROOT),
        serialised_facts=serialise_facts(facts, style="compact"),
        target_narrative=narrative.text,
        target_slots=narrative.slots,
        generator={
            "method": "template",
            "family": narrative.family,
            "variant": narrative.variant,
            "renderer_version": narrative.renderer_version,
            "seed": seed,
        },
        verification={
            "supported": verdicts["by_verdict"]["supported"],
            "contradicted": verdicts["by_verdict"]["contradicted"],
            "unverifiable": verdicts["by_verdict"]["unverifiable"],
            "unverifiable_rate": round(verdicts["unverifiable_rate"], 6),
            "n_claims": verdicts["n_claims"],
            "critical_error_rate": round(verdicts["critical_error_rate"], 6),
            "by_hallucination_class": verdicts["by_hallucination_class"],
        },
        length={
            "n_tokens": counter.count(narrative.text),
            "n_words": word_count(narrative.text),
            "n_chars": len(narrative.text),
            "tokenizer": counter.name,
        },
        salience={
            "required": list(salience.required),
            "excused": list(salience.excused),
            "mentioned": mentioned,
            "coverage": round(len(mentioned) / len(salience.required), 6)
            if salience.required
            else 1.0,
        },
    )
    return record, narrative.family, narrative.variant


def _write_samples(path: Path, samples: dict[str, tuple[str, str]]) -> None:
    """Write one narrative per family, plain and annotated, for a human to read.

    Args:
        path: Destination.
        samples: Family to ``(plain text, annotated text)``.
    """
    lines = [
        "# Bronze corpus samples",
        "",
        "One rendered narrative per template family, plain and slot-annotated. Generated",
        "by `scripts/04_build_bronze.py`; do not edit by hand.",
        "",
    ]
    for family, (plain, annotated) in sorted(samples.items()):
        lines += [
            f"## {family}",
            "",
            "```",
            plain,
            "```",
            "",
            "<details><summary>annotated</summary>",
            "",
            "```",
            annotated,
            "```",
            "",
            "</details>",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


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
def _run(cfg: DictConfig) -> None:  # noqa: PLR0912, PLR0915 - one linear build; splitting
    # separate the stage() accounting from the work it accounts for.
    """Build, validate and write the Bronze corpus.

    Args:
        cfg: Composed Hydra configuration.

    Returns:
        Nothing. The exit code is recorded in `_EXIT_CODE`; 0 on success, 1 when the fact
        records are absent or the ten-point gate fails.
    """
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    configure_logging(log_file=run_dir / "bronze.log")
    log = get_logger(__name__)

    processed_dir = Path(cfg.paths.processed_dir) / str(cfg.data.interim_name)
    facts_dir = processed_dir / "facts"
    case_store = processed_dir / "cases"
    corpus_dir = processed_dir / "corpus"

    declared = str(cfg.schema_version.case_facts)
    if declared != CASE_FACTS_SCHEMA_VERSION:
        log.error(
            "config declares case_facts schema %s but the code is frozen at %s (invariant 3)",
            declared,
            CASE_FACTS_SCHEMA_VERSION,
        )
        _EXIT_CODE.append(1)
        return

    with stage("bronze", log, facts=str(facts_dir), out=str(corpus_dir)) as summary:
        if not facts_dir.is_dir() or not any(facts_dir.glob("*.json")):
            log.warning("no fact records at %s; run `make facts` first", facts_dir)
            summary["status"] = "skipped: no fact records"
            _EXIT_CODE.append(0)
            return

        manifest_dir = Path(cfg.data.split.manifest_dir)
        if not (manifest_dir / "train.txt").is_file():
            log.warning("no frozen split manifest at %s; run `make cases` first", manifest_dir)
            summary["status"] = "skipped: no split manifest"
            _EXIT_CODE.append(0)
            return
        split_manifest = load_split_manifest(manifest_dir)
        log.info("split manifest holds %d cases", len(split_manifest))

        vocabulary = load_vocabulary()
        counter = get_token_counter(str(cfg.corpus.tokenizer))
        seed = int(cfg.seed)
        min_nodes = int(cfg.corpus.min_nodes)
        limit = int(cfg.corpus.limit or 0)

        case_ids = sorted(split_manifest)
        if limit:
            case_ids = case_ids[:limit]

        records: list[TrainingRecord] = []
        payloads: list[dict[str, Any]] = []
        families: Counter[str] = Counter()
        variants: list[int] = []
        typologies: list[str] = []
        samples: dict[str, tuple[str, str]] = {}
        excluded: Counter[str] = Counter()

        for n, case_id in enumerate(case_ids, start=1):
            path = facts_dir / f"{case_id}.json"
            if not path.is_file():
                excluded["no_fact_record"] += 1
                continue
            facts = load_case_facts_file(path)
            if facts.structure.n_nodes < min_nodes:
                excluded["below_min_nodes"] += 1
                continue
            try:
                record, family, variant = _build_record(
                    facts,
                    split_manifest[case_id],
                    str(cfg.corpus.tier),
                    case_store,
                    vocabulary,
                    counter,
                    seed,
                )
            except RenderError as exc:
                log.error("case %s did not render: %s", case_id, exc)
                excluded["render_error"] += 1
                continue
            records.append(record)
            payloads.append(record.to_dict())
            families[family] += 1
            variants.append(variant)
            typologies.append(facts.typology.label)
            if family not in samples:
                narrative = render_bronze(
                    facts, vocabulary=vocabulary, seed=seed, token_counter=counter
                )
                samples[family] = (narrative.text, narrative.annotated)
            if n % 2000 == 0:
                log.info("  %d / %d cases, %d rendered", n, len(case_ids), len(records))

        log.info("rendered %d narratives across %d families", len(records), len(families))
        for reason, count in excluded.most_common():
            log.info("  excluded %d cases: %s", count, reason)

        log.info("running the ten-point validation harness")
        report = validate_corpus(
            payloads,
            repo_root=REPO_ROOT,
            split_manifest=split_manifest,
            vocabulary=vocabulary,
            token_counter=counter,
            dedup_threshold=float(cfg.corpus.validation.dedup_jaccard),
        )

        dropped = frozenset(report.duplicates.dropped) if report.duplicates else frozenset()
        kept = [(r, p) for r, p in zip(records, payloads, strict=True) if r.case_id not in dropped]
        if dropped:
            log.info("dropping %d near-duplicate narratives", len(dropped))
            records = [r for r, _ in kept]
            payloads = [p for _, p in kept]
            report = validate_corpus(
                payloads,
                repo_root=REPO_ROOT,
                split_manifest=split_manifest,
                vocabulary=vocabulary,
                token_counter=counter,
                dedup_threshold=float(cfg.corpus.validation.dedup_jaccard),
            )
            families = Counter(str(p["generator"]["family"]) for p in payloads)
            variants = [int(p["generator"]["variant"]) for p in payloads]
            typologies = [str(p["facts"]["typology"]["label"]) for p in payloads]

        corpus_dir.mkdir(parents=True, exist_ok=True)
        write_json(corpus_dir / "bronze_validation.json", report.to_dict(), canonical=True)
        write_json(run_dir / "bronze_validation.json", report.to_dict(), canonical=True)
        print(report.summary())

        diversity = measure_diversity(
            [r.target_narrative for r in records],
            typologies=typologies,
            families=[str(p["generator"]["family"]) for p in payloads],
            variants=variants,
            slot_spans=[
                [(int(s["span"][0]), int(s["span"][1])) for s in p["target_slots"]]
                for p in payloads
            ],
            seed=int(cfg.corpus.diversity.seed),
        )
        write_json(corpus_dir / "bronze_diversity.json", diversity.to_dict(), canonical=True)
        write_json(run_dir / "bronze_diversity.json", diversity.to_dict(), canonical=True)
        print(diversity.summary())

        warn_above = float(cfg.corpus.diversity.self_bleu_warn_above)
        if diversity.self_bleu > warn_above:
            log.warning(
                "self-BLEU %.4f exceeds %.2f: the template pack is collapsing. Add "
                "realisation variants before training on this corpus.",
                diversity.self_bleu,
                warn_above,
            )

        _write_samples(corpus_dir / "bronze_samples.md", samples)

        if not report.gate_passed and bool(cfg.corpus.validation.fail_on_gate):
            log.error(
                "ten-point gate FAILED: %d of %d records. Bronze failing the harness "
                "means a bug in the renderer or the fact layer; do not weaken the gate.",
                report.failed,
                report.total,
            )
            summary["status"] = "gate failed"
            _EXIT_CODE.append(1)
            return

        write_jsonl(corpus_dir / "bronze.jsonl", payloads, canonical=True)

        RunContext.capture(
            experiment_name=cfg.experiment.name,
            cfg=cfg,
            seeds={"global": seed},
            repo_root=REPO_ROOT,
            phase="4",
        ).save(run_dir)

        summary["n_records"] = len(records)
        summary["n_families"] = len(families)
        summary["gate_passed"] = report.gate_passed
        summary["self_bleu"] = round(diversity.self_bleu, 4)
        summary["renderer_version"] = RENDERER_VERSION
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
