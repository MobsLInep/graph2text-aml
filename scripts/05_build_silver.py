#!/usr/bin/env python
"""Build the Silver corpus: rewrite Bronze with two teachers, verify, repair, discard.

Writes seven things:

- ``corpus/silver.jsonl`` — one verified training record per surviving case.
- ``corpus/silver_discards.jsonl`` — **a deliverable, not a debug artifact.** Every case
  that could not be made to verify, with its teacher, typology, split, attempt count and
  per-class violation breakdown. This becomes a table in the paper.
- ``corpus/silver_validation.json`` — the same ten-point harness that gates Bronze.
- ``corpus/silver_cost.json`` — spend by teacher, cache hit rate, token totals.
- ``corpus/silver_quality.json`` — dedup and degeneracy filtering, with per-teacher drop
  asymmetry.
- ``corpus/silver_diversity.json`` — distinct-n, self-BLEU at fixed references, and the
  comparison against Bronze that says whether Silver is meaningfully distinct at all.
- ``corpus/silver_samples.md`` — narratives a human can read, Bronze beside Silver.

Exits non-zero when the ten-point gate fails or the discard rate exceeds its ceiling.

Usage:
    uv run python scripts/05_build_silver.py corpus=silver corpus.generation.dry_run=true
    uv run python scripts/05_build_silver.py corpus=silver
    uv run python scripts/05_build_silver.py corpus=silver corpus.generation.resume=false
"""

from __future__ import annotations

import random
import sys
import threading
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from g2t_aml.corpus.bronze.renderer import RENDERER_VERSION, RenderError, render_bronze
from g2t_aml.corpus.diversity import measure_diversity
from g2t_aml.corpus.factsio import load_case_facts_file
from g2t_aml.corpus.graphref import build_graph_ref
from g2t_aml.corpus.silver.api_client import (
    BudgetGuard,
    CheckpointStore,
    CostTracker,
    ErrorLog,
    ResponseCache,
    RetryPolicy,
    TeacherSpec,
    build_teacher,
    preflight,
    specs_from_config,
)
from g2t_aml.corpus.silver.generate import (
    CaseInput,
    SilverConfig,
    assign_teachers,
    discard_report,
    teacher_balance_report,
)
from g2t_aml.corpus.silver.quality import QualityConfig, filter_records
from g2t_aml.corpus.silver.run import run_generation
from g2t_aml.corpus.tokenization import get_token_counter
from g2t_aml.corpus.validate import load_split_manifest, validate_corpus
from g2t_aml.facts.schema import CASE_FACTS_SCHEMA_VERSION
from g2t_aml.facts.vocab import load_vocabulary
from g2t_aml.utils.io import write_json
from g2t_aml.utils.logging import configure_logging, get_logger, stage
from g2t_aml.utils.run_context import RunContext

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_cases(
    facts_dir: Path, split_manifest: dict[str, str], min_nodes: int, limit: int, vocabulary: Any
) -> tuple[list[CaseInput], dict[str, int]]:
    """Load fact records and re-render their Bronze narratives.

    Bronze is re-rendered rather than read back from ``bronze.jsonl``. Rendering is a pure
    function of the fact record — Phase 4 verified the corpus rebuilds byte-identically —
    and the rewrite prompt needs the *annotated* form, which the corpus file does not
    carry. Reading the 240 MB corpus to recover text that regenerates exactly would buy
    nothing; the renderer version is recorded on every Silver record so the pairing stays
    attributable.

    Args:
        facts_dir: Directory of fact records.
        split_manifest: Case id to split, from the frozen Phase 2 manifest.
        min_nodes: Exclude cases below this many accounts.
        limit: Cap on cases, 0 for all.
        vocabulary: The controlled vocabulary.

    Returns:
        ``(cases, exclusion counts)``.
    """
    log = get_logger(__name__)
    excluded: dict[str, int] = {}
    cases: list[CaseInput] = []
    case_ids = sorted(split_manifest)
    if limit:
        case_ids = case_ids[:limit]

    for n, case_id in enumerate(case_ids, start=1):
        path = facts_dir / f"{case_id}.json"
        if not path.is_file():
            excluded["no_fact_record"] = excluded.get("no_fact_record", 0) + 1
            continue
        facts = load_case_facts_file(path)
        if facts.structure.n_nodes < min_nodes:
            excluded["below_min_nodes"] = excluded.get("below_min_nodes", 0) + 1
            continue
        try:
            bronze = render_bronze(facts, vocabulary=vocabulary)
        except RenderError as exc:
            log.error("case %s did not render: %s", case_id, exc)
            excluded["render_error"] = excluded.get("render_error", 0) + 1
            continue
        cases.append(
            CaseInput(case_id=case_id, split=split_manifest[case_id], facts=facts, bronze=bronze)
        )
        if n % 2000 == 0:
            log.info("  loaded %d / %d cases", n, len(case_ids))
    return cases, excluded


def _build_teachers(
    cfg: DictConfig, tracker: CostTracker, cache: ResponseCache, errors: ErrorLog
) -> tuple[tuple[TeacherSpec, ...], dict[str, Any]]:
    """Construct every configured teacher.

    Args:
        cfg: Composed configuration.
        tracker: Cost tracker.
        cache: Response cache.
        errors: Error log.

    Returns:
        ``(specs, teacher key to client)``.

    Raises:
        ValueError: If fewer than two teachers, or fewer than two families, are
            configured. Refused rather than warned about: single-teacher Silver is the
            objection the tier exists to answer.
    """
    entries = [dict(OmegaConf.to_container(t, resolve=True)) for t in cfg.corpus.teachers]  # type: ignore[arg-type]
    specs = specs_from_config(entries)
    budget = BudgetGuard(float(cfg.corpus.budget.cap_usd), tracker)
    retry = RetryPolicy(
        max_attempts=int(cfg.corpus.budget.max_retries),
        base_delay_s=float(cfg.corpus.budget.retry_base_delay_s),
        max_delay_s=float(cfg.corpus.budget.retry_max_delay_s),
    )
    semaphore = threading.Semaphore(max(1, int(cfg.corpus.generation.concurrency)))
    teachers = {
        spec.key: build_teacher(
            spec,
            cache=cache,
            tracker=tracker,
            budget=budget,
            errors=errors,
            retry=retry,
            semaphore=semaphore,
        )
        for spec in specs
    }
    return specs, teachers


def _write_samples(path: Path, samples: list[tuple[str, str, str, str]]) -> None:
    """Write Bronze beside Silver for a human to read.

    A corpus nobody has looked at is a corpus nobody has checked, and for Silver the thing
    to look at is specifically the *pair*: whether the rewrite reads like an investigator
    wrote it, and whether it still says the same things.

    Args:
        path: Destination.
        samples: ``(case_id, teacher, bronze text, silver text)`` per sample.
    """
    lines = [
        "# Silver corpus samples",
        "",
        "Bronze beside its verified Silver rewrite. Generated by",
        "`scripts/05_build_silver.py`; do not edit by hand.",
        "",
    ]
    for case_id, teacher, bronze, silver in samples:
        lines += [
            f"## {case_id} — {teacher}",
            "",
            "**Bronze**",
            "",
            "```",
            bronze,
            "```",
            "",
            "**Silver**",
            "",
            "```",
            silver,
            "```",
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
def _run(cfg: DictConfig) -> None:  # noqa: PLR0911, PLR0912, PLR0915 - one linear build, in
    # the order the phase log reports it. Splitting it would separate each stage()
    # accounting from the work it accounts for, and each early return is a distinct,
    # separately-reported exit condition: missing inputs, a schema mismatch, a budget halt,
    # a gate failure, a discard rate over its ceiling.
    """Generate, verify, filter, gate and write the Silver corpus.

    Args:
        cfg: Composed Hydra configuration.

    Returns:
        Nothing. The exit code is recorded in `_EXIT_CODE`; 0 on success, 1 when inputs are
        missing, the ten-point gate fails, or the discard rate exceeds its configured
        ceiling.
    """
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    configure_logging(log_file=run_dir / "silver.log")
    log = get_logger(__name__)

    if str(cfg.corpus.tier) != "silver":
        log.error("this script builds the silver tier; got corpus=%s", cfg.corpus.tier)
        _EXIT_CODE.append(1)
        return

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

    with stage("silver", log, facts=str(facts_dir), out=str(corpus_dir)) as summary:
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
        vocabulary = load_vocabulary()
        counter = get_token_counter(str(cfg.corpus.tokenizer))
        dry_run = bool(cfg.corpus.generation.dry_run)

        log.info("loading fact records and rendering Bronze references")
        cases, excluded = _load_cases(
            facts_dir,
            split_manifest,
            int(cfg.corpus.min_nodes),
            int(cfg.corpus.limit or 0),
            vocabulary,
        )
        for reason, count in sorted(excluded.items()):
            log.info("  excluded %d cases: %s", count, reason)
        if not cases:
            log.error("no eligible cases")
            summary["status"] = "no eligible cases"
            _EXIT_CODE.append(1)
            return

        # Teacher assignment is over the WHOLE eligible corpus, before any truncation for
        # a dry run. Assigning over a 20-case subset would give those 20 a different
        # teacher than the real run does, so a dry run would not be a preview of it.
        specs_preview = specs_from_config(
            [dict(OmegaConf.to_container(t, resolve=True)) for t in cfg.corpus.teachers]  # type: ignore[arg-type]
        )
        assignment = assign_teachers(cases, specs_preview)
        log.info("assigned %d cases across %d teachers", len(assignment), len(specs_preview))

        if dry_run:
            n = int(cfg.corpus.generation.dry_run_records)
            rng = random.Random(int(cfg.seed))
            cases = rng.sample(cases, min(n, len(cases)))
            log.warning(
                "DRY RUN: %d records, projected cost only, nothing written to the corpus",
                len(cases),
            )

        # Everything that would stop the run, reported now rather than discovered on the
        # first call inside a worker pool several minutes into the job.
        if problems := preflight(specs_preview):
            for problem in problems:
                log.error("preflight: %s", problem)
            log.error(
                "refusing to start. A run that fails on its first call has already spent "
                "the time it took to load and render %d cases.",
                len(cases),
            )
            summary["status"] = "preflight failed"
            _EXIT_CODE.append(1)
            return

        tracker = CostTracker()
        cache = ResponseCache(
            Path(str(cfg.corpus.budget.cache_dir)),
            enabled=bool(cfg.corpus.budget.cache_enabled),
        )
        errors = ErrorLog(run_dir / "silver_errors.jsonl")
        specs, teachers = _build_teachers(cfg, tracker, cache, errors)

        silver_config = SilverConfig(
            max_unverifiable_rate=float(cfg.corpus.verification.max_unverifiable_rate),
            min_salience_coverage=float(cfg.corpus.verification.min_fact_recall),
            max_repair_attempts=int(cfg.corpus.generation.max_repair_attempts),
            min_words=int(cfg.corpus.generation.min_words),
            max_words=int(cfg.corpus.generation.max_words),
            tier=str(cfg.corpus.tier),
        )

        records_path = None if dry_run else corpus_dir / "silver.jsonl"
        discards_path = None if dry_run else corpus_dir / "silver_discards.jsonl"
        checkpoint = (
            CheckpointStore(corpus_dir / "silver_checkpoint.txt")
            if bool(cfg.corpus.generation.resume) and not dry_run
            else None
        )
        if checkpoint is None and records_path is not None and records_path.exists():
            # resume=false means rebuild; the previous stream must not be appended to.
            records_path.unlink()
            if discards_path is not None:
                discards_path.unlink(missing_ok=True)

        log.info("generating with teachers: %s", ", ".join(sorted(teachers)))
        result = run_generation(
            cases,
            assignment,
            teachers,
            vocabulary=vocabulary,
            config=silver_config,
            token_counter=counter,
            graph_ref_for=lambda case_id: build_graph_ref(case_store, case_id, REPO_ROOT),
            tracker=tracker,
            checkpoint=checkpoint,
            records_path=records_path,
            discards_path=discards_path,
            concurrency=int(cfg.corpus.generation.concurrency),
        )

        payloads = result.records
        log.info(
            "generated %d records, discarded %d, spent $%.2f",
            len(payloads),
            len(result.discards),
            tracker.total_usd,
        )

        # --- quality filtering ------------------------------------------------
        bronze_texts = {c.case_id: c.bronze.text for c in cases}
        kept, quality = filter_records(
            payloads,
            bronze_texts,
            config=QualityConfig(
                dedup_jaccard=float(cfg.corpus.quality.dedup_jaccard),
                bronze_verbatim_jaccard=float(cfg.corpus.quality.bronze_verbatim_jaccard),
                min_sections=int(cfg.corpus.quality.min_sections),
                min_section_words=int(cfg.corpus.quality.min_section_words),
                max_ngram_repeats=int(cfg.corpus.quality.max_ngram_repeats),
                min_type_token_ratio=float(cfg.corpus.quality.min_type_token_ratio),
            ),
        )
        print(quality.summary())

        # --- the ten-point harness, unchanged from Bronze ---------------------
        log.info("running the ten-point validation harness")
        report = validate_corpus(
            kept,
            repo_root=REPO_ROOT,
            split_manifest=split_manifest,
            vocabulary=vocabulary,
            token_counter=counter,
            dedup_threshold=float(cfg.corpus.quality.dedup_jaccard),
        )
        print(report.summary())

        sample_size = int(cfg.corpus.verification.gate_sample_size)
        sample_report = None
        if len(kept) > sample_size:
            sample = random.Random(int(cfg.seed)).sample(kept, sample_size)
            sample_report = validate_corpus(
                sample,
                repo_root=REPO_ROOT,
                split_manifest=split_manifest,
                vocabulary=vocabulary,
                token_counter=counter,
                dedup_threshold=float(cfg.corpus.quality.dedup_jaccard),
            )
            log.info(
                "%d-record random sample: %d/%d passed",
                sample_size,
                sample_report.passed,
                sample_report.total,
            )

        # --- reports ----------------------------------------------------------
        n_attempted = result.n_attempted
        discards = discard_report(result.discards, n_attempted)
        balance = teacher_balance_report(assignment, cases, kept={str(r["case_id"]) for r in kept})
        cost = {
            **tracker.to_dict(),
            "cache": cache.stats(),
            "errors": errors.summary(),
            "budget_cap_usd": float(cfg.corpus.budget.cap_usd),
            "halted_on_budget": result.halted,
            "halt_reason": result.halt_reason,
            "repair_attempts_on_accepted": result.attempt_histogram(),
        }

        silver_texts = [str(r["target_narrative"]) for r in kept]
        diversity = measure_diversity(
            silver_texts,
            typologies=[str(r["facts"]["typology"]["label"]) for r in kept],
            families=[str(r["generator"].get("teacher", "-")) for r in kept],
            variants=[int(r["generator"].get("repair_attempts", 0)) for r in kept],
            slot_spans=[
                [(int(s["span"][0]), int(s["span"][1])) for s in r["target_slots"]] for r in kept
            ],
            seed=int(cfg.corpus.diversity.seed),
        )
        print(diversity.summary())

        target = run_dir if dry_run else corpus_dir
        target.mkdir(parents=True, exist_ok=True)
        write_json(target / "silver_validation.json", report.to_dict(), canonical=True)
        if sample_report is not None:
            write_json(
                target / "silver_validation_sample.json", sample_report.to_dict(), canonical=True
            )
        write_json(target / "silver_discard_report.json", discards, canonical=True)
        write_json(target / "silver_teacher_balance.json", balance, canonical=True)
        write_json(target / "silver_cost.json", cost, canonical=True)
        write_json(target / "silver_quality.json", quality.to_dict(), canonical=True)
        write_json(target / "silver_diversity.json", diversity.to_dict(), canonical=True)

        by_case = {c.case_id: c for c in cases}
        samples = [
            (
                str(r["case_id"]),
                str(r["generator"].get("teacher", "-")),
                by_case[str(r["case_id"])].bronze.text,
                str(r["target_narrative"]),
            )
            for r in kept[:6]
            if str(r["case_id"]) in by_case
        ]
        _write_samples(target / "silver_samples.md", samples)

        RunContext.capture(
            experiment_name=cfg.experiment.name,
            cfg=cfg,
            seeds={"global": int(cfg.seed)},
            repo_root=REPO_ROOT,
            phase="5",
        ).save(run_dir)

        summary["n_records"] = len(kept)
        summary["n_discarded"] = len(result.discards)
        summary["discard_rate"] = discards["discard_rate"]
        summary["cost_usd"] = round(tracker.total_usd, 2)
        summary["renderer_version"] = RENDERER_VERSION
        summary["dry_run"] = dry_run

        if dry_run:
            projected = _project(cost, discards, len(kept), n_attempted, len(assignment))
            write_json(run_dir / "silver_projection.json", projected, canonical=True)
            print(_format_projection(projected))
            summary["status"] = "dry run"
            _EXIT_CODE.append(0)
            return

        if result.halted:
            log.error(
                "run halted on the budget cap; the corpus is incomplete: %s", result.halt_reason
            )
            summary["status"] = "halted on budget"
            _EXIT_CODE.append(1)
            return

        if not report.gate_passed and bool(cfg.corpus.verification.fail_on_gate):
            log.error(
                "ten-point gate FAILED: %d of %d records. Do not weaken the gate.",
                report.failed,
                report.total,
            )
            summary["status"] = "gate failed"
            _EXIT_CODE.append(1)
            return

        ceiling = float(cfg.corpus.verification.max_discard_rate)
        if discards["verification_discard_rate"] > ceiling:
            log.error(
                "verification discard rate %.4f exceeds the %.2f ceiling. This is a "
                "finding, not a threshold to raise: report it and investigate the "
                "per-class breakdown in silver_discard_report.json.",
                discards["verification_discard_rate"],
                ceiling,
            )
            summary["status"] = "discard rate exceeded"
            _EXIT_CODE.append(1)
            return

        summary["status"] = "ok"

    _EXIT_CODE.append(0)
    return


def _project(
    cost: dict[str, Any],
    discards: dict[str, Any],
    n_kept: int,
    n_attempted: int,
    n_eligible: int,
) -> dict[str, Any]:
    """Extrapolate a dry run to the full corpus.

    Args:
        cost: The cost report.
        discards: The discard report.
        n_kept: Records kept in the dry run.
        n_attempted: Cases attempted in the dry run.
        n_eligible: Cases the full run would attempt.

    Returns:
        Projected spend and yield, with the sample size stated beside them — a projection
        from twenty cases is a projection from twenty cases, and reporting it without its
        denominator invites it to be quoted as a measurement.
    """
    spent = float(cost.get("total_usd", 0.0))
    per_case = spent / n_attempted if n_attempted else 0.0
    keep_rate = n_kept / n_attempted if n_attempted else 0.0
    return {
        "sample_size": n_attempted,
        "eligible_cases": n_eligible,
        "observed_cost_usd": round(spent, 4),
        "cost_per_case_usd": round(per_case, 6),
        "projected_full_run_usd": round(per_case * n_eligible, 2),
        "observed_keep_rate": round(keep_rate, 4),
        "projected_records": int(keep_rate * n_eligible),
        "observed_discard_rate": discards.get("discard_rate", 0.0),
        "repair_attempts_on_accepted": cost.get("repair_attempts_on_accepted", {}),
        "caveat": (
            "Extrapolated linearly from a small sample. Cost per case varies with fact "
            "record size and repair rate, and the sample is drawn at random rather than "
            "stratified by typology, so a rare typology may be absent from it entirely."
        ),
    }


def _format_projection(projection: dict[str, Any]) -> str:
    """Render the dry-run projection for a terminal.

    Args:
        projection: From :func:`_project`.

    Returns:
        A short block of text.
    """
    return "\n".join(
        [
            "",
            "dry run projection",
            f"  sample                 {projection['sample_size']} cases",
            f"  observed cost          ${projection['observed_cost_usd']:.4f}",
            f"  cost per case          ${projection['cost_per_case_usd']:.6f}",
            f"  eligible cases         {projection['eligible_cases']:,}",
            f"  PROJECTED FULL RUN     ${projection['projected_full_run_usd']:,.2f}",
            f"  projected records      {projection['projected_records']:,}",
            f"  observed keep rate     {projection['observed_keep_rate']:.3f}",
            "",
            f"  {projection['caveat']}",
        ]
    )


def main() -> int:
    """Run the Hydra entrypoint and return its exit code.

    Returns:
        The code ``_run`` produced, or 0 when it produced none.
    """
    _run()
    return _EXIT_CODE[-1] if _EXIT_CODE else 0


if __name__ == "__main__":
    sys.exit(main())
