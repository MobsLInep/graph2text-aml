#!/usr/bin/env python
"""Score every case with the trained encoder and write ``model_signal`` into its fact record.

This is the write-back path Phase 3 designed for: ``CaseFacts.with_model_signal`` returns a
new record rather than mutating one another phase may already have hashed, and the schema
has carried the null-valued ``model_signal`` block since 1.0.0. Nothing about the frozen
schema changes here — the block is *populated*, not added.

Three things it writes per case: ``gnn_risk_score`` (the sigmoid risk probability),
``score_percentile`` (rank within the population scored in this run, so the number is
interpretable without knowing the score distribution), and ``top_contributing_nodes``
(accounts by pooling-attention mass, which is the attribution an investigator would be
shown).

**What this does and does not invalidate.** ``model_signal`` is additive and no Bronze
template reads it — the renderer never touches the block, so not one of the 15,707 Bronze
narratives changes and the corpus does **not** need regenerating. But
``facts.serialiser._compact`` *does* emit ``gnn_risk_score``, and that string is stored as
``serialised_facts`` on every training record. Regenerating Bronze after this run would
therefore push the encoder's own risk score into the **serialisation baseline** — the
"flatten the facts, no graph encoder" arm — and quietly stop it being a no-encoder control.
So Bronze is deliberately *not* regenerated, and
``tests/unit/test_encoder_writeback.py::test_bronze_serialised_facts_carry_no_model_signal``
pins that. See DECISIONS.md D-063.

Usage:
    uv run python scripts/07b_score_cases.py
    uv run python scripts/07b_score_cases.py \
        encoder.checkpoint=artifacts/checkpoints/encoder/gatv2/gatv2_seed42.pt
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import polars as pl
import torch
from omegaconf import DictConfig, OmegaConf

from g2t_aml.corpus.factsio import load_case_facts_file
from g2t_aml.facts.schema import (
    CASE_FACTS_SCHEMA_VERSION,
    EXTRACTOR_VERSION,
    ModelSignal,
    facts_to_dict,
    validate_facts,
)
from g2t_aml.models.encoder.attention_viz import extract_attention
from g2t_aml.models.encoder.dataset import (
    ALL_SPLITS,
    load_feature_space,
    load_split,
)
from g2t_aml.models.encoder.features import FeatureSpace
from g2t_aml.models.encoder.registry import build_encoder
from g2t_aml.utils.io import write_json
from g2t_aml.utils.logging import configure_logging, get_logger, stage
from g2t_aml.utils.run_context import RunContext

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
REPO_ROOT = Path(__file__).resolve().parents[1]

#: How many accounts are recorded as contributors per case.
TOP_CONTRIBUTORS = 5

_EXIT_CODE: list[int] = []


def _load_checkpoint(path: Path, log: Any) -> tuple[Any, FeatureSpace, str]:
    """Rebuild a trained arm from its checkpoint and the feature space stored beside it."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    space = FeatureSpace.from_dict(payload["feature_space"])
    encoder_cfg = OmegaConf.create(payload["encoder_config"])
    model = build_encoder(encoder_cfg, space)
    model.load_state_dict(payload["state_dict"])
    version = f"{payload['arm']}-seed{payload['seed']}-epoch{payload['best_epoch']}"
    log.info(
        "loaded %s (val AUC-PR %.4f) from %s",
        version,
        payload.get("best_val_auc_pr", float("nan")),
        path,
    )
    return model, space, version


def _default_checkpoint(cfg: DictConfig) -> Path:
    """Return the primary arm's first-seed checkpoint."""
    arm = str(cfg.encoder.arch)
    seed = int(cfg.training.seeds[0])
    return Path(cfg.paths.checkpoints_dir) / "encoder" / arm / f"{arm}_seed{seed}.pt"


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="config")
def _run(cfg: DictConfig) -> None:  # noqa: PLR0915 -- one linear score-and-write pass.
    """Score every cached case and populate ``model_signal`` in its fact record.

    Args:
        cfg: Composed Hydra configuration.

    Returns:
        Nothing. The exit code is recorded in ``_EXIT_CODE``; 1 when the checkpoint or the
        feature cache is absent, or a rewritten record fails schema validation.
    """
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    configure_logging(log_file=run_dir / "score_cases.log")
    log = get_logger(__name__)

    processed = Path(cfg.paths.processed_dir) / str(cfg.data.interim_name)
    cache_dir = processed / "encoder" / "features"
    facts_dir = processed / "facts"

    checkpoint = Path(str(cfg.encoder.get("checkpoint") or _default_checkpoint(cfg)))
    if not checkpoint.is_file():
        log.error("no encoder checkpoint at %s; run `make train-encoder` first", checkpoint)
        _EXIT_CODE.append(1)
        return
    if not facts_dir.is_dir():
        log.error("no fact records at %s; run `make facts` first", facts_dir)
        _EXIT_CODE.append(1)
        return

    with stage("score-cases", log, checkpoint=str(checkpoint), facts=str(facts_dir)) as summary:
        device = torch.device(
            "cuda"
            if str(cfg.experiment.device).startswith("cuda") and torch.cuda.is_available()
            else "cpu"
        )
        model, space, model_version = _load_checkpoint(checkpoint, log)
        model = model.to(device)
        cached_space = load_feature_space(cache_dir)
        if cached_space.to_dict() != space.to_dict():
            log.error(
                "the checkpoint's feature space differs from the cache's; the model would "
                "be applied to differently-encoded cases. Retrain or rebuild the cache."
            )
            _EXIT_CODE.append(1)
            return

        # Every split the cache holds, so a fact record exists for a case in any of them.
        # The realistic stream has no fact records and is skipped when its cases are not
        # on disk, rather than being an error.
        graphs = []
        for split in ALL_SPLITS:
            try:
                graphs.extend(load_split(cache_dir, split))
            except Exception as exc:
                log.warning("split %s not in the cache: %s", split, exc)
        log.info("scoring %d cases on %s", len(graphs), device)

        attentions = extract_attention(model, graphs, device, batch_size=64)

        # The percentile is computed over exactly the cases that receive one, and no
        # others. Ranking against the whole scored population would put a percentile in a
        # fact record whose reference set includes 10,000 realistic-stream cases that have
        # no fact record and are drawn at a completely different prevalence — a number an
        # investigator could not interpret and a checker could not verify. The population
        # size is recorded in the report so the percentile is never read without it.
        seen: set[str] = set()
        scorable: list[Any] = []
        missing = 0
        skipped_duplicates = 0
        for attention in attentions:
            if attention.case_id in seen:
                skipped_duplicates += 1
                continue
            seen.add(attention.case_id)
            if not (facts_dir / f"{attention.case_id}.json").is_file():
                missing += 1
                continue
            scorable.append(attention)

        scores = np.asarray([a.risk_score for a in scorable])
        ranks = scores.argsort().argsort()
        percentiles = 100.0 * ranks / max(len(scores) - 1, 1)

        written = 0
        rows: list[dict[str, Any]] = []

        for attention, percentile in zip(scorable, percentiles, strict=True):
            case_id = attention.case_id
            path = facts_dir / f"{case_id}.json"
            facts = load_case_facts_file(path)
            signal = ModelSignal(
                gnn_risk_score=round(float(attention.risk_score), 6),
                score_percentile=round(float(percentile), 4),
                top_contributing_nodes=tuple(
                    (node, round(float(weight), 6))
                    for node, weight in attention.top_nodes(TOP_CONTRIBUTORS)
                ),
                model_version=model_version,
            )
            payload = facts_to_dict(facts.with_model_signal(signal))
            validate_facts(payload)
            write_json(path, payload, canonical=True)
            written += 1
            rows.append(
                {
                    "case_id": case_id,
                    "gnn_risk_score": signal.gnn_risk_score,
                    "score_percentile": signal.score_percentile,
                    "model_version": model_version,
                }
            )
            if written % 5000 == 0:
                log.info("  %d records written", written)

        # Regenerate the aggregate Parquet's model_signal columns. The rest of the
        # aggregate is unchanged, so the existing frame is joined rather than rebuilt
        # from 30,000 JSON files — rebuilding it would take half an hour to produce
        # identical values in every other column.
        aggregate = processed / "facts.parquet"
        if aggregate.is_file() and rows:
            frame = pl.read_parquet(aggregate)
            update = pl.DataFrame(rows).drop("model_version")
            frame = frame.drop(
                [c for c in ("gnn_risk_score", "score_percentile") if c in frame.columns]
            ).join(update, on="case_id", how="left")
            frame = frame.with_columns(pl.lit(model_version).alias("model_signal_version"))
            frame.write_parquet(aggregate, compression="zstd")
            log.info("updated %s with the model_signal columns", aggregate)

        report = {
            "model_version": model_version,
            "checkpoint": str(checkpoint),
            "schema_version": CASE_FACTS_SCHEMA_VERSION,
            "extractor_version": EXTRACTOR_VERSION,
            "n_scored": len(seen),
            "n_written": written,
            "n_without_fact_record": missing,
            "n_duplicate_case_ids": skipped_duplicates,
            # What every `score_percentile` in a fact record ranks against. A percentile
            # without its reference population is unreadable, in the same way self-BLEU
            # without its reference count is (D-043).
            "percentile_population": len(scores),
            "percentile_population_note": (
                "score_percentile ranks a case against exactly the cases that received "
                "one -- the split-manifest population that has fact records -- and not "
                "against the realistic-imbalance stream, which has none."
            ),
            "score_quantiles": {
                str(q): float(np.quantile(scores, q))
                for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
            },
            # The regeneration question, answered explicitly rather than left implicit.
            "bronze_regeneration_required": False,
            "bronze_regeneration_note": (
                "model_signal is additive and no Bronze template reads it, so no narrative "
                "changes and the corpus is not regenerated. facts.serialiser._compact does "
                "emit gnn_risk_score into training_record.serialised_facts, so regenerating "
                "Bronze would contaminate the serialisation baseline -- the no-encoder "
                "ablation arm -- with the encoder's own output. See DECISIONS.md D-063."
            ),
        }
        write_json(run_dir / "score_cases.json", report)
        write_json(Path(cfg.paths.metrics_dir) / "encoder" / "score_cases.json", report)
        RunContext.capture(
            experiment_name=str(cfg.experiment.name),
            cfg=cfg,
            seeds={"global": int(cfg.seed)},
            repo_root=REPO_ROOT,
            phase="7b",
        ).save(run_dir)

        log.info("wrote model_signal into %d fact records (%d cases had none)", written, missing)
        summary["n_written"] = written
        summary["model_version"] = model_version
        summary["status"] = "ok"

    _EXIT_CODE.append(0)


def main() -> int:
    """Run the Hydra entrypoint and return its exit code.

    Returns:
        The code ``_run`` produced, or 0 when it produced none.
    """
    _run()
    return _EXIT_CODE[-1] if _EXIT_CODE else 0


if __name__ == "__main__":
    sys.exit(main())
