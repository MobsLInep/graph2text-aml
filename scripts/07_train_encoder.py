#!/usr/bin/env python
"""Train, evaluate and compare the Phase 7 encoder arms.

Runs every arm named in ``experiment.arms`` at every seed in ``training.seeds``, evaluates
each on the balanced test split and on the realistic-imbalance stream, runs the embedding
battery, and writes a comparison keyed on the two things the Phase 7 gate asks about:
GATv2 against the MLP control on AUC-PR with bootstrap intervals, and typology macro-F1
against chance.

The feature cache is built on first use and reused thereafter. It is verified against the
frozen split manifest's content hashes before training, so a cache built from a stale
manifest raises rather than quietly training on a different population.

Usage:
    uv run python scripts/07_train_encoder.py
    uv run python scripts/07_train_encoder.py experiment=encoder_debug
    uv run python scripts/07_train_encoder.py encoder=gin training.seeds=[42]
    uv run python scripts/07_train_encoder.py encoder.arch=mlp training.loss=weighted_bce
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from g2t_aml.models.encoder.analysis import (
    analyse_embeddings,
    save_analysis,
    umap_figure,
)
from g2t_aml.models.encoder.attention_viz import (
    extract_attention,
    laundering_path_nodes,
    path_attention_alignment,
    save_alignment,
)
from g2t_aml.models.encoder.dataset import (
    MANIFEST_SPLITS,
    REALISTIC_SPLIT,
    TYPOLOGY_CLASSES,
    build_feature_cache,
    load_feature_space,
    load_split,
    load_typologies,
    verify_cache_against_manifest,
)
from g2t_aml.models.encoder.metrics import (
    aggregate_over_seeds,
    bootstrap_auc_pr,
    paired_difference,
)
from g2t_aml.models.encoder.train import (
    TrainingResult,
    save_result,
    train_one,
    zero_positional_encodings,
)
from g2t_aml.utils.io import write_json
from g2t_aml.utils.logging import configure_logging, get_logger, stage
from g2t_aml.utils.run_context import RunContext
from g2t_aml.utils.seeding import seed_everything

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Case-size window for the attention figures. A three-node case shows nothing and a
#: 150-node one is unreadable, so the drawn examples come from the middle of the
#: distribution (Phase 2 median: 6 nodes, p90: 30).
FIGURE_MIN_NODES = 6
FIGURE_MAX_NODES = 25

#: The only splits whose embeddings the analysis reads: the linear probe fits on train and
#: scores on test. Retaining val and the realistic stream's embeddings costs about 200 MB
#: per arm for arrays nothing consumes.
ANALYSIS_SPLITS: frozenset[str] = frozenset({"train", "test"})

#: See D-051: `@hydra.main` discards its wrapped function's return value.
_EXIT_CODE: list[int] = []


def _device(requested: str, log: Any) -> torch.device:
    """Resolve the configured device, falling back to CPU with a warning."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        log.warning("experiment.device=%s but no CUDA device is visible; using cpu", requested)
        return torch.device("cpu")
    return torch.device(requested)


def _wandb(cfg: DictConfig, name: str, log: Any) -> Any:
    """Start a W&B run, or return None. Never blocks a training run on credentials."""
    try:
        import wandb
    except ImportError:
        return None
    try:
        return wandb.init(
            project="graph2text-aml",
            group="phase7-encoder",
            name=name,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode="online" if wandb.api.api_key else "offline",
            reinit=True,
        )
    except Exception as exc:  # pragma: no cover - network/credential failures vary
        log.warning("W&B logging disabled: %s", exc)
        return None


def _ensure_cache(cfg: DictConfig, log: Any) -> Path:
    """Build the feature cache if absent, then verify it against the frozen manifest."""
    processed = Path(cfg.paths.processed_dir) / str(cfg.data.interim_name)
    cache_dir = processed / "encoder" / "features"
    splits_dir = Path(str(cfg.data.split.manifest_dir))
    limit = cfg.experiment.get("limit_cases")

    if not (cache_dir / "cache_manifest.json").is_file():
        log.info("building the feature cache at %s", cache_dir)
        build_feature_cache(
            cases_dir=processed / "cases",
            realistic_dir=processed / "cases" / "realistic_test",
            interim_dir=Path(cfg.paths.interim_dir) / str(cfg.data.interim_name),
            splits_dir=splits_dir,
            facts_parquet=processed / "facts.parquet",
            out_dir=cache_dir,
            lap_pe_dim=int(cfg.training.lap_pe_dim),
            rw_pe_dim=int(cfg.training.rw_pe_dim),
            limit=int(limit) if limit else None,
            log=log,
        )
    if not limit:
        # A limited cache carries different id hashes on purpose, so it cannot be
        # mistaken for a full one; verification only applies to the real thing.
        verify_cache_against_manifest(cache_dir, splits_dir)
    return cache_dir


def _run_arm(
    cfg: DictConfig,
    arm: str,
    seed: int,
    space: Any,
    splits: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    log: Any,
    *,
    tag: str | None = None,
    use_edge_features: bool | None = None,
) -> tuple[Any, TrainingResult, dict[str, Any]]:
    """Train one arm at one seed and write its per-run artifacts."""
    label = tag or arm
    arm_cfg = OmegaConf.merge(cfg, {"encoder": {"arch": arm, "name": arm}})
    if arm != str(cfg.encoder.arch):
        loaded = OmegaConf.load(Path(CONFIG_DIR) / "encoder" / f"{arm}.yaml")
        arm_cfg = OmegaConf.merge(cfg, {"encoder": loaded})

    run = _wandb(arm_cfg, f"{label}-seed{seed}", log)
    log.info("training %s (seed %d)", label, seed)
    model, result, predictions = train_one(
        cfg=arm_cfg,
        space=space,
        splits=splits,
        seed=seed,
        device=device,
        checkpoint_dir=Path(cfg.paths.checkpoints_dir) / "encoder" / label,
        use_edge_features=use_edge_features,
        log=log,
    )
    result.arm = label
    save_result(result, run_dir / "runs" / f"{label}_seed{seed}.json")

    if run is not None:
        for record in result.history:
            run.log(
                {
                    "epoch": record.epoch,
                    "train/loss": record.train_loss,
                    "val/auc_pr": record.val_auc_pr,
                    "val/auc_roc": record.val_auc_roc,
                    "lr": record.learning_rate,
                }
            )
        run.log(
            {
                f"test/{k}": v
                for k, v in result.metrics.get("test", result.metrics["val"]).to_dict().items()
                if isinstance(v, int | float)
            }
        )
        run.finish()

    log.info(
        "  %s seed %d: test auc_pr %.4f  auc_roc %.4f  (best epoch %d of %d)",
        label,
        seed,
        result.metrics["test"].auc_pr,
        result.metrics["test"].auc_roc,
        result.best_epoch,
        result.epochs_run,
    )
    return model, result, predictions


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="config")
def _run(cfg: DictConfig) -> None:  # noqa: PLR0912, PLR0915 -- a linear sweep: cache, arms,
    # ablations, comparison, write-out. Splitting it would scatter the run directory.
    """Train and compare every encoder arm.

    Args:
        cfg: Composed Hydra configuration.

    Returns:
        Nothing. The exit code is recorded in ``_EXIT_CODE``; 1 when the corpus is absent
        or the GATv2-versus-MLP gate fails.
    """
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    configure_logging(log_file=run_dir / "train_encoder.log")
    log = get_logger(__name__)
    seeds_record = seed_everything(int(cfg.seed), deterministic=bool(cfg.deterministic))
    device = _device(str(cfg.experiment.device), log)

    with stage("train-encoder", log, run_dir=str(run_dir), device=str(device)) as summary:
        processed = Path(cfg.paths.processed_dir) / str(cfg.data.interim_name)
        if not (processed / "cases" / "cases.jsonl").is_file():
            log.warning("no case corpus at %s; run `make cases` first", processed / "cases")
            summary["status"] = "skipped: no case corpus"
            _EXIT_CODE.append(0)
            return

        cache_dir = _ensure_cache(cfg, log)
        space = load_feature_space(cache_dir)
        splits = {name: load_split(cache_dir, name) for name in MANIFEST_SPLITS}
        try:
            splits[REALISTIC_SPLIT] = load_split(cache_dir, REALISTIC_SPLIT)
        except Exception as exc:
            log.warning("realistic-imbalance stream unavailable: %s", exc)
        log.info("cache: %s", {k: len(v) for k, v in splits.items()} | {"node_dim": space.node_dim})

        arms = list(cfg.experiment.get("arms") or [str(cfg.encoder.arch)])
        seeds = [int(s) for s in cfg.training.seeds]
        results: dict[str, list[TrainingResult]] = {}
        predictions: dict[str, dict[int, Any]] = {}
        models: dict[str, Any] = {}

        # Embeddings are retained only where something reads them: the analysis arms at
        # the first seed. Everything else keeps scores and targets. Holding the full
        # `[n, 16, 256]` pooled tokens for nine arm-tags at three seeds across four splits
        # needs ~12 GB and OOM-killed the first full sweep 24 runs in.
        def _keep(prediction: dict[str, Any], *, embeddings: bool) -> dict[str, Any]:
            # Even where embeddings are kept, only train and test carry them: the probe
            # fits on train and scores on test, and val and the realistic stream are
            # never probed. Keeping all four would add 1.2 GB across six arms for arrays
            # nothing reads.
            return {
                name: value if embeddings and name in ANALYSIS_SPLITS else value.slim()
                for name, value in prediction.items()
            }

        for arm in arms:
            results[arm], predictions[arm] = [], {}
            for seed in seeds:
                model, result, prediction = _run_arm(
                    cfg, arm, seed, space, splits, device, run_dir, log
                )
                results[arm].append(result)
                predictions[arm][seed] = _keep(prediction, embeddings=seed == seeds[0])
                if seed == seeds[0]:
                    models[arm] = model

        # ------------------------------------------------------------ ablations ---
        ablations = cfg.experiment.get("ablations") or {}
        primary = str(cfg.encoder.arch)
        # Ablation predictions are kept alongside the arms', so each ablation gets a
        # *paired* bootstrap interval against the primary rather than only a mean over
        # seeds. Two marginal intervals cannot answer "did removing this cost anything?";
        # the interval on the difference can.
        if bool(ablations.get("positional_encodings")) and primary in arms:
            log.info("ablation: positional encodings off (%s)", primary)
            ablated = {k: zero_positional_encodings(v, space) for k, v in splits.items()}
            tag = f"{primary}_no_pe"
            predictions[tag] = {}
            for seed in seeds:
                _, result, prediction = _run_arm(
                    cfg, primary, seed, space, ablated, device, run_dir, log, tag=tag
                )
                results.setdefault(tag, []).append(result)
                predictions[tag][seed] = _keep(prediction, embeddings=False)

        if bool(ablations.get("edge_features")) and primary in arms:
            log.info("ablation: edge features zeroed (%s)", primary)
            tag = f"{primary}_no_edge"
            predictions[tag] = {}
            for seed in seeds:
                _, result, prediction = _run_arm(
                    cfg,
                    primary,
                    seed,
                    space,
                    splits,
                    device,
                    run_dir,
                    log,
                    tag=tag,
                    use_edge_features=False,
                )
                results.setdefault(tag, []).append(result)
                predictions[tag][seed] = _keep(prediction, embeddings=False)

        if bool(ablations.get("loss_function")) and primary in arms:
            log.info("ablation: weighted BCE instead of focal (%s)", primary)
            bce_cfg = OmegaConf.merge(cfg, {"training": {"loss": "weighted_bce"}})
            tag = f"{primary}_bce"
            predictions[tag] = {}
            for seed in seeds:
                _, result, prediction = _run_arm(
                    bce_cfg,
                    primary,
                    seed,
                    space,
                    splits,
                    device,
                    run_dir,
                    log,
                    tag=tag,
                )
                results.setdefault(tag, []).append(result)
                predictions[tag][seed] = _keep(prediction, embeddings=False)

        # ------------------------------------------------------------- analysis ---
        analyses: dict[str, Any] = {}
        for arm in arms:
            seed = seeds[0]
            train_prediction = predictions[arm][seed]["train"]
            test_prediction = predictions[arm][seed]["test"]
            analysis = analyse_embeddings(
                arm=arm,
                seed=seed,
                train_embeddings=train_prediction.graph_embeddings,
                train_labels=train_prediction.typology_targets,
                test_embeddings=test_prediction.graph_embeddings,
                test_labels=test_prediction.typology_targets,
                test_tokens=test_prediction.pooled_tokens,
                train_tokens=train_prediction.pooled_tokens,
                class_names=TYPOLOGY_CLASSES,
            )
            analyses[arm] = analysis.to_dict()
            save_analysis(analysis, run_dir / "analysis" / f"{arm}_seed{seed}.json")
            probe = next(
                (p for p in analysis.probes if p.representation == "pooled_tokens"),
                analysis.probes[0] if analysis.probes else None,
            )
            umap_figure(
                test_prediction.graph_embeddings,
                test_prediction.typology_targets,
                TYPOLOGY_CLASSES,
                Path(cfg.paths.figures_dir) / "encoder" / f"umap_{arm}.png",
                caption_metrics={f"kNN purity (k={p.k})": p.purity for p in analysis.purity[:1]}
                | ({"linear probe struct. macro-F1": probe.structural_macro_f1} if probe else {}),
                seed=seed,
            )

        # ------------------------------------------------- attention alignment ---
        alignment: dict[str, Any] = {}
        if primary in models:
            log.info("measuring attention alignment against the laundering path")
            suspicious = [g for g in splits["test"] if int(g.y.item()) == 1][:400]
            if suspicious:
                attentions = extract_attention(models[primary], suspicious, device)
                path_nodes = _path_nodes_for(cfg, [a.case_id for a in attentions], log)
                typologies = load_typologies(processed / "facts.parquet")
                report = path_attention_alignment(attentions, path_nodes, typologies)
                save_alignment(report, run_dir / "attention_alignment.json")
                alignment = report.to_dict()
                log.info(
                    "  attention on the laundering path: %.3f against a %.3f uniform "
                    "baseline (lift %.2f), top-1 hit rate %.3f",
                    report.mean_path_attention,
                    report.mean_path_share,
                    report.lift,
                    report.top1_hit_rate,
                )
                _draw_examples(
                    cfg, models[primary], suspicious, attentions, path_nodes, device, log
                )

        # ------------------------------------------------------------ the gate ---
        comparison = _compare(cfg, results, predictions, seeds, arms, log)
        report = {
            "arms": {
                name: {
                    "n_seeds": len(runs),
                    "n_parameters": runs[0].n_parameters,
                    "test": aggregate_over_seeds([r.metrics["test"].auc_pr for r in runs]),
                    "test_auc_roc": aggregate_over_seeds([r.metrics["test"].auc_roc for r in runs]),
                    "realistic": aggregate_over_seeds(
                        [
                            r.metrics[REALISTIC_SPLIT].auc_pr
                            for r in runs
                            if REALISTIC_SPLIT in r.metrics
                        ]
                        or [float("nan")]
                    ),
                    "typology_macro_f1_structural": aggregate_over_seeds(
                        [r.typology["test"].macro_f1_structural for r in runs]
                    ),
                    "typology_chance": aggregate_over_seeds(
                        [r.typology["test"].chance_macro_f1_structural for r in runs]
                    ),
                    "seconds": aggregate_over_seeds([r.seconds for r in runs]),
                }
                for name, runs in results.items()
            },
            "comparison": comparison,
            "embedding_analysis": analyses,
            "attention_alignment": alignment,
            "seeds": seeds,
            "device": str(device),
        }
        write_json(run_dir / "encoder_report.json", report)
        write_json(Path(cfg.paths.metrics_dir) / "encoder" / "encoder_report.json", report)
        RunContext.capture(
            experiment_name=str(cfg.experiment.name),
            cfg=cfg,
            seeds=seeds_record | {"arm_seeds": seeds},
            repo_root=REPO_ROOT,
            phase="7",
        ).save(run_dir)

        gate = comparison.get("gate", {})
        summary["arms"] = len(results)
        summary["gate_passed"] = gate.get("passed")
        summary["status"] = "ok" if gate.get("passed", True) else "gate failed"
        _EXIT_CODE.append(0 if gate.get("passed", True) else 1)


def _path_nodes_for(cfg: DictConfig, case_ids: list[str], log: Any) -> dict[str, set[str]]:
    """Materialise the named cases and return the accounts on each laundering path."""
    from g2t_aml.data.canonical import CanonicalGraph
    from g2t_aml.data.case_extraction import GraphIndex
    from g2t_aml.data.case_sampling import CaseCollection

    processed = Path(cfg.paths.processed_dir) / str(cfg.data.interim_name)
    try:
        collection = CaseCollection.load(processed / "cases")
        index = GraphIndex(
            CanonicalGraph.load(Path(cfg.paths.interim_dir) / str(cfg.data.interim_name))
        )
    except Exception as exc:  # pragma: no cover - only when the case store is absent
        log.warning("cannot measure attention alignment: %s", exc)
        return {}
    return {
        case_id: laundering_path_nodes(collection.materialise(case_id, index).edges)
        for case_id in case_ids
    }


def _draw_examples(
    cfg: DictConfig,
    model: Any,
    graphs: list[Any],
    attentions: list[Any],
    path_nodes: dict[str, set[str]],
    device: torch.device,
    log: Any,
) -> None:
    """Draw a handful of the highest-scoring cases as interpretability figures."""
    del model, device
    from g2t_aml.data.canonical import CanonicalGraph
    from g2t_aml.data.case_extraction import GraphIndex
    from g2t_aml.data.case_sampling import CaseCollection
    from g2t_aml.models.encoder.attention_viz import draw_case

    processed = Path(cfg.paths.processed_dir) / str(cfg.data.interim_name)
    try:
        collection = CaseCollection.load(processed / "cases")
        index = GraphIndex(
            CanonicalGraph.load(Path(cfg.paths.interim_dir) / str(cfg.data.interim_name))
        )
    except Exception as exc:  # pragma: no cover
        log.warning("cannot draw attention figures: %s", exc)
        return

    sizes = {a.case_id: len(a.node_ids) for a in attentions}
    candidates = [a for a in attentions if FIGURE_MIN_NODES <= sizes[a.case_id] <= FIGURE_MAX_NODES]
    chosen = sorted(candidates, key=lambda a: -a.risk_score)[:6]
    out_dir = Path(cfg.paths.figures_dir) / "encoder" / "attention"
    for attention in chosen:
        case = collection.materialise(attention.case_id, index)
        attention.typology = case.typology
        draw_case(
            attention,
            case.edges,
            out_dir / f"{attention.case_id}.png",
            path_nodes=path_nodes.get(attention.case_id),
        )
    log.info("wrote %d attention figures to %s", len(chosen), out_dir)
    del graphs


def _compare(
    cfg: DictConfig,
    results: dict[str, list[TrainingResult]],
    predictions: dict[str, dict[int, Any]],
    seeds: list[int],
    arms: list[str],
    log: Any,
) -> dict[str, Any]:
    """Run the Phase 7 gate comparison: GATv2 against the MLP control on AUC-PR."""
    primary, control = str(cfg.encoder.arch), "mlp"
    n_bootstrap = int(cfg.training.n_bootstrap)
    ci = float(cfg.training.ci)

    # Intervals at *every* seed, not just the first. A CI read off one seed is an
    # interval on that seed's model, and the reader cannot tell a genuine separation from
    # a lucky initialisation. The bootstrap is cheap here -- 2,000 resamples over 3,196
    # cases is a couple of seconds -- so there is no reason to economise.
    intervals: dict[str, dict[str, Any]] = {}
    for arm, runs in results.items():
        if arm not in predictions:
            continue
        intervals[arm] = {"n_seeds": len(runs), "per_seed": {}}
        for seed, prediction in predictions[arm].items():
            entry = {
                "test": bootstrap_auc_pr(
                    prediction["test"].scores,
                    prediction["test"].targets,
                    n_resamples=n_bootstrap,
                    ci=ci,
                    seed=int(cfg.seed),
                )
            }
            if REALISTIC_SPLIT in prediction:
                entry["realistic"] = bootstrap_auc_pr(
                    prediction[REALISTIC_SPLIT].scores,
                    prediction[REALISTIC_SPLIT].targets,
                    n_resamples=n_bootstrap,
                    ci=ci,
                    seed=int(cfg.seed),
                )
            intervals[arm]["per_seed"][str(seed)] = entry

    def _paired(arm_a: str, arm_b: str, split: str = "test") -> dict[str, Any]:
        """Bootstrap `arm_a - arm_b` on `split`, at every seed both arms were run at."""
        shared = sorted(set(predictions.get(arm_a, {})) & set(predictions.get(arm_b, {})))
        per_seed = {}
        for seed in shared:
            a, b = predictions[arm_a][seed][split], predictions[arm_b][seed][split]
            per_seed[str(seed)] = paired_difference(
                a.scores, b.scores, a.targets, n_resamples=n_bootstrap, ci=ci, seed=int(cfg.seed)
            )
        differences = [v["difference"] for v in per_seed.values()]
        return {
            "a": arm_a,
            "b": arm_b,
            "split": split,
            "per_seed": per_seed,
            "mean_difference": float(np.mean(differences)) if differences else float("nan"),
            # The strict reading: the difference must exclude zero in the same direction
            # at *every* seed. One seed out of three is a coin flip dressed as a result.
            "excludes_zero_at_every_seed": bool(per_seed)
            and all(v["excludes_zero"] for v in per_seed.values()),
            "positive_at_every_seed": bool(per_seed)
            and all(v["difference"] > 0 for v in per_seed.values()),
        }

    gate: dict[str, Any] = {"primary": primary, "control": control}
    if primary in predictions and control in predictions:
        difference = _paired(primary, control)
        realistic = _paired(primary, control, split=REALISTIC_SPLIT)
        first = str(seeds[0])
        primary_interval = intervals[primary]["per_seed"][first]["test"]
        control_interval = intervals[control]["per_seed"][first]["test"]
        gate |= {
            "paired_difference": difference,
            "paired_difference_realistic": realistic,
            "primary_ci": primary_interval,
            "control_ci": control_interval,
            # Reported because the brief asks for non-overlapping CIs specifically.
            "non_overlapping_marginal_cis": bool(primary_interval["lo"] > control_interval["hi"]),
            # The paired interval is the honest test and is what decides: two marginal
            # intervals can overlap while the difference excludes zero, because the arms'
            # errors are correlated across cases. Both are reported.
            "passed": bool(
                difference["excludes_zero_at_every_seed"] and difference["positive_at_every_seed"]
            ),
        }
        log.info(
            "GATE  %s - %s AUC-PR = %+.4f mean over %d seeds  -> %s",
            primary,
            control,
            difference["mean_difference"],
            len(difference["per_seed"]),
            "PASS" if gate["passed"] else "FAIL — halt and report (project-level finding)",
        )
        for seed, value in difference["per_seed"].items():
            log.info(
                "        seed %s: %+.4f  [%.4f, %.4f]  excludes zero: %s",
                seed,
                value["difference"],
                value["lo"],
                value["hi"],
                value["excludes_zero"],
            )
        if not gate["passed"]:
            log.error(
                "The MLP control matches %s. On this data topology carries no signal "
                "beyond node-local summary statistics, and the project premise needs "
                "revisiting. This is a finding, not a bug: report it.",
                primary,
            )

    # The primary against every other *arm*, paired and per seed. The Phase 7 brief asks
    # explicitly what to do if GIN beats GATv2, and that decision cannot be made from two
    # means and their standard deviations: with three seeds those overlap long before the
    # difference does. This is the evidence the primary-arm choice is made on.
    arm_comparison = {other: _paired(primary, other) for other in arms if other != primary}
    for other, value in arm_comparison.items():
        log.info(
            "ARM  %s - %s AUC-PR = %+.4f mean (excludes zero at every seed: %s)",
            primary,
            other,
            value["mean_difference"],
            value["excludes_zero_at_every_seed"],
        )

    # Each ablation against the primary, paired and per seed, so "the positional
    # encodings contributed X" is an interval rather than a difference of two means.
    ablation_comparison = {
        tag: _paired(primary, tag)
        for tag in predictions
        if tag.startswith(f"{primary}_") and tag != primary
    }
    for tag, value in ablation_comparison.items():
        log.info(
            "ABLATION  %s - %s AUC-PR = %+.4f mean (excludes zero at every seed: %s)",
            primary,
            tag,
            value["mean_difference"],
            value["excludes_zero_at_every_seed"],
        )

    return {
        "bootstrap": intervals,
        "gate": gate,
        "arms": arm_comparison,
        "ablations": ablation_comparison,
    }


def main() -> int:
    """Run the Hydra entrypoint and return its exit code.

    Returns:
        The code ``_run`` produced, or 0 when it produced none.
    """
    _run()
    return _EXIT_CODE[-1] if _EXIT_CODE else 0


if __name__ == "__main__":
    sys.exit(main())
