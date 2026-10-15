#!/usr/bin/env python
"""Train one Phase 9 generator arm, with the overfit check and the shuffled control wired in.

One arm per invocation, selected by ``experiment=generator_<arm>``. The arms are trained in
the priority order S1, A1, S2, B7, B8; the first four are the minimum viable set for the
central claim and B8 may go to Phase 11 if compute runs out.

**A1 is not optional.** ``experiment=generator_a1`` trains the shuffled control under
identical settings, and ``scripts/09b_compare_arms.py`` runs the comparison Gate 8 is
decided on. Skipping it does not save GPU time, it removes the result.

The run refuses to start until the 20-example overfit check has passed, unless
``training.require_overfit_check=false``.

After training the test split is decoded greedily into ``generations.jsonl`` in the run
directory, which is what ``scripts/10_evaluate.py`` and the Phase 11 matrix score. With
``generator.resume_from=<checkpoint>`` nothing is trained: the checkpoint is loaded and
decoded.

Usage:
    uv run python scripts/09_train_generator.py experiment=generator_s1
    uv run python scripts/09_train_generator.py experiment=generator_a1
    uv run python scripts/09_train_generator.py experiment=generator_debug
    uv run python scripts/09_train_generator.py experiment=generator_s1 training.epochs=1
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig

from g2t_aml.corpus.bronze.renderer import render_bronze
from g2t_aml.corpus.factsio import facts_from_dict
from g2t_aml.corpus.silver.claim_extraction import SlotAlignmentExtractor
from g2t_aml.facts.vocab import load_vocabulary
from g2t_aml.human.reservation import load_reservation
from g2t_aml.models.encoder.dataset import load_feature_space, load_split
from g2t_aml.models.encoder.registry import build_encoder
from g2t_aml.models.fusion.variants import build_fusion
from g2t_aml.models.generator.callbacks import (
    AttentionMassCallback,
    FaithfulnessCallback,
    ProbeCase,
)
from g2t_aml.models.generator.dataset import (
    GeneratorDataset,
    GraphCollator,
    build_curriculum,
    load_curriculum_records,
)
from g2t_aml.models.generator.inference import GenerationConfig, run_test_set
from g2t_aml.models.generator.model import (
    GeneratorConfig,
    LoraConfigSpec,
    QuantizationSpec,
    build_generator,
)
from g2t_aml.models.generator.profiling import RunProfile, device_info, profile_phase
from g2t_aml.models.generator.prompts import PromptBuilder
from g2t_aml.models.generator.train import GeneratorTrainer, TrainingConfig, overfit_check
from g2t_aml.utils.io import read_jsonl, write_json, write_jsonl
from g2t_aml.utils.logging import configure_logging, get_logger, stage
from g2t_aml.utils.run_context import RunContext
from g2t_aml.utils.seeding import seed_everything

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
REPO_ROOT = Path(__file__).resolve().parents[1]

#: See D-051: `@hydra.main` discards its wrapped function's return value.
_EXIT_CODE: list[int] = []


def _device(requested: str, log: Any) -> str:
    """Resolve the configured device, falling back to CPU with a warning.

    Args:
        requested: The configured device.
        log: The logger.

    Returns:
        The device to use.
    """
    if requested.startswith("cuda") and not torch.cuda.is_available():
        log.warning("experiment.device=%s but no CUDA device is visible; using cpu", requested)
        return "cpu"
    return requested


def _generator_config(cfg: DictConfig) -> GeneratorConfig:
    """Build the generator configuration from the composed Hydra config.

    Args:
        cfg: The resolved config.

    Returns:
        The generator configuration.
    """
    gen = cfg.generator
    return GeneratorConfig(
        base_model=str(gen.base_model),
        dtype=str(gen.dtype),
        attn_implementation=str(gen.attn_implementation),
        quantization=QuantizationSpec(
            load_in_4bit=bool(gen.quantization.load_in_4bit),
            bnb_4bit_quant_type=str(gen.quantization.bnb_4bit_quant_type),
            bnb_4bit_use_double_quant=bool(gen.quantization.bnb_4bit_use_double_quant),
            bnb_4bit_compute_dtype=str(gen.quantization.bnb_4bit_compute_dtype),
        ),
        lora=LoraConfigSpec(
            r=int(gen.lora.r),
            alpha=int(gen.lora.alpha),
            dropout=float(gen.lora.dropout),
            target_modules=tuple(str(m) for m in gen.lora.target_modules),
            modules_to_save=tuple(str(m) for m in gen.lora.modules_to_save),
            bias=str(gen.lora.bias),
        ),
        max_seq_len=int(gen.max_seq_len),
        gradient_checkpointing=bool(gen.gradient_checkpointing),
        freeze_encoder=bool(gen.freeze_encoder),
        text_mode=str(gen.text_mode),
        use_fusion=bool(gen.use_fusion),
    )


def _training_config(cfg: DictConfig) -> TrainingConfig:
    """Build the training configuration from the composed Hydra config.

    Args:
        cfg: The resolved config.

    Returns:
        The training regime.
    """
    training = cfg.training
    return TrainingConfig(
        epochs=int(training.epochs),
        lr=float(training.lr),
        fusion_lr=float(training.fusion_lr),
        encoder_lr=float(training.encoder_lr),
        scheduler=str(training.scheduler),
        warmup_ratio=float(training.warmup_ratio),
        per_device_batch_size=int(training.per_device_batch_size),
        gradient_accumulation=int(training.gradient_accumulation),
        max_seq_len=int(training.max_seq_len),
        gradient_checkpointing=bool(training.gradient_checkpointing),
        optim=str(training.optim),
        max_grad_norm=float(training.max_grad_norm),
        weight_decay=float(training.weight_decay),
        loss_on_completion_only=bool(training.loss_on_completion_only),
        bf16=bool(training.bf16),
        seed=int(cfg.seed),
        eval_every=int(training.eval_every),
        save_every=int(training.save_every) if training.save_every else None,
        require_overfit_check=bool(training.require_overfit_check),
        resume_every=int(training.resume_every) if training.get("resume_every") else None,
    )


def _load_graphs(cfg: DictConfig, log: Any) -> dict[str, Any]:
    """Load the Phase 7 feature cache, keyed by case id.

    Args:
        cfg: The resolved config.
        log: The logger.

    Returns:
        Case id to PyG ``Data``.

    Raises:
        FileNotFoundError: If the feature cache has not been built.
    """
    processed = Path(cfg.paths.processed_dir) / str(cfg.data.interim_name)
    cache_dir = processed / "encoder" / "features"
    if not (cache_dir / "cache_manifest.json").is_file():
        raise FileNotFoundError(
            f"no encoder feature cache at {cache_dir}; run `make encoder-features` first"
        )
    graphs: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        for data in load_split(cache_dir, split):
            graphs[str(data.case_id)] = data
    log.info("loaded %d cached graphs", len(graphs))
    return graphs


def _probe_cases(
    records: list[dict[str, Any]],
    graphs: dict[str, Any],
    builder: PromptBuilder,
    *,
    n: int,
    log: Any,
) -> list[ProbeCase]:
    """Build the fixed held-out probe cases the diagnostics generate from.

    The same cases at every checkpoint, in a deterministic order, so that drift in the
    curve is drift in the model rather than drift in the sample.

    Args:
        records: Validation records to draw from.
        graphs: The cached graphs.
        builder: The prompt builder.
        n: How many cases.
        log: The logger.

    Returns:
        The probe cases.
    """
    vocabulary = load_vocabulary()
    chosen = sorted(records, key=lambda r: str(r["case_id"]))[:n]
    cases: list[ProbeCase] = []
    for record in chosen:
        facts = facts_from_dict(dict(record["facts"]))
        bronze = render_bronze(facts, vocabulary=vocabulary)
        cases.append(
            ProbeCase(
                case_id=str(record["case_id"]),
                prompt=builder.build(record, for_training=False),
                graph=graphs.get(str(record["case_id"])),
                facts=facts,
                extractor=SlotAlignmentExtractor(bronze, vocabulary=vocabulary),
            )
        )
    log.info("probe cases: %s", ", ".join(c.case_id for c in cases))
    return cases


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="config")
def _run(cfg: DictConfig) -> None:
    """Train one generator arm.

    Args:
        cfg: The composed Hydra config.
    """
    configure_logging()
    log = get_logger("phase9")
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().run.dir)

    seeds = seed_everything(int(cfg.seed), deterministic=bool(cfg.deterministic))
    arm = str(cfg.experiment.arm)
    device = _device(str(cfg.experiment.device), log)

    context = RunContext.capture(
        experiment_name=str(cfg.experiment.name),
        cfg=cfg,
        seeds=seeds,
        repo_root=REPO_ROOT,
        arm=arm,
    )
    write_json(run_dir / "run_context.json", context.to_dict())

    generator_cfg = _generator_config(cfg)
    training_cfg = _training_config(cfg)
    profile = RunProfile(arm=arm, device=device_info(device))

    with stage("loading corpus and graphs", log):
        corpus_dir = Path(cfg.paths.processed_dir) / str(cfg.data.interim_name) / "corpus"
        reservation = load_reservation(str(cfg.data.split.manifest_dir))
        graphs = _load_graphs(cfg, log) if generator_cfg.use_fusion else None
        stages_spec = build_curriculum(cfg.training.get("curriculum"))

    with stage("building the model", log):
        encoder = None
        fusion = None
        if generator_cfg.use_fusion:
            processed = Path(cfg.paths.processed_dir) / str(cfg.data.interim_name)
            space = load_feature_space(processed / "encoder" / "features")
            encoder = build_encoder(cfg.encoder, space)
            checkpoint = (
                Path(cfg.paths.checkpoints_dir)
                / "encoder"
                / str(cfg.encoder.arch)
                / f"{cfg.encoder.arch}_seed{int(cfg.seed)}.pt"
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            encoder.load_state_dict(payload["state_dict"])
            log.info(
                "loaded encoder %s (val AUC-PR %.4f)", checkpoint.name, payload["best_val_auc_pr"]
            )
            fusion = build_fusion(cfg.fusion, graph_dim=encoder.hidden_dim)

        generator, tokenizer = build_generator(generator_cfg, fusion=fusion, encoder=encoder)
        # The assertion the brief asks for, at the moment it can still catch something.
        if generator.fusion_projector is not None:
            from g2t_aml.models.fusion.base import assert_projector_is_fp32

            assert_projector_is_fp32(generator.fusion_projector, name="fusion_projector")
            log.info("fusion projector verified fp32, %d soft tokens", generator.n_soft_tokens)

    builder = PromptBuilder(
        tokenizer,
        n_soft_tokens=generator.n_soft_tokens,
        soft_token_id=tokenizer.convert_tokens_to_ids(tokenizer.pad_token),
        text_mode=generator_cfg.text_mode,
        max_seq_len=generator_cfg.max_seq_len,
    )
    collator = GraphCollator(pad_token_id=tokenizer.pad_token_id)

    resume_from = cfg.generator.get("resume_from")
    if resume_from:
        with stage("loading the checkpoint", log):
            _load_weights(generator, Path(str(resume_from)), log)
    else:
        trained = _train_arm(
            cfg,
            generator,
            tokenizer=tokenizer,
            builder=builder,
            collator=collator,
            graphs=graphs,
            corpus_dir=corpus_dir,
            reservation=reservation,
            stages_spec=stages_spec,
            generator_cfg=generator_cfg,
            training_cfg=training_cfg,
            run_dir=run_dir,
            arm=arm,
            device=device,
            run_context=context.to_dict(),
            profile=profile,
            log=log,
        )
        if not trained:
            _EXIT_CODE.append(1)
            return

    with stage("decoding the test set", log) as summary:
        summary["n_cases"] = _decode_test_set(
            cfg,
            generator,
            tokenizer=tokenizer,
            builder=builder,
            collator=collator,
            graphs=graphs,
            corpus_dir=corpus_dir,
            run_dir=run_dir,
            arm=arm,
            device=device,
            log=log,
        )
    _EXIT_CODE.append(0)


def _train_arm(
    cfg: DictConfig,
    generator: Any,
    *,
    tokenizer: Any,
    builder: PromptBuilder,
    collator: GraphCollator,
    graphs: dict[str, Any] | None,
    corpus_dir: Path,
    reservation: Any,
    stages_spec: Any,
    generator_cfg: GeneratorConfig,
    training_cfg: TrainingConfig,
    run_dir: Path,
    arm: str,
    device: str,
    run_context: dict[str, Any],
    profile: RunProfile,
    log: Any,
) -> bool:
    """Run the overfit check and, if it passes, the curriculum.

    Args:
        cfg: The composed Hydra config.
        generator: The assembled generator.
        tokenizer: The tokeniser.
        builder: The prompt builder.
        collator: The collator.
        graphs: The cached graphs, or None on a text-only arm.
        corpus_dir: Where the corpus tiers live.
        reservation: The Gold reservation, or None.
        stages_spec: The curriculum stages.
        generator_cfg: The generator configuration.
        training_cfg: The training regime.
        run_dir: The run directory.
        arm: The arm's name.
        device: The device to train on.
        run_context: The run context, written into every checkpoint.
        profile: The run profile, extended in place with the training phase.
        log: The logger.

    Returns:
        False when the overfit check failed and the regime requires it; True otherwise.
    """
    with stage("assembling the curriculum", log):
        datasets = []
        for spec in stages_spec:
            records = load_curriculum_records(
                corpus_dir, split="train", tiers=spec.tiers, reservation=reservation
            )
            limit = cfg.experiment.get("limit_cases")
            if limit:
                records = records[: int(limit)]
            datasets.append(
                (
                    spec.name,
                    GeneratorDataset(
                        records, builder=builder, graphs=graphs, reservation=reservation
                    ),
                )
            )
            log.info("stage %s: %d records from %s", spec.name, len(records), spec.tiers)

    resume_file = cfg.training.get("resume_from")
    result: dict[str, float] | None = None
    if resume_file:
        # A continuation of a run that already passed the check and recorded the result in
        # its resume file. Its probe samples come forward so the drift record stays whole;
        # everything else the interrupted run wrote stays where it is (invariant 6).
        previous = Path(str(resume_file)).parent
        for name in (f"samples_{arm}.jsonl", "overfit_check.json"):
            if (previous / name).is_file() and not (run_dir / name).exists():
                shutil.copy2(previous / name, run_dir / name)
        write_json(
            run_dir / "resumed_from.json",
            {"resume_file": str(resume_file), "previous_run_dir": str(previous), "arm": arm},
        )
        log.info("continuing the run in %s; the overfit check is not repeated", previous)
    else:
        with stage("the overfit check", log):
            result = overfit_check(
                generator,
                datasets[0][1],
                collator,
                n_examples=20,
                n_steps=100,
                micro_batch_size=training_cfg.per_device_batch_size,
                device=device,
                log=log,
            )
            write_json(run_dir / "overfit_check.json", result)
            if not result["passed"] and training_cfg.require_overfit_check:
                log.error(
                    "the overfit check did not converge (%.4f). Something is miswired; a "
                    "full run would spend hours producing a model that looks fine and is not.",
                    result["final_loss"],
                )
                return False

    with stage("building the diagnostics", log):
        val_records = load_curriculum_records(
            corpus_dir, split="val", tiers=stages_spec[0].tiers, reservation=reservation
        )
        callbacks = [
            FaithfulnessCallback(
                _probe_cases(
                    val_records, graphs or {}, builder, n=int(cfg.training.n_probe_cases), log=log
                ),
                collator=collator,
                tokenizer=tokenizer,
                device=device,
                run_shuffled_control=generator_cfg.use_fusion,
                samples_path=run_dir / f"samples_{arm}.jsonl",
                log=log,
            ),
            AttentionMassCallback(log=log),
        ]

    trainer = GeneratorTrainer(
        generator,
        config=training_cfg,
        collator=collator,
        run_dir=run_dir,
        arm=arm,
        callbacks=callbacks,
        device=device,
        log=log,
        run_context=run_context,
    )

    with stage(f"training {arm}", log), profile_phase("train", arm=arm, device=device) as phase:
        state = trainer.train(
            datasets,
            overfit_result=result,
            resume_from=Path(str(resume_file)) if resume_file else None,
        )
        phase.n_examples = sum(len(d) for _, d in datasets) * training_cfg.epochs

    profile.add(phase)
    profile.write(run_dir / f"profile_{arm}.json")
    log.info(
        "%s finished at step %d; peak reserved %.2f GB",
        arm,
        state.step,
        phase.peak_reserved_gb or 0.0,
    )
    if any(getattr(c, "tracking_alarm", False) for c in callbacks):
        log.warning(
            "THE SHUFFLED CONTROL TRACKED THIS MODEL. Read %s before trusting this arm.",
            run_dir / f"history_{arm}.jsonl",
        )
    return True


def _load_weights(generator: Any, checkpoint: Path, log: Any) -> None:
    """Load a trained arm's adapters, fusion projector and encoder for decoding.

    Not :meth:`GeneratorTrainer.load_checkpoint`: that refuses a regime mismatch because it
    exists to *resume training*, and a decode-only run composes ``training.epochs=0``.
    Nothing here trains, so there is no run for a mismatched regime to be mislabelled as.

    Args:
        generator: The assembled generator, built from the same arm config.
        checkpoint: A file written by :meth:`GeneratorTrainer.save_checkpoint`.
        log: The logger.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        ValueError: If the checkpoint and the model disagree about the fusion layer, or
            the checkpoint carries no adapter weights.
    """
    if not checkpoint.is_file():
        raise FileNotFoundError(f"no generator checkpoint at {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    generator.load_state_for_checkpoint(payload)
    adapters = payload.get("lm_trainable_state")
    if not adapters:
        raise ValueError(f"{checkpoint.name} carries no adapter weights; nothing to decode with")
    outcome = generator.language_model.load_state_dict(adapters, strict=False)
    if outcome.unexpected_keys:
        raise ValueError(
            f"{checkpoint.name} holds {len(outcome.unexpected_keys)} adapter tensors this "
            f"model has no place for, e.g. {outcome.unexpected_keys[0]}"
        )
    log.info(
        "loaded %s (arm %s, step %s): %d adapter tensors",
        checkpoint.name,
        payload.get("arm"),
        payload.get("state", {}).get("step"),
        len(adapters),
    )


def _decode_test_set(
    cfg: DictConfig,
    generator: Any,
    *,
    tokenizer: Any,
    builder: PromptBuilder,
    collator: GraphCollator,
    graphs: dict[str, Any] | None,
    corpus_dir: Path,
    run_dir: Path,
    arm: str,
    device: str,
    log: Any,
) -> int:
    """Greedily decode the test split into ``generations.jsonl`` in the run directory.

    The file is the one every Phase 11 executor writes and ``scripts/10_evaluate.py``
    scores. Decoding is greedy and **unguarded**: the inference guard is a separate table
    row (D-073) and is not applied here, and every row says so.

    Args:
        cfg: The composed Hydra config.
        generator: The trained generator.
        tokenizer: The tokeniser.
        builder: The prompt builder.
        collator: The collator.
        graphs: The cached graphs, or None on a text-only arm.
        corpus_dir: Where the corpus tiers live.
        run_dir: The run directory.
        arm: The arm's name, recorded as the system on every row.
        device: The device to decode on.
        log: The logger.

    Returns:
        How many test cases the file holds.

    Raises:
        FileNotFoundError: If the Bronze corpus is missing.
    """
    bronze = corpus_dir / "bronze.jsonl"
    if not bronze.is_file():
        raise FileNotFoundError(f"no Bronze corpus at {bronze}; run `make bronze` first")
    # Read straight from the corpus rather than through the training loader: that loader
    # exists to keep reserved Gold test cases out of training, and these are test cases.
    records = sorted(
        (r for r in read_jsonl(bronze) if isinstance(r, dict) and r.get("split") == "test"),
        key=lambda r: str(r["case_id"]),
    )
    decode = cfg.generator.generate
    limit = decode.get("test_limit")
    if limit:
        records = records[: int(limit)]
        log.warning("decoding the first %d test cases by case id, not the whole split", limit)
    if cfg.training.guard.enabled:
        log.warning("the inference guard is not applied by this entrypoint; decoding unguarded")

    dataset = GeneratorDataset(records, builder=builder, graphs=graphs, for_training=False)
    out = run_dir / "generations.jsonl"
    if torch.cuda.is_available():
        # Training leaves the allocator holding blocks shaped for a backward pass.
        torch.cuda.empty_cache()
    run_test_set(
        generator,
        [dataset[i] for i in range(len(dataset))],
        collator=collator,
        tokenizer=tokenizer,
        cfg=GenerationConfig.deterministic(max_new_tokens=int(decode.max_new_tokens)),
        output_path=out,
        batch_size=int(decode.get("batch_size") or 4),
        device=device,
        log=log,
    )
    rows = [dict(row) for row in read_jsonl(out)]
    for row in rows:
        row.update(system=arm, seed=int(cfg.seed), stream="balanced", split="test", guard=False)
    write_jsonl(out, rows)
    return len(rows)


def main() -> int:
    """Run the Hydra entrypoint and return its exit code.

    Returns:
        The code ``_run`` produced, or 0 when it produced none.
    """
    _run()
    return _EXIT_CODE[-1] if _EXIT_CODE else 0


if __name__ == "__main__":
    sys.exit(main())
