"""The training loop: three parameter groups, a curriculum, and a checkpoint that reloads.

**Three learning rates, not one.** The single most consequential decision in this file.
The LoRA adapters modulate an already-trained 8B model and want 2e-4. The fusion projector
is randomly initialised and has to travel from wherever ``nn.Linear`` put it to inside the
language model's embedding distribution; at 2e-4 it is still in transit when the adapters
have converged, and a model whose graph channel is still noise learns to solve the task
from the text — permanently, because by the time the projector arrives the model no longer
has any use for it. That failure produces a healthy loss curve, a plausible narrative, and
S1 ≈ A1. The encoder, if unfrozen at all, wants 1e-5, because it was selected on val AUC-PR
in Phase 7 and the narrative loss will happily undo that selection.

**Gradient norms are logged per group** for the same reason. One aggregate norm cannot
distinguish "the projector is learning" from "the adapters are learning and the projector
is receiving nothing", and those look identical in the loss.

**The overfit test is not optional.** :func:`overfit_check` runs 20 examples for 100 steps
and requires the loss to approach zero. It takes under a minute and it catches the whole
class of wiring bugs — a mis-aligned loss mask, a detached fusion path, a splice that
overwrites the wrong positions — that otherwise surface as an eighteen-hour run producing
a model that generates fluent text with no relationship to its graph. The trainer refuses
a full run unless it has passed, unless explicitly overridden.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from g2t_aml.models.generator.dataset import GeneratorBatch, GeneratorDataset, GraphCollator
from g2t_aml.models.generator.model import Graph2TextGenerator
from g2t_aml.utils.io import ensure_dir

__all__ = [
    "TrainingConfig",
    "TrainingState",
    "GeneratorTrainer",
    "build_optimizer",
    "cosine_schedule_with_warmup",
    "overfit_check",
]


@dataclass(frozen=True)
class TrainingConfig:
    """The training regime, recorded verbatim into every checkpoint.

    Attributes:
        epochs: Total epochs across the curriculum.
        lr: LoRA learning rate.
        fusion_lr: Fusion projector learning rate. An order of magnitude above ``lr``, for
            the reason in the module docstring.
        encoder_lr: Encoder learning rate, used only when the encoder is unfrozen.
        scheduler: ``cosine`` or ``constant``.
        warmup_ratio: Fraction of total steps spent warming up.
        per_device_batch_size: Rows per forward pass.
        gradient_accumulation: Forward passes per optimiser step. Effective batch is the
            product; at 2 x 16 that is 32.
        max_seq_len: Truncation length.
        gradient_checkpointing: Trade compute for activation memory.
        optim: ``paged_adamw_8bit`` where bitsandbytes is available, else ``adamw``.
        max_grad_norm: Clipping threshold, applied across all groups jointly.
        weight_decay: L2 on the adapters and projector.
        loss_on_completion_only: Kept as an explicit switch so the ablation that measures
            what prompt-loss costs is a config change. Turning it off is not a supported
            training regime; it exists to be measured once.
        bf16: Run the LM in bfloat16.
        seed: The run seed.
        eval_every: Steps between diagnostic callbacks.
        save_every: Steps between checkpoints, or None to save only at stage boundaries.
        require_overfit_check: Refuse to start a full run until the overfit test passes.
    """

    epochs: int = 3
    lr: float = 2e-4
    fusion_lr: float = 1e-3
    encoder_lr: float = 1e-5
    scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    per_device_batch_size: int = 2
    gradient_accumulation: int = 16
    max_seq_len: int = 2048
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    max_grad_norm: float = 0.3
    weight_decay: float = 0.0
    loss_on_completion_only: bool = True
    bf16: bool = True
    seed: int = 42
    eval_every: int = 100
    save_every: int | None = 500
    require_overfit_check: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a plain mapping.

        Returns:
            A JSON-serialisable mapping, written into the checkpoint so a resume can
            refuse a checkpoint trained under a different regime (D-067).
        """
        return asdict(self)


@dataclass
class TrainingState:
    """Mutable progress, checkpointed alongside the weights.

    Attributes:
        step: Optimiser steps taken.
        epoch: Fractional epochs completed.
        stage: Name of the current curriculum stage.
        best_faithfulness: Best held-out supported-rate seen.
        history: Per-diagnostic-step records, written to the run directory as JSONL.
    """

    step: int = 0
    epoch: float = 0.0
    stage: str = ""
    best_faithfulness: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)


def build_optimizer(
    generator: Graph2TextGenerator, cfg: TrainingConfig, *, log: Any = None
) -> torch.optim.Optimizer:
    """Build the optimiser over the three parameter groups.

    Args:
        generator: The assembled generator.
        cfg: The training regime.
        log: Optional logger, used to record a fallback away from the paged optimiser.

    Returns:
        The optimiser. ``paged_adamw_8bit`` where bitsandbytes is importable **and the
        trainable parameters are on CUDA**, otherwise ``AdamW`` with the substitution
        logged — the paged optimiser exists to survive memory spikes on a small card, and
        silently getting a different one changes the memory profile Phase 13 reports.

        **Neither importability nor `torch.cuda.is_available()` is the right test.**
        ``PagedAdamW8bit`` constructs happily over CPU tensors and only raises
        ``ValueError: Expected a cuda device, but got: cpu`` from ``functional.pre_call``
        at the first ``.step()`` — so the failure lands mid-training rather than at setup.
        And a host can have a perfectly working CUDA runtime while the *model* sits on CPU:
        every CPU test in this repository, on a development laptop that does have a card,
        is exactly that case. What ``optimizer_update_8bit_blockwise`` rejects is the
        tensor, so the tensor is what gets checked. The fallback also costs nothing here —
        paging device memory is meaningless without a device.

    Raises:
        ValueError: If the generator has no trainable parameters at all, which means
            every component was frozen and the run would produce nothing.
    """
    groups = generator.trainable_parameter_groups(
        lora_lr=cfg.lr, fusion_lr=cfg.fusion_lr, encoder_lr=cfg.encoder_lr
    )
    if not groups:
        raise ValueError("no trainable parameters; every component is frozen")
    for group in groups:
        group["weight_decay"] = cfg.weight_decay

    if cfg.optim == "paged_adamw_8bit":
        reason: str | None = None
        # The parameters' device, not the machine's. A host can have a working CUDA
        # runtime while the model sits on CPU -- every CPU test on this project's own
        # development laptop is exactly that -- and it is the *tensor* that
        # `optimizer_update_8bit_blockwise` rejects.
        on_cuda = any(
            param.is_cuda
            for group in groups
            for param in group["params"]  # type: ignore[union-attr]
        )
        if not on_cuda:
            reason = "the trainable parameters are on CPU"
        else:
            try:
                import bitsandbytes as bnb

                return bnb.optim.PagedAdamW8bit(groups)
            except ImportError:
                reason = "bitsandbytes is unavailable"
        if log is not None:
            log.warning(
                "%s; falling back to AdamW. Memory profile will differ from the "
                "configured regime.",
                reason,
            )
    return torch.optim.AdamW(groups)


def cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer, *, warmup_steps: int, total_steps: int, kind: str = "cosine"
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a warmup-then-decay schedule that respects per-group base learning rates.

    ``LambdaLR`` multiplies each group's own ``lr``, which is what keeps the three
    learning rates in their configured ratio for the whole run. A scheduler that sets an
    absolute learning rate would collapse them to one and undo the entire point of the
    three groups.

    Args:
        optimizer: The optimiser.
        warmup_steps: Linear warmup length.
        total_steps: Total optimiser steps.
        kind: ``cosine`` or ``constant``.

    Returns:
        The scheduler.

    Raises:
        ValueError: If ``total_steps`` is not positive, or ``kind`` is unknown.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if kind not in {"cosine", "constant"}:
        raise ValueError(f"unknown scheduler {kind!r}; expected cosine or constant")

    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return step / max(1, warmup_steps)
        if kind == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def group_grad_norms(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    """Measure the gradient norm of each parameter group separately.

    One aggregate norm cannot tell a projector that is learning from a projector that is
    receiving no gradient at all, and the second is a wiring bug that trains happily.

    Args:
        optimizer: The optimiser, whose groups carry a ``name``.

    Returns:
        Group name to L2 gradient norm. A group with no gradients reports 0.0, which is
        itself the finding when it happens to the fusion group.
    """
    norms: dict[str, float] = {}
    for group in optimizer.param_groups:
        name = str(group.get("name", "group"))
        total = 0.0
        for param in group["params"]:
            if param.grad is not None:
                total += float(param.grad.detach().float().pow(2).sum())
        norms[f"grad_norm/{name}"] = math.sqrt(total)
    return norms


def _batches(
    dataset: GeneratorDataset, collator: GraphCollator, *, batch_size: int, shuffle: bool, seed: int
) -> Iterator[GeneratorBatch]:
    """Iterate a dataset in collated batches.

    Args:
        dataset: The dataset.
        collator: The collator.
        batch_size: Rows per batch.
        shuffle: Whether to shuffle the order.
        seed: Shuffle seed.

    Returns:
        An iterator of batches.
    """
    order = list(range(len(dataset)))
    if shuffle:
        generator = torch.Generator().manual_seed(seed)
        order = [order[i] for i in torch.randperm(len(order), generator=generator).tolist()]
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        yield collator([dataset[i] for i in chunk])


def overfit_check(
    generator: Graph2TextGenerator,
    dataset: GeneratorDataset,
    collator: GraphCollator,
    *,
    n_examples: int = 20,
    n_steps: int = 100,
    lr: float = 1e-3,
    fusion_lr: float = 1e-3,
    threshold: float = 0.15,
    device: str = "cpu",
    log: Any = None,
) -> dict[str, float]:
    """Train on a handful of examples until the loss collapses, or report that it did not.

    This is the cheapest test in the project and the one that pays for itself most often.
    A model that cannot memorise twenty examples in a hundred steps has a wiring bug — the
    loss mask is misaligned, the fusion path is detached from the graph, the splice is
    overwriting the wrong positions — and every one of those bugs produces a full training
    run that looks entirely normal and a model that is not doing what the paper says.

    Args:
        generator: The assembled generator.
        dataset: A dataset to draw the examples from.
        collator: The collator.
        n_examples: How many examples to memorise.
        n_steps: How many steps to allow.
        lr: LoRA learning rate for the check, higher than the real run's.
        fusion_lr: Fusion learning rate for the check.
        threshold: Final loss below this counts as passed.
        device: Device to run on.
        log: Optional logger.

    Returns:
        ``initial_loss``, ``final_loss``, ``threshold`` and ``passed`` as a mapping.

    Raises:
        ValueError: If the dataset holds fewer than ``n_examples`` examples.
    """
    if len(dataset) < n_examples:
        raise ValueError(
            f"need {n_examples} examples for the overfit check, dataset has {len(dataset)}"
        )

    subset = [dataset[i] for i in range(n_examples)]
    batch = collator(subset).to(device)
    generator.train()

    groups = generator.trainable_parameter_groups(
        lora_lr=lr, fusion_lr=fusion_lr, encoder_lr=fusion_lr
    )
    optimizer = torch.optim.AdamW(groups)

    initial = float("nan")
    loss_value = float("nan")
    for step in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        out = generator(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
            soft_mask=batch.soft_mask,
            graph_batch=batch.graph_batch,
        )
        if out.loss is None:
            raise ValueError("the model returned no loss; labels were not supplied")
        out.loss.backward()
        optimizer.step()
        loss_value = float(out.loss.detach())
        if step == 0:
            initial = loss_value
        if log is not None and step % 20 == 0:
            log.info("  overfit step %3d  loss %.4f", step, loss_value)

    passed = loss_value < threshold
    if log is not None:
        log.info(
            "  overfit check %s: %.4f -> %.4f (threshold %.2f)",
            "PASSED" if passed else "FAILED",
            initial,
            loss_value,
            threshold,
        )
    return {
        "initial_loss": initial,
        "final_loss": loss_value,
        "threshold": threshold,
        "passed": float(passed),
    }


class GeneratorTrainer:
    """Runs the curriculum, logs the diagnostics, and writes resumable checkpoints."""

    def __init__(
        self,
        generator: Graph2TextGenerator,
        *,
        config: TrainingConfig,
        collator: GraphCollator,
        run_dir: str | Path,
        arm: str = "S1",
        callbacks: Sequence[Any] = (),
        device: str = "cpu",
        log: Any = None,
        run_context: dict[str, Any] | None = None,
    ) -> None:
        """Build the trainer.

        Args:
            generator: The assembled generator.
            config: The training regime.
            collator: The collator.
            run_dir: Where checkpoints, the history JSONL and the profile are written.
            arm: Which system this run is — S1, A1, S2, B7, B8. Recorded on every
                checkpoint and every history row so two arms' traces cannot be confused.
            callbacks: Objects with an ``on_step`` method, run every ``eval_every``
                steps. See :mod:`~g2t_aml.models.generator.callbacks`.
            device: Device to train on.
            log: Optional logger.
            run_context: The provenance snapshot (invariant 5), written into every
                checkpoint.
        """
        self.generator = generator
        self.config = config
        self.collator = collator
        self.run_dir = ensure_dir(run_dir)
        self.arm = arm
        self.callbacks = list(callbacks)
        self.device = device
        self.log = log
        self.run_context = run_context or {}
        self.state = TrainingState()
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any = None

    def _total_steps(self, dataset_size: int) -> int:
        """Compute how many optimiser steps the run will take.

        Args:
            dataset_size: Examples per epoch.

        Returns:
            The step count, at least 1.
        """
        per_epoch = math.ceil(dataset_size / self.config.per_device_batch_size)
        return max(1, math.ceil(per_epoch * self.config.epochs / self.config.gradient_accumulation))

    def train(
        self,
        stages: Sequence[tuple[str, GeneratorDataset]],
        *,
        overfit_result: dict[str, float] | None = None,
    ) -> TrainingState:
        """Run the curriculum.

        Args:
            stages: ``(stage_name, dataset)`` in curriculum order.
            overfit_result: The result of :func:`overfit_check`. Required when
                ``config.require_overfit_check`` is set.

        Returns:
            The final training state.

        Raises:
            RuntimeError: If the overfit check is required and did not pass. Eighteen
                hours is too long to spend finding out that the loss mask was misaligned.
            ValueError: If no stages were given.
        """
        if not stages:
            raise ValueError("no curriculum stages to train on")
        if self.config.require_overfit_check and not (overfit_result or {}).get("passed"):
            raise RuntimeError(
                "the overfit check has not passed; run overfit_check() first or set "
                "require_overfit_check=false to override. A full run costs hours and "
                "every wiring bug this catches is invisible in the loss curve."
            )

        total = self._total_steps(sum(len(d) for _, d in stages))
        self.optimizer = build_optimizer(self.generator, self.config, log=self.log)
        self.scheduler = cosine_schedule_with_warmup(
            self.optimizer,
            warmup_steps=int(total * self.config.warmup_ratio),
            total_steps=total,
            kind=self.config.scheduler,
        )

        for stage_name, dataset in stages:
            self.state.stage = stage_name
            if self.log is not None:
                self.log.info("stage %s: %d examples", stage_name, len(dataset))
            self._run_stage(dataset)
            self.save_checkpoint(tag=f"stage_{stage_name}")

        self.save_checkpoint(tag="final")
        self._write_history()
        return self.state

    def _run_stage(self, dataset: GeneratorDataset) -> None:
        """Train one curriculum stage to completion.

        Args:
            dataset: The stage's dataset.
        """
        assert self.optimizer is not None and self.scheduler is not None
        self.generator.train()
        accumulated = 0
        started = time.perf_counter()
        tokens_seen = 0

        for raw_batch in _batches(
            dataset,
            self.collator,
            batch_size=self.config.per_device_batch_size,
            shuffle=True,
            seed=self.config.seed + self.state.step,
        ):
            batch = raw_batch.to(self.device)
            out = self.generator(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
                soft_mask=batch.soft_mask,
                graph_batch=batch.graph_batch,
            )
            if out.loss is None:
                raise ValueError("the model returned no loss; labels were not supplied")

            (out.loss / self.config.gradient_accumulation).backward()
            accumulated += 1
            tokens_seen += int(batch.attention_mask.sum())

            if accumulated < self.config.gradient_accumulation:
                continue

            norms = group_grad_norms(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for g in self.optimizer.param_groups for p in g["params"]],
                self.config.max_grad_norm,
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            self.state.step += 1

            if self.state.step % self.config.eval_every == 0:
                elapsed = max(1e-9, time.perf_counter() - started)
                self._diagnose(
                    batch,
                    out,
                    norms,
                    tokens_per_second=tokens_seen / elapsed,
                )
            if self.config.save_every and self.state.step % self.config.save_every == 0:
                self.save_checkpoint(tag=f"step_{self.state.step}")

    def _diagnose(
        self,
        batch: GeneratorBatch,
        out: Any,
        norms: dict[str, float],
        *,
        tokens_per_second: float,
    ) -> None:
        """Assemble one diagnostic row and hand it to the callbacks.

        Args:
            batch: The batch that produced the step.
            out: The generator output.
            norms: Per-group gradient norms.
            tokens_per_second: Throughput since the stage started.
        """
        from g2t_aml.models.fusion.diagnostics import gate_summary

        row: dict[str, Any] = {
            "arm": self.arm,
            "step": self.state.step,
            "stage": self.state.stage,
            "loss": float(out.loss.detach()),
            "tokens_per_second": tokens_per_second,
            "n_supervised_tokens": batch.n_supervised_tokens,
            **norms,
            **{
                f"lr/{g.get('name', i)}": g["lr"]
                for i, g in enumerate(self.optimizer.param_groups)  # type: ignore[union-attr]
            },
        }
        if out.fusion is not None:
            mean, low, high = gate_summary(out.fusion.gate)
            row.update(
                {
                    "gate_mean": mean,
                    "gate_min": low,
                    "gate_max": high,
                    "soft_token_norm_mean": float(out.fusion.token_norms.mean()),
                    "pre_gate_rms": float(out.fusion.pre_gate_rms),
                }
            )

        for callback in self.callbacks:
            extra = callback.on_step(self.generator, self.state, batch)
            if extra:
                row.update(extra)

        self.state.history.append(row)
        if self.log is not None:
            self.log.info(
                "step %5d  loss %.4f  gate %s  grad(fusion) %s",
                self.state.step,
                row["loss"],
                f"{row['gate_mean']:.3f}" if row.get("gate_mean") is not None else "n/a",
                f"{row.get('grad_norm/fusion', 0.0):.2e}",
            )

    def _write_history(self) -> None:
        """Write the diagnostic history to the run directory as JSONL."""
        from g2t_aml.utils.io import write_jsonl

        write_jsonl(self.run_dir / f"history_{self.arm}.jsonl", self.state.history)

    def save_checkpoint(self, *, tag: str) -> Path:
        """Write a checkpoint carrying everything needed to reproduce the model.

        The LoRA adapters, the fusion projector, the encoder, the optimiser, the training
        regime and the run context all go in. **The fusion and encoder state are the
        parts a naive checkpoint omits**, because PEFT's ``save_pretrained`` does not know
        about them, and a checkpoint missing them reloads into a model whose graph pathway
        is randomly initialised — which runs, generates fluent text, and is not the model
        that was evaluated.

        Args:
            tag: Checkpoint name suffix.

        Returns:
            The path written.
        """
        path = self.run_dir / f"{self.arm}_{tag}.pt"
        payload: dict[str, Any] = {
            "arm": self.arm,
            "tag": tag,
            "state": asdict(self.state),
            "training_config": self.config.to_dict(),
            "run_context": self.run_context,
            **self.generator.state_for_checkpoint(),
        }
        lm = self.generator.language_model
        if hasattr(lm, "state_dict"):
            payload["lm_trainable_state"] = {
                name: param.detach().cpu()
                for name, param in lm.state_dict().items()
                if "lora" in name.lower()
            }
        torch.save(payload, path)
        return path

    def load_checkpoint(self, path: str | Path, *, strict: bool = True) -> dict[str, Any]:
        """Restore a checkpoint written by :meth:`save_checkpoint`.

        Args:
            path: The checkpoint file.
            strict: Require every state-dict key to match.

        Returns:
            The payload, so the caller can inspect the recorded regime and run context.

        Raises:
            ValueError: If the checkpoint was trained under a different regime. Following
                D-067: a checkpoint carries no evidence of how long it trained once its
                weights are loaded, and a smoke-run checkpoint resumes exactly as happily
                as a converged one.
        """
        payload = torch.load(path, map_location="cpu", weights_only=False)
        recorded = payload.get("training_config", {})
        mismatched = {
            key: (recorded.get(key), getattr(self.config, key))
            for key in ("epochs", "lr", "fusion_lr", "per_device_batch_size", "max_seq_len")
            if key in recorded and recorded[key] != getattr(self.config, key)
        }
        if mismatched:
            listed = ", ".join(f"{k}: {was!r} != {now!r}" for k, (was, now) in mismatched.items())
            raise ValueError(
                f"checkpoint {Path(path).name} was trained under a different regime ({listed}); "
                "resuming it would put one run's numbers under another run's label"
            )

        self.generator.load_state_for_checkpoint(payload, strict=strict)
        lm_state = payload.get("lm_trainable_state")
        if lm_state and hasattr(self.generator.language_model, "load_state_dict"):
            self.generator.language_model.load_state_dict(lm_state, strict=False)

        saved = payload.get("state", {})
        self.state = TrainingState(
            step=int(saved.get("step", 0)),
            epoch=float(saved.get("epoch", 0.0)),
            stage=str(saved.get("stage", "")),
            best_faithfulness=float(saved.get("best_faithfulness", 0.0)),
            history=list(saved.get("history", [])),
        )
        return payload
