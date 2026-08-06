"""Training one encoder arm at one seed, and evaluating it honestly.

Three things in here are load-bearing rather than boilerplate.

**Model selection is on validation AUC-PR.** Not loss, not accuracy, not ROC-AUC. The
checkpoint that gets evaluated on test is the one that maximised validation AUC-PR, and
early stopping counts patience against the same quantity. Selecting on anything else and
then reporting AUC-PR is a silent way of reporting a number the run never optimised.

**val is used for selection and for nothing else.** It is a three-day temporal band and
Phase 2 recorded that it contains no ``fan_in``, ``gather_scatter`` or ``scatter_gather``
cases at all. A per-typology breakdown computed on it would report F1 = 0 on three
classes that simply are not there. Per-typology numbers come from test, which covers all
nine.

**Every model is evaluated on both populations.** The balanced construction the model was
trained on, and the realistic-imbalance stream at 7.3% prevalence (D-023). The second is
the honest operating point and the first is the one the training distribution matches;
reporting only one of them would be choosing which question to answer after seeing the
answer.
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from g2t_aml.models.encoder.dataset import TYPOLOGY_CLASSES
from g2t_aml.models.encoder.features import FeatureSpace
from g2t_aml.models.encoder.losses import (
    EncoderLoss,
    build_binary_loss,
    inverse_frequency_weights,
)
from g2t_aml.models.encoder.metrics import (
    BinaryMetrics,
    TypologyMetrics,
    binary_metrics,
    typology_metrics,
)
from g2t_aml.models.encoder.registry import build_encoder, count_parameters
from g2t_aml.utils.io import write_json

if TYPE_CHECKING:  # pragma: no cover
    from torch_geometric.data import Data

    from g2t_aml.models.encoder.base import BaseEncoder


@dataclass
class Predictions:
    """Raw model output over one population, kept for downstream analysis.

    Attributes:
        case_ids: One per case, in evaluation order.
        scores: Sigmoid risk probability per case.
        targets: Binary ground truth per case.
        typology_predictions: Argmax typology class index per case.
        typology_targets: True typology class index, ``-1`` where absent.
        graph_embeddings: ``[n, d]`` graph embeddings, for the probe and the UMAP.
        pooled_tokens: ``[n, k, d]`` pooled tokens, for the fusion-layer probe.
    """

    case_ids: list[str]
    scores: np.ndarray
    targets: np.ndarray
    typology_predictions: np.ndarray
    typology_targets: np.ndarray
    graph_embeddings: np.ndarray
    pooled_tokens: np.ndarray

    def slim(self) -> Predictions:
        """Return a copy without the embedding arrays.

        The two embedding tensors dominate memory: at 10,932 training cases,
        ``pooled_tokens`` alone is 179 MB per split per run, and a sweep holding them for
        nine arm-tags at three seeds across four splits needs about 12 GB. Only the
        embedding battery reads them, and only for the arms at the first seed; every other
        consumer — the bootstrap, the gate, the per-arm metrics — needs `scores` and
        `targets`, which are two float arrays per split.

        This is not an optimisation. An earlier revision retained everything and the
        sweep was OOM-killed 24 runs in on a 7 GB machine.

        Returns:
            A new ``Predictions`` sharing the small arrays and holding empty embeddings.
        """
        return dataclasses.replace(
            self,
            graph_embeddings=np.empty((0, 0), dtype=np.float32),
            pooled_tokens=np.empty((0, 0, 0), dtype=np.float32),
        )


@dataclass
class EpochRecord:
    """One epoch's training and validation summary."""

    epoch: int
    train_loss: float
    train_risk_loss: float
    train_typology_loss: float
    val_auc_pr: float
    val_auc_roc: float
    learning_rate: float
    seconds: float


@dataclass
class TrainingResult:
    """Everything one (arm, seed) run produced.

    Attributes:
        arm: Architecture name.
        seed: The seed this run used.
        best_epoch: The epoch whose checkpoint was selected.
        best_val_auc_pr: Validation AUC-PR at that epoch.
        epochs_run: How many epochs actually ran before early stopping.
        n_parameters: Trainable parameter count.
        history: Per-epoch records.
        metrics: Population name to binary metrics.
        typology: Population name to typology metrics.
        seconds: Wall-clock training time.
        checkpoint: Path the selected checkpoint was written to, if any.
    """

    arm: str
    seed: int
    best_epoch: int
    best_val_auc_pr: float
    epochs_run: int
    n_parameters: int
    history: list[EpochRecord] = field(default_factory=list)
    metrics: dict[str, BinaryMetrics] = field(default_factory=dict)
    typology: dict[str, TypologyMetrics] = field(default_factory=dict)
    seconds: float = 0.0
    checkpoint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary.

        Returns:
            Every field, with the metric dataclasses expanded.
        """
        payload = asdict(self)
        payload["metrics"] = {k: v.to_dict() for k, v in self.metrics.items()}
        payload["typology"] = {k: v.to_dict() for k, v in self.typology.items()}
        return payload


def zero_positional_encodings(graphs: list[Data], space: FeatureSpace) -> list[Data]:
    """Return copies of the graphs with the positional-encoding block zeroed.

    The PE ablation, done by masking rather than by rebuilding the cache: the input width
    stays identical, so the ablated model has exactly the same parameter count and the
    comparison isolates the information rather than the capacity.

    Args:
        graphs: Encoded cases.
        space: The feature space, which knows where the block is.

    Returns:
        New ``Data`` objects sharing everything but ``x``.
    """
    start, stop = space.pe_slice
    out = []
    for graph in graphs:
        clone = graph.clone()
        clone.x = clone.x.clone()
        clone.x[:, start:stop] = 0.0
        out.append(clone)
    return out


def flip_lap_pe_signs(
    x: Tensor,
    batch_index: Tensor,
    n_graphs: int,
    space: FeatureSpace,
    generator: torch.Generator,
) -> Tensor:
    """Randomly flip the sign of each Laplacian eigenvector component, per graph.

    An eigenvector and its negation are equally valid, and ``numpy.linalg.eigh`` picks
    one arbitrarily — meaning a case's encoding can flip between two runs of the cache
    build. Flipping at random during training forces the model to learn a sign-invariant
    function instead of memorising whichever sign the builder happened to emit.

    The flip is drawn **per graph**, not per batch. Each case's eigenbasis has its own
    independent sign ambiguity, so one draw shared across a batch would teach the model
    that all cases in a batch flip together — an artefact of the batching rather than a
    property of the encoding.

    Args:
        x: ``[N, node_dim]`` node features. Modified in place.
        batch_index: ``[N]`` graph assignment per node.
        n_graphs: Number of graphs in the batch.
        space: The feature space, which knows the Laplacian block's location.
        generator: CPU torch generator, so the flips are seed-controlled and reproducible
            independently of which device training runs on.

    Returns:
        The same tensor, for chaining.
    """
    start, stop = space.lap_pe_slice
    if stop <= start:
        return x
    draws = torch.randint(0, 2, (n_graphs, stop - start), generator=generator)
    signs = (draws.float() * 2 - 1).to(x.device)
    x[:, start:stop] = x[:, start:stop] * signs[batch_index]
    return x


@torch.no_grad()
def predict(
    model: BaseEncoder,
    graphs: list[Data],
    device: torch.device,
    *,
    batch_size: int = 256,
) -> Predictions:
    """Run a trained model over a population and collect everything downstream needs.

    Args:
        model: The arm to run.
        graphs: The encoded cases.
        device: Device to run on.
        batch_size: Evaluation batch size, larger than training since no backward graph
            is retained.

    Returns:
        The raw predictions and embeddings.
    """
    model.eval()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

    case_ids: list[str] = []
    scores: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    typology_predictions: list[np.ndarray] = []
    typology_targets: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    tokens: list[np.ndarray] = []

    for cpu_batch in loader:
        batch = cpu_batch.to(device)
        out = model(batch)
        scores.append(torch.sigmoid(out.risk_logits.reshape(-1)).cpu().numpy())
        targets.append(batch.y.reshape(-1).cpu().numpy())
        embeddings.append(out.graph_embedding.cpu().numpy())
        tokens.append(out.pooled_tokens.cpu().numpy())
        typology_targets.append(batch.y_typ.reshape(-1).cpu().numpy())
        typology_predictions.append(
            out.typology_logits.argmax(dim=-1).cpu().numpy()
            if out.typology_logits is not None
            else np.full(int(batch.num_graphs), -1)
        )
        case_ids.extend(batch.case_id)

    return Predictions(
        case_ids=case_ids,
        scores=np.concatenate(scores),
        targets=np.concatenate(targets),
        typology_predictions=np.concatenate(typology_predictions),
        typology_targets=np.concatenate(typology_targets),
        graph_embeddings=np.concatenate(embeddings),
        pooled_tokens=np.concatenate(tokens),
    )


#: Training keys a resumable checkpoint must agree with the current config on. Not the
#: whole config: `resume`, `seeds` and the bootstrap settings say nothing about how the
#: weights were produced, and requiring them to match would refuse every legitimate
#: resume. These four are what a results table means by "trained the same way".
RESUME_CRITICAL_KEYS: tuple[str, ...] = ("epochs", "loss", "lr", "batch_size")


def _same_setting(saved: Any, current: Any) -> bool:
    """Compare two config values across the string/number boundary.

    ``lr: 1e-3`` is a float to OmegaConf and the *string* ``"1e-3"`` to PyYAML, so a
    naive string comparison reports a regime change on a config nothing has touched and
    silently retrains everything. Numbers are compared numerically; everything else is
    compared as text.

    Args:
        saved: The value stored in the checkpoint.
        current: The value in the active config.

    Returns:
        True when the two denote the same setting.
    """
    try:
        return float(saved) == float(current)
    except (TypeError, ValueError):
        return str(saved) == str(current)


def _checkpoint_regime_matches(checkpoint: Path, training: Any, *, log: Any = None) -> bool:
    """Report whether a checkpoint was trained under the current training regime.

    A checkpoint carries no evidence of how long it trained once its weights are loaded,
    so a two-epoch smoke checkpoint left in the checkpoint directory resumes as happily
    as a converged one and puts a smoke-run number into a results table. **This happened
    during Phase 7** and is the reason the check exists.

    A checkpoint written before this field was introduced has no ``training_config`` and
    is refused, which costs one retrain and cannot produce a wrong number.

    Args:
        checkpoint: The checkpoint to inspect.
        training: The current ``cfg.training`` node.
        log: Optional logger.

    Returns:
        True when every key in :data:`RESUME_CRITICAL_KEYS` agrees.
    """
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - a corrupt checkpoint is not resumable
        if log is not None:
            log.warning("  cannot read %s: %s", checkpoint.name, exc)
        return False

    saved = payload.get("training_config")
    if not saved:
        if log is not None:
            log.warning(
                "  %s carries no training_config; it predates the resume guard and is "
                "refused rather than trusted",
                checkpoint.name,
            )
        return False

    for key in RESUME_CRITICAL_KEYS:
        if not _same_setting(saved.get(key), training.get(key)):
            if log is not None:
                log.warning(
                    "  %s was trained with %s=%s, config says %s",
                    checkpoint.name,
                    key,
                    saved.get(key),
                    training.get(key),
                )
            return False
    return True


def _resume_from_checkpoint(
    *,
    model: BaseEncoder,
    checkpoint: Path,
    splits: dict[str, list[Data]],
    device: torch.device,
    arm: str,
    seed: int,
    n_parameters: int,
    eval_batch_size: int,
    log: Any = None,
) -> tuple[BaseEncoder, TrainingResult, dict[str, Predictions]]:
    """Re-evaluate a completed run from its saved checkpoint.

    The metrics are **recomputed** from the restored weights rather than read out of the
    old result JSON. Reading them back would make the summary a copy of a file rather
    than a measurement, and would silently survive a checkpoint that no longer loads into
    the current architecture.

    Args:
        model: A freshly built arm of the right architecture.
        checkpoint: The saved checkpoint.
        splits: Split name to encoded cases.
        device: Device to evaluate on.
        arm: Architecture name.
        seed: The seed this run used.
        n_parameters: Trainable parameter count.
        eval_batch_size: Evaluation batch size.
        log: Optional logger.

    Returns:
        ``(model, result, predictions)``, exactly as :func:`train_one` returns them.
        ``history`` is empty and ``seconds`` is zero: this run did no training, and
        pretending otherwise would put fabricated timings in the results table.
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model = model.to(device)
    if log is not None:
        log.info(
            "  resumed %s seed %d from %s (val AUC-PR %.4f, epoch %d)",
            arm,
            seed,
            checkpoint.name,
            payload.get("best_val_auc_pr", float("nan")),
            payload.get("best_epoch", -1),
        )

    predictions = {
        name: predict(model, graphs, device, batch_size=eval_batch_size)
        for name, graphs in splits.items()
    }
    result = TrainingResult(
        arm=arm,
        seed=seed,
        best_epoch=int(payload.get("best_epoch", -1)),
        best_val_auc_pr=float(payload.get("best_val_auc_pr", float("nan"))),
        epochs_run=0,
        n_parameters=n_parameters,
        history=[],
        metrics={name: binary_metrics(p.scores, p.targets) for name, p in predictions.items()},
        typology={
            name: typology_metrics(p.typology_predictions, p.typology_targets, TYPOLOGY_CLASSES)
            for name, p in predictions.items()
        },
        seconds=0.0,
        checkpoint=str(checkpoint),
    )
    return model, result, predictions


def train_one(  # noqa: PLR0912, PLR0915 -- one training loop with its configuration
    # spelled out; every argument corresponds to a documented config key and hiding them
    # behind a bag would make the resolved config harder to read, not easier.
    *,
    cfg: Any,
    space: FeatureSpace,
    splits: dict[str, list[Data]],
    seed: int,
    device: torch.device,
    checkpoint_dir: Path | None = None,
    use_edge_features: bool | None = None,
    log: Any = None,
) -> tuple[BaseEncoder, TrainingResult, dict[str, Predictions]]:
    """Train one arm at one seed and evaluate it on every population.

    Args:
        cfg: The composed Hydra config. Reads ``cfg.encoder`` and ``cfg.training``.
        space: The fitted feature space.
        splits: Split name to encoded cases. Must contain ``train`` and ``val``.
        seed: The seed for this run.
        device: Device to train on.
        checkpoint_dir: Where to write the selected checkpoint, or None to skip.
        use_edge_features: Override for the edge-feature ablation.
        log: Optional logger.

    Returns:
        ``(model, result, predictions)`` — the model restored to its selected weights,
        the run summary, and per-population predictions.

    Raises:
        KeyError: If ``train`` or ``val`` is absent from ``splits``.
    """
    from g2t_aml.utils.seeding import seed_everything

    seed_everything(seed, deterministic=bool(cfg.deterministic))
    training = cfg.training

    model = build_encoder(cfg.encoder, space, use_edge_features=use_edge_features).to(device)
    n_parameters = count_parameters(model)

    # Resume: a completed (arm, seed) is re-evaluated from its checkpoint rather than
    # retrained. Selection already happened when that checkpoint was written, so
    # re-running the loop would produce the same weights at considerable cost. This is
    # what makes a six-arm three-seed sweep survivable on one machine -- the first full
    # sweep was OOM-killed 24 runs in, and resume recovered all 24.
    resume_path = checkpoint_dir / f"{cfg.encoder.arch}_seed{seed}.pt" if checkpoint_dir else None
    if bool(training.get("resume")) and resume_path is not None and resume_path.is_file():
        if _checkpoint_regime_matches(resume_path, training, log=log):
            return _resume_from_checkpoint(
                model=model,
                checkpoint=resume_path,
                splits=splits,
                device=device,
                arm=str(cfg.encoder.arch),
                seed=seed,
                n_parameters=n_parameters,
                eval_batch_size=int(training.eval_batch_size),
                log=log,
            )
        if log is not None:
            log.warning(
                "  %s seed %d: checkpoint was trained under a different regime; "
                "retraining rather than resuming it",
                cfg.encoder.arch,
                seed,
            )

    train_graphs, val_graphs = splits["train"], splits["val"]
    train_targets = torch.tensor([int(g.y.item()) for g in train_graphs])
    n_positive = int(train_targets.sum())
    n_negative = len(train_graphs) - n_positive
    # Inverse frequency, as the config specifies. Both losses take the same alpha so the
    # focal-versus-BCE comparison isolates the focusing term.
    alpha = (n_negative / n_positive) if n_positive else 1.0

    typology_targets = torch.tensor([int(g.y_typ.item()) for g in train_graphs])
    class_weights = inverse_frequency_weights(typology_targets, len(TYPOLOGY_CLASSES)).to(device)

    criterion = EncoderLoss(
        build_binary_loss(str(training.loss), gamma=float(training.focal_gamma), alpha=alpha),
        typology_weight=float(training.typology_weight),
        typology_class_weights=class_weights if bool(cfg.encoder.typology_head) else None,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=float(training.lr),
        weight_decay=float(training.weight_decay),
    )
    epochs = int(training.epochs)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=float(training.lr) * 0.01)

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_graphs,
        batch_size=int(training.batch_size),
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    pe_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    augment_pe = bool(training.lap_pe_sign_flip) and space.lap_pe_dim > 0

    best_auc_pr = -math.inf
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    patience = int(training.early_stop_patience)
    since_improvement = 0
    history: list[EpochRecord] = []
    started = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_started = time.time()
        totals = {"loss_total": 0.0, "loss_risk": 0.0, "loss_typology": 0.0}
        n_batches = 0

        for cpu_batch in loader:
            batch = cpu_batch.to(device)
            if augment_pe:
                batch.x = flip_lap_pe_signs(
                    batch.x.clone(), batch.batch, int(batch.num_graphs), space, pe_generator
                )
            out = model(batch)
            loss, components = criterion(
                out.risk_logits,
                batch.y,
                out.typology_logits,
                batch.y_typ,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.grad_clip))
            optimizer.step()
            for key in totals:
                totals[key] += components.get(key, 0.0)
            n_batches += 1

        scheduler.step()
        val = predict(model, val_graphs, device, batch_size=int(training.eval_batch_size))
        val_metrics = binary_metrics(val.scores, val.targets)

        history.append(
            EpochRecord(
                epoch=epoch,
                train_loss=totals["loss_total"] / max(n_batches, 1),
                train_risk_loss=totals["loss_risk"] / max(n_batches, 1),
                train_typology_loss=totals["loss_typology"] / max(n_batches, 1),
                val_auc_pr=val_metrics.auc_pr,
                val_auc_roc=val_metrics.auc_roc,
                learning_rate=float(scheduler.get_last_lr()[0]),
                seconds=time.time() - epoch_started,
            )
        )
        if log is not None and (epoch % 5 == 0 or epoch == epochs - 1):
            log.info(
                "  epoch %3d  loss %.4f  val_auc_pr %.4f  val_auc_roc %.4f",
                epoch,
                history[-1].train_loss,
                val_metrics.auc_pr,
                val_metrics.auc_roc,
            )

        # Selection on validation AUC-PR, and on nothing else.
        if val_metrics.auc_pr > best_auc_pr:
            best_auc_pr = val_metrics.auc_pr
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since_improvement = 0
        else:
            since_improvement += 1
            if since_improvement >= patience:
                if log is not None:
                    log.info("  early stop at epoch %d (patience %d)", epoch, patience)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path: str | None = None
    if checkpoint_dir is not None and best_state is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = str(checkpoint_dir / f"{cfg.encoder.arch}_seed{seed}.pt")
        torch.save(
            {
                "arm": str(cfg.encoder.arch),
                "seed": seed,
                "best_epoch": best_epoch,
                "best_val_auc_pr": best_auc_pr,
                "state_dict": best_state,
                "feature_space": space.to_dict(),
                "encoder_config": _plain(cfg.encoder),
                # Recorded so `resume` can refuse a checkpoint trained under a different
                # regime. A two-epoch smoke checkpoint left in the checkpoint directory
                # is otherwise indistinguishable from a converged one, and resuming it
                # would put a smoke-run number into a results table. This happened.
                "training_config": _plain(cfg.training),
            },
            checkpoint_path,
        )

    predictions = {
        name: predict(model, graphs, device, batch_size=int(training.eval_batch_size))
        for name, graphs in splits.items()
    }
    metrics = {name: binary_metrics(p.scores, p.targets) for name, p in predictions.items()}
    typology = {
        name: typology_metrics(p.typology_predictions, p.typology_targets, TYPOLOGY_CLASSES)
        for name, p in predictions.items()
    }

    result = TrainingResult(
        arm=str(cfg.encoder.arch),
        seed=seed,
        best_epoch=best_epoch,
        best_val_auc_pr=best_auc_pr,
        epochs_run=len(history),
        n_parameters=n_parameters,
        history=history,
        metrics=metrics,
        typology=typology,
        seconds=time.time() - started,
        checkpoint=checkpoint_path,
    )
    return model, result, predictions


def _plain(node: Any) -> Any:
    """Return an OmegaConf node as a plain container, or the value unchanged."""
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(node, resolve=True)
    except Exception:  # pragma: no cover - only when omegaconf is absent
        return node


def save_result(result: TrainingResult, path: str | Path) -> Path:
    """Write a run summary atomically.

    Args:
        result: The run to write.
        path: Destination file.

    Returns:
        The path written.
    """
    return write_json(path, result.to_dict())
