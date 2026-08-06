"""Loss functions for an imbalanced binary task with an auxiliary multiclass head.

Both candidate binary losses are implemented, because "we used focal loss" is not a
result unless the alternative was run. They attack the imbalance differently:

- **Weighted BCE** reweights by class frequency. Every negative counts the same, so the
  gradient is dominated by the tens of thousands of easy negatives that the model got
  right in the first epoch and will keep getting right.
- **Focal loss** reweights by *difficulty*, scaling each example's loss by
  ``(1 - p_t)^gamma``. A confidently-correct easy negative contributes almost nothing,
  so the gradient concentrates on the boundary. On this corpus the boundary is the
  hard-negative population — 25.8% of cases are licit neighbourhoods mined for
  structural resemblance to a laundering motif (D-024) — which is exactly where the
  reweighting should be spent.

Both take the same ``alpha`` class weight, so the comparison isolates the focusing term
rather than confounding it with a different class balance.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as nnf

#: Sentinel target for a case with no typology ground truth. Matches the value
#: ``dataset.typology_index`` returns and is passed to ``cross_entropy`` as
#: ``ignore_index``, so such a case trains the binary head and abstains from the
#: auxiliary one rather than being dropped from the batch.
TYPOLOGY_IGNORE_INDEX = -1


def inverse_frequency_weights(labels: Tensor, n_classes: int) -> Tensor:
    """Return inverse-frequency class weights, normalised to mean one.

    Normalising to mean one keeps the loss on the same scale as an unweighted one, so a
    weighted and an unweighted run are comparable without rescaling the learning rate.

    Args:
        labels: Integer class labels. Values below zero are ignored, so the typology
            sentinel does not distort the counts.
        n_classes: Number of classes.

    Returns:
        A ``[n_classes]`` weight vector. A class with no examples gets weight zero
        rather than an infinite one.
    """
    counts = torch.zeros(n_classes, dtype=torch.float)
    valid = labels[labels >= 0]
    if valid.numel():
        counts.scatter_add_(0, valid.long(), torch.ones_like(valid, dtype=torch.float))
    weights = torch.where(counts > 0, 1.0 / counts.clamp(min=1.0), torch.zeros_like(counts))
    present = weights[counts > 0]
    if present.numel():
        weights = weights / present.mean()
    return weights


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al., 2017) on logits.

    Attributes:
        gamma: Focusing exponent. 0 recovers weighted BCE exactly, which is what makes
            the two comparable.
        alpha: Weight on the positive class. Under ``inverse_freq`` this is
            ``n_negative / n_positive`` normalised.
    """

    def __init__(self, *, gamma: float = 2.0, alpha: float = 1.0) -> None:
        """Build the loss.

        Args:
            gamma: Focusing exponent.
            alpha: Positive-class weight.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute the mean focal loss over a batch.

        Args:
            logits: ``[B]`` or ``[B, 1]`` raw scores.
            targets: ``[B]`` binary targets in {0, 1}.

        Returns:
            A scalar loss.
        """
        logits = logits.reshape(-1)
        targets = targets.reshape(-1).float()
        bce = nnf.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        # p_t: the probability the model assigned to the *true* class.
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - targets)
        return (alpha_t * (1 - p_t).pow(self.gamma) * bce).mean()


class WeightedBCELoss(nn.Module):
    """Frequency-weighted binary cross entropy on logits.

    Attributes:
        alpha: Weight on the positive class.
    """

    def __init__(self, *, alpha: float = 1.0) -> None:
        """Build the loss.

        Args:
            alpha: Positive-class weight.
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute the mean weighted BCE over a batch.

        Args:
            logits: ``[B]`` or ``[B, 1]`` raw scores.
            targets: ``[B]`` binary targets in {0, 1}.

        Returns:
            A scalar loss.
        """
        logits = logits.reshape(-1)
        targets = targets.reshape(-1).float()
        weight = self.alpha * targets + (1 - targets)
        return nnf.binary_cross_entropy_with_logits(
            logits, targets, weight=weight, reduction="mean"
        )


def build_binary_loss(name: str, *, gamma: float, alpha: float) -> nn.Module:
    """Construct the binary loss named in the config.

    Args:
        name: ``"focal"`` or ``"weighted_bce"``.
        gamma: Focusing exponent, used by focal only.
        alpha: Positive-class weight.

    Returns:
        The loss module.

    Raises:
        ValueError: If ``name`` is neither of the two. There is deliberately no default:
            which loss produced a published number is not something to infer.
    """
    if name == "focal":
        return FocalLoss(gamma=gamma, alpha=alpha)
    if name == "weighted_bce":
        return WeightedBCELoss(alpha=alpha)
    raise ValueError(f"unknown binary loss {name!r}; expected 'focal' or 'weighted_bce'")


class EncoderLoss(nn.Module):
    """The composite objective: binary risk plus a weighted auxiliary typology term.

    Attributes:
        typology_weight: Multiplier on the typology cross entropy. At 0 the auxiliary
            head is still built and still scored, but contributes no gradient — which is
            how the "does the auxiliary head help?" ablation is run without changing the
            parameter count.
    """

    def __init__(
        self,
        binary: nn.Module,
        *,
        typology_weight: float = 0.3,
        typology_class_weights: Tensor | None = None,
    ) -> None:
        """Build the composite loss.

        Args:
            binary: The binary loss module.
            typology_weight: Multiplier on the typology term.
            typology_class_weights: Per-class weights for the typology cross entropy.
                Without them the head collapses onto ``unclassified``, which is 93.8% of
                the fact records.
        """
        super().__init__()
        self.binary = binary
        self.typology_weight = typology_weight
        self.register_buffer(
            "typology_class_weights",
            typology_class_weights if typology_class_weights is not None else torch.empty(0),
        )

    def forward(
        self,
        risk_logits: Tensor,
        risk_targets: Tensor,
        typology_logits: Tensor | None,
        typology_targets: Tensor | None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute the composite loss and its components.

        Args:
            risk_logits: ``[B, 1]``.
            risk_targets: ``[B]`` binary targets.
            typology_logits: ``[B, C]`` or None.
            typology_targets: ``[B]`` class targets, ``-1`` where absent, or None.

        Returns:
            ``(total, components)`` where ``components`` holds the two terms as floats
            for logging.
        """
        risk = self.binary(risk_logits, risk_targets)
        components = {"loss_risk": float(risk.detach())}
        total = risk

        if typology_logits is not None and typology_targets is not None:
            weights = self.typology_class_weights
            typology = nnf.cross_entropy(
                typology_logits,
                typology_targets.reshape(-1).long(),
                weight=weights if weights.numel() else None,
                ignore_index=TYPOLOGY_IGNORE_INDEX,
            )
            # Every case in the batch can lack typology ground truth, and cross_entropy
            # returns NaN rather than zero when everything is ignored.
            if torch.isfinite(typology):
                components["loss_typology"] = float(typology.detach())
                total = total + self.typology_weight * typology
            else:
                components["loss_typology"] = 0.0

        components["loss_total"] = float(total.detach())
        return total, components
