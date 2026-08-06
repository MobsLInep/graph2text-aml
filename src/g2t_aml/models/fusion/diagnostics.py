"""Measuring whether the language model actually reads the soft tokens.

Three quantities, logged every N steps by Phase 9's callbacks and reported in the paper.

**Attention mass, against its uniform baseline.** "The model paid 12% of its attention to
the graph tokens" is not a finding on its own — if the graph occupies 16 of 130 positions,
uniform attention *is* 12%, and the number says only that the tokens exist. What matters
is the ratio of observed mass to the uniform baseline, exactly as Phase 7 reported pooling
attention against its own uniform baseline (85.1% against 47.8%, lift 1.78). A lift near
1.0 means the model is looking straight through the graph.

**Gate value.** F2's per-token gate. A trajectory that decays towards zero is the model
learning to close the graph channel, and it is the earliest signal available that S1 will
not beat A1.

**Soft-token norm distribution.** Distinguishes "the gate is open but the projector emits
nothing" from "the gate is closed". Both produce a graph-free model and they need
different fixes, so they are logged separately.

Every function here takes tensors the caller already has and returns plain floats, because
these get written into a JSONL trace and plotted; none of them holds a reference to a
model or a batch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor

__all__ = ["AttentionMass", "FusionDiagnostics", "gate_summary", "soft_token_attention_mass"]


@dataclass(frozen=True)
class AttentionMass:
    """Attention paid to the soft tokens, and what to compare it against.

    Attributes:
        mass: Fraction of attention mass on soft-token positions, averaged over the
            layers, heads and query positions requested.
        uniform_baseline: ``n_soft / n_keys`` — the mass a model attending uniformly would
            pay. **The interpretation of** :attr:`mass` **is undefined without this.**
        lift: ``mass / uniform_baseline``. 1.0 means indistinguishable from uniform.
        n_soft_tokens: How many positions were counted as soft.
        n_key_positions: How many positions were attendable.
    """

    mass: float
    uniform_baseline: float
    lift: float
    n_soft_tokens: int
    n_key_positions: int

    def to_dict(self) -> dict[str, float | int]:
        """Return the measurement as a loggable mapping.

        Returns:
            The fields, ready for a JSONL trace.
        """
        return asdict(self)


@dataclass(frozen=True)
class FusionDiagnostics:
    """The full per-step diagnostic record for one fusion layer.

    Attributes:
        gate_mean: Mean gate value across soft tokens, or None when ungated.
        gate_min: Smallest per-token gate, or None. A single collapsed token is invisible
            in the mean and visible here.
        gate_max: Largest per-token gate, or None.
        soft_token_norm_mean: Mean L2 norm of the emitted soft tokens.
        soft_token_norm_std: Spread of those norms across tokens and batch.
        pre_gate_rms: RMS of the projected tokens before gating, to be read against the
            language model's own embedding RMS.
        attention: Attention mass on the soft tokens, when it was measured this step.
    """

    gate_mean: float | None
    gate_min: float | None
    gate_max: float | None
    soft_token_norm_mean: float
    soft_token_norm_std: float
    pre_gate_rms: float | None
    attention: AttentionMass | None = None

    def to_dict(self) -> dict[str, object]:
        """Return the record as a flat loggable mapping.

        Returns:
            The fields, with the attention block flattened under an ``attention_`` prefix
            so a metrics backend that cannot nest still receives every number.
        """
        out: dict[str, object] = {
            "gate_mean": self.gate_mean,
            "gate_min": self.gate_min,
            "gate_max": self.gate_max,
            "soft_token_norm_mean": self.soft_token_norm_mean,
            "soft_token_norm_std": self.soft_token_norm_std,
            "pre_gate_rms": self.pre_gate_rms,
        }
        if self.attention is not None:
            out.update({f"attention_{k}": v for k, v in self.attention.to_dict().items()})
        return out


def soft_token_attention_mass(
    attentions: tuple[Tensor, ...] | list[Tensor],
    *,
    soft_start: int,
    n_soft: int,
    query_start: int | None = None,
    layers: tuple[int, ...] | None = None,
) -> AttentionMass:
    """Measure how much attention the generated positions pay to the soft tokens.

    Args:
        attentions: The per-layer attention tensors a transformer returns under
            ``output_attentions=True``, each ``[B, heads, queries, keys]``.
        soft_start: Index of the first soft token in the sequence.
        n_soft: How many soft tokens there are.
        query_start: Only count attention from queries at or after this position, which
            is how the measurement is restricted to the *completion* rather than the
            prompt. None counts every query. Passing the completion start is strongly
            preferred: prompt positions adjacent to the soft tokens attend to them for
            positional reasons that say nothing about whether the graph informed the
            narrative.
        layers: Which layers to average over. None uses all of them. Reporting a single
            late layer is common and misleading — the mass varies by an order of
            magnitude across depth — so the default is the average and the caller must
            opt in to a subset.

    Returns:
        The measurement, with its uniform baseline and lift.

    Raises:
        ValueError: If no attention tensors were given, or the soft-token span falls
            outside the key axis.
    """
    if not attentions:
        raise ValueError(
            "no attention tensors given; the forward pass needs output_attentions=True"
        )

    chosen = attentions if layers is None else [attentions[i] for i in layers]
    n_keys = int(chosen[0].size(-1))
    if soft_start < 0 or soft_start + n_soft > n_keys:
        raise ValueError(
            f"soft tokens span [{soft_start}, {soft_start + n_soft}) but the attention "
            f"matrix has only {n_keys} key positions"
        )

    totals: list[Tensor] = []
    for layer in chosen:
        queries = layer if query_start is None else layer[:, :, query_start:, :]
        if queries.size(2) == 0:
            continue
        # Rows of an attention matrix already sum to 1, so the mass on a span is its sum.
        totals.append(queries[..., soft_start : soft_start + n_soft].sum(dim=-1).mean())

    if not totals:
        raise ValueError(f"query_start={query_start} left no query positions to measure")

    mass = float(torch.stack(totals).mean())
    baseline = n_soft / n_keys
    return AttentionMass(
        mass=mass,
        uniform_baseline=baseline,
        lift=mass / baseline if baseline else 0.0,
        n_soft_tokens=n_soft,
        n_key_positions=n_keys,
    )


def gate_summary(gate: Tensor | None) -> tuple[float | None, float | None, float | None]:
    """Reduce a gate tensor to the three numbers worth plotting.

    Args:
        gate: ``[n_tokens]`` or ``[B, n_tokens]`` gate values, or None on an ungated
            variant.

    Returns:
        ``(mean, min, max)``, each None when there is no gate. The minimum is reported
        because a mean near 0.5 is equally consistent with every token half-open and with
        half the tokens fully closed, and those are different models.
    """
    if gate is None:
        return None, None, None
    values = gate.detach().float().flatten()
    return float(values.mean()), float(values.min()), float(values.max())
