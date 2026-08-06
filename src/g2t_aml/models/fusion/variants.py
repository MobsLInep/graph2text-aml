"""The fusion variants: F1 without a gate, F2 with one.

Two variants and one control, differing in exactly one thing each, so that the ablation
reads cleanly:

===  =====================================================================
F1   Project and splice. The G-Retriever-style prefix: the soft tokens enter
     the sequence at full strength and the language model has no mechanism
     for turning them down. This is the published-baseline arm (B8).
F2   Project, normalise, and scale by a **learned per-token gate**. The model
     can attenuate a graph token it finds unhelpful. This is the primary
     arm (S1, S2).
A1   F2 with the graph tokens deranged across the batch — see
     :mod:`g2t_aml.models.fusion.control`.
===  =====================================================================

**The gate is a measurement instrument, not only a capacity.** F1 cannot tell you whether
the model used the graph, because its soft tokens are present at fixed strength whether
they help or not. F2's gate is a scalar per soft token that the model is free to drive to
zero, and it does so when the tokens are noise. That makes the gate trajectory an early,
cheap read on Gate 8: if the gate collapses on S1 the way it should on A1, the graph is
contributing nothing and there is no point waiting for the faithfulness curves to say so
fourteen hours later.

**Why the gate is bounded and not free.** An unbounded scalar can compensate for a
badly-scaled projector by growing, which hides exactly the scale problem
:func:`g2t_aml.models.fusion.base.embedding_rms` exists to prevent, and makes the logged
gate value uninterpretable — is 4.7 an enthusiastic gate or a projector emitting tokens at
a fifth of the right magnitude? A sigmoid keeps it in ``(0, 1)`` so that "the gate is at
0.6" means one thing, and the scale correction stays where it can be inspected.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from g2t_aml.models.fusion.base import (
    FUSION_DTYPE,
    FusionOutput,
    ProjectorSpec,
    build_projector,
)

__all__ = ["PrefixFusion", "build_fusion"]

#: Sigmoid pre-activation giving a gate of ~0.62 at initialisation. Deliberately not 0.5
#: (which a symmetric init would make suspiciously easy to sit at and hard to distinguish
#: from "untrained") and deliberately not near 1.0, which leaves the gate no headroom to
#: signal that it *wants* more graph and makes an increase unmeasurable.
_GATE_INIT = 0.5

#: Pooled tokens are always [B, k, graph_dim].
_POOLED_DIMS = 3


class PrefixFusion(nn.Module):
    """Projects pooled graph tokens into the LM embedding space, optionally gated.

    This is the class ``configs/fusion/prefix.yaml`` names as its ``_target_``.

    The whole module is held in fp32 (:data:`~g2t_aml.models.fusion.base.FUSION_DTYPE`)
    and :func:`~g2t_aml.models.fusion.base.assert_projector_is_fp32` is what keeps it
    there once PEFT has walked the tree.
    """

    def __init__(
        self,
        *,
        graph_dim: int,
        lm_dim: int,
        num_prefix_tokens: int,
        projector: str = "mlp",
        projector_hidden: int = 2048,
        dropout: float = 0.05,
        layer_norm: bool = True,
        gated: bool = True,
        n_output_tokens: int | None = None,
        target_rms: float | None = None,
    ) -> None:
        """Build the fusion layer.

        Args:
            graph_dim: The encoder arm's ``hidden_dim``, the pooled tokens' width.
            lm_dim: The language model's hidden size.
            num_prefix_tokens: ``k``, the encoder's pooled-token count. Must match
                ``encoder.n_pooled_tokens``; ``configs/fusion/prefix.yaml`` interpolates
                both from the encoder config so they cannot silently disagree.
            projector: One of ``linear``, ``mlp``, ``perceiver``.
            projector_hidden: Hidden width for ``mlp`` and the perceiver's FFN.
            dropout: Dropout on the projected tokens.
            layer_norm: Normalise projected tokens before scaling. Off only for the
                ablation that asks whether the normalisation is load-bearing.
            gated: F2 when True, F1 when False. **This is the only difference between the
                two variants**, which is what makes the comparison a measurement of the
                gate rather than of two separately-tuned models.
            n_output_tokens: Soft-token budget. Defaults to ``num_prefix_tokens``; only
                the perceiver projector may differ.
            target_rms: Scale the soft tokens to this RMS, normally the base model's
                input-embedding RMS. None leaves the natural scale.

        Raises:
            ValueError: If a non-perceiver projector is asked for an output token count
                different from its input count, which it cannot honour.
        """
        super().__init__()
        n_out = num_prefix_tokens if n_output_tokens is None else n_output_tokens
        if projector != "perceiver" and n_out != num_prefix_tokens:
            raise ValueError(
                f"projector {projector!r} maps tokens one-to-one, so n_output_tokens "
                f"({n_out}) must equal num_prefix_tokens ({num_prefix_tokens}); use the "
                "perceiver projector to resample to a different budget"
            )

        self.spec = ProjectorSpec(
            kind=projector,  # type: ignore[arg-type]
            graph_dim=graph_dim,
            lm_dim=lm_dim,
            n_input_tokens=num_prefix_tokens,
            n_output_tokens=n_out,
            hidden=projector_hidden,
            dropout=dropout,
            layer_norm=layer_norm,
            target_rms=target_rms,
        )
        self.projector = build_projector(self.spec)
        self.norm = nn.LayerNorm(lm_dim, dtype=FUSION_DTYPE) if layer_norm else None
        self.dropout = nn.Dropout(dropout)
        self.gate_logit = (
            nn.Parameter(torch.full((n_out,), _GATE_INIT, dtype=FUSION_DTYPE)) if gated else None
        )
        self.n_tokens = n_out
        self.lm_dim = lm_dim
        self.gated = gated
        self.target_rms = target_rms

    def set_target_rms(self, value: float) -> None:
        """Set the output scale after construction, once the base model is known.

        The base model is loaded long after the fusion layer is configured — it is 8B
        parameters and 4-bit quantised, and building it to read one statistic would be
        absurd — so Phase 9 constructs the fusion layer, loads the LM, measures its
        embedding RMS and calls this.

        Args:
            value: The target RMS, normally from
                :func:`~g2t_aml.models.fusion.base.embedding_rms`.

        Raises:
            ValueError: If ``value`` is not strictly positive.
        """
        if not value > 0:
            raise ValueError(f"target_rms must be positive, got {value}")
        self.target_rms = value

    def gate_value(self) -> Tensor | None:
        """Return the current per-token gate, for logging without a forward pass.

        Returns:
            ``[n_tokens]`` gate values in ``(0, 1)``, or None on an ungated variant.
        """
        if self.gate_logit is None:
            return None
        return torch.sigmoid(self.gate_logit.detach())

    def forward(self, pooled_tokens: Tensor) -> FusionOutput:
        """Project pooled graph tokens into the language model's embedding space.

        Args:
            pooled_tokens: ``[B, k, graph_dim]`` from the encoder. Cast to fp32 on entry
                so that an encoder running in bf16 cannot drag the projector's gradient
                down with it.

        Returns:
            The soft tokens and their diagnostics.

        Raises:
            ValueError: If the input is not three-dimensional, or its token count or width
                disagrees with what this layer was built for. Both are silent-corruption
                failures otherwise: a width mismatch broadcasts, and a token-count
                mismatch shifts every position the loss mask was computed for.
        """
        if pooled_tokens.dim() != _POOLED_DIMS:
            raise ValueError(
                f"expected [B, k, graph_dim] pooled tokens, got shape {tuple(pooled_tokens.shape)}"
            )
        _, k, width = pooled_tokens.shape
        if k != self.spec.n_input_tokens or width != self.spec.graph_dim:
            raise ValueError(
                f"pooled tokens are [*, {k}, {width}] but this fusion layer was built for "
                f"[*, {self.spec.n_input_tokens}, {self.spec.graph_dim}]; the encoder and "
                "fusion configs disagree"
            )

        projected = self.projector(pooled_tokens.to(FUSION_DTYPE))
        pre_gate_rms = projected.detach().pow(2).mean().sqrt()

        if self.norm is not None:
            projected = self.norm(projected)
        if self.target_rms is not None:
            current = projected.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-8).sqrt()
            projected = projected * (self.target_rms / current)
        projected = self.dropout(projected)

        gate: Tensor | None = None
        if self.gate_logit is not None:
            gate = torch.sigmoid(self.gate_logit).unsqueeze(0)
            projected = projected * gate.unsqueeze(-1)
            gate = gate.expand(projected.size(0), -1)

        return FusionOutput(
            soft_tokens=projected,
            gate=gate,
            pre_gate_rms=pre_gate_rms,
            token_norms=projected.detach().norm(dim=-1),
        )


def build_fusion(cfg: Any, *, graph_dim: int | None = None, lm_dim: int | None = None) -> Any:
    """Construct a fusion variant from a config node.

    Args:
        cfg: The ``fusion`` config node. Reads ``name``, ``projector``,
            ``projector_hidden``, ``dropout``, ``layer_norm``, ``num_prefix_tokens``,
            ``graph_dim``, ``lm_dim``, and optionally ``gated``, ``n_output_tokens``,
            ``shuffle`` and ``shuffle_mode``.
        graph_dim: Override the config's graph width, normally with the loaded encoder's
            actual ``hidden_dim`` so a checkpoint and a config cannot disagree.
        lm_dim: Override the config's LM width with the loaded model's actual hidden size.

    Returns:
        The fusion module. Wrapped in
        :class:`~g2t_aml.models.fusion.control.ShuffledGraphFusion` when ``cfg.shuffle``
        is set, which is how the A1 arm is configured.

    Raises:
        ValueError: If ``cfg.name`` is not a known variant.
    """
    from g2t_aml.models.fusion.control import ShuffledGraphFusion

    name = str(getattr(cfg, "name", "prefix"))
    if name not in {"prefix", "f1", "f2"}:
        raise ValueError(f"unknown fusion variant {name!r}; expected prefix, f1 or f2")

    # `f1` and `f2` are the paper's names for ungated and gated prefix fusion. `prefix`
    # reads its `gated` flag from the config, so an experiment can set it explicitly.
    gated = {"f1": False, "f2": True}.get(name, bool(getattr(cfg, "gated", True)))

    fusion = PrefixFusion(
        graph_dim=int(graph_dim if graph_dim is not None else cfg.graph_dim),
        lm_dim=int(lm_dim if lm_dim is not None else cfg.lm_dim),
        num_prefix_tokens=int(cfg.num_prefix_tokens),
        projector=str(getattr(cfg, "projector", "mlp")),
        projector_hidden=int(getattr(cfg, "projector_hidden", 2048)),
        dropout=float(getattr(cfg, "dropout", 0.05)),
        layer_norm=bool(getattr(cfg, "layer_norm", True)),
        gated=gated,
        n_output_tokens=(
            int(cfg.n_output_tokens) if getattr(cfg, "n_output_tokens", None) is not None else None
        ),
        target_rms=(
            float(cfg.target_rms) if getattr(cfg, "target_rms", None) is not None else None
        ),
    )
    if bool(getattr(cfg, "shuffle", False)):
        return ShuffledGraphFusion(
            fusion,
            mode=str(getattr(cfg, "shuffle_mode", "across_batch")),  # type: ignore[arg-type]
        )
    return fusion
