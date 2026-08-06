"""Phase 8: how graph structure enters the language model's embedding space.

The encoder emits ``[B, k, graph_dim]``; the language model consumes ``[B, T, lm_dim]``.
This package is the map between them, and it is the project's technical contribution.

Two variants and three controls:

- :class:`~g2t_aml.models.fusion.variants.PrefixFusion` — F1 (ungated, the
  G-Retriever-style baseline) and F2 (gated, primary), selected by one flag so the
  comparison measures the gate and nothing else.
- :class:`~g2t_aml.models.fusion.control.ShuffledGraphFusion` — the A1 control, which
  pairs each narrative with another case's graph. **Read that module before changing
  anything here**; the project's central claim is a comparison against it.

Two rules run through the package. The projector is fp32 and never quantised
(:func:`~g2t_aml.models.fusion.base.assert_projector_is_fp32`), and the soft tokens are
scaled to the base model's own embedding RMS
(:func:`~g2t_aml.models.fusion.base.embedding_rms`) rather than to whatever an
``nn.Linear`` default produces.
"""

from g2t_aml.models.fusion.base import (
    FUSION_DTYPE,
    FusionOutput,
    GraphFusion,
    ProjectorKind,
    ProjectorSpec,
    assert_projector_is_fp32,
    build_projector,
    embedding_rms,
)
from g2t_aml.models.fusion.control import (
    SHUFFLE_MODES,
    ShuffledGraphFusion,
    ShuffleMode,
    ShuffleStats,
    derange,
)
from g2t_aml.models.fusion.diagnostics import (
    AttentionMass,
    FusionDiagnostics,
    gate_summary,
    soft_token_attention_mass,
)
from g2t_aml.models.fusion.variants import PrefixFusion, build_fusion

__all__ = [
    "FUSION_DTYPE",
    "SHUFFLE_MODES",
    "AttentionMass",
    "FusionDiagnostics",
    "FusionOutput",
    "GraphFusion",
    "PrefixFusion",
    "ProjectorKind",
    "ProjectorSpec",
    "ShuffleMode",
    "ShuffleStats",
    "ShuffledGraphFusion",
    "assert_projector_is_fp32",
    "build_fusion",
    "build_projector",
    "derange",
    "embedding_rms",
    "gate_summary",
    "soft_token_attention_mass",
]
