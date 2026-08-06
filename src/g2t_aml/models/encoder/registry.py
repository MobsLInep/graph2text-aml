"""Building an arm by name, from a config and a fitted feature space.

The registry exists so that a sweep over arms is a sweep over one string. Every arm is
constructed through :func:`build_encoder`, which reads its input widths from the feature
space rather than from the config — a model whose ``in_dim`` disagrees with the cache it
is fed would fail at the first batch, and a model whose ``in_dim`` was *configured*
correctly by hand would fail silently the day the feature spec changed.
"""

from __future__ import annotations

from typing import Any

from g2t_aml.models.encoder.arms import (
    GATv2Encoder,
    GCNEncoder,
    GINEncoder,
    GraphSAGEEncoder,
    GraphTransformerEncoder,
    MLPEncoder,
)
from g2t_aml.models.encoder.base import BaseEncoder
from g2t_aml.models.encoder.dataset import TYPOLOGY_CLASSES
from g2t_aml.models.encoder.features import FeatureSpace

#: Arm name to class. The keys are what ``encoder.arch`` takes in a config and what the
#: results tables are keyed by.
ARMS: dict[str, type[BaseEncoder]] = {
    "gatv2": GATv2Encoder,
    "gin": GINEncoder,
    "sage": GraphSAGEEncoder,
    "gcn": GCNEncoder,
    "graph_transformer": GraphTransformerEncoder,
    "mlp": MLPEncoder,
}

#: Constructor keywords that only some arms accept. Passing ``heads`` to GIN would raise,
#: so the builder filters against this map rather than the caller remembering which arm
#: takes what.
_ARM_KWARGS: dict[str, tuple[str, ...]] = {
    "gatv2": ("heads", "concat_heads", "num_layers", "residual"),
    "gin": ("num_layers", "residual"),
    "sage": ("num_layers", "residual"),
    "gcn": ("num_layers", "residual"),
    "graph_transformer": ("heads", "num_layers", "residual"),
    "mlp": ("num_layers",),
}


class UnknownArmError(KeyError):
    """Raised when a config names an arm that does not exist."""


def build_encoder(
    cfg: Any,
    space: FeatureSpace,
    *,
    use_edge_features: bool | None = None,
) -> BaseEncoder:
    """Construct an encoder arm from a config and a fitted feature space.

    Args:
        cfg: The ``encoder`` config node. Reads ``arch``, ``hidden_dim``, ``edge_dim``,
            ``dropout``, ``n_pooled_tokens``, ``typology_head`` and whichever of
            ``layers`` / ``heads`` / ``concat_heads`` / ``residual`` the arm accepts.
        space: The fitted feature space, which supplies every input width.
        use_edge_features: Override the config's edge-feature switch. Used by the
            ablation, which builds the same arm twice.

    Returns:
        The constructed arm, on CPU and untrained.

    Raises:
        UnknownArmError: If ``cfg.arch`` is not in :data:`ARMS`.
    """
    arch = str(cfg.arch)
    if arch not in ARMS:
        raise UnknownArmError(f"unknown encoder arm {arch!r}; expected one of {sorted(ARMS)}")

    shared: dict[str, Any] = {
        "node_dim": space.node_dim,
        "edge_continuous_dim": space.edge_continuous_dim,
        "n_currencies": space.n_currencies,
        "n_formats": space.n_payment_formats,
        "hidden_dim": int(cfg.hidden_dim),
        "edge_dim": int(cfg.edge_dim),
        "n_typologies": len(TYPOLOGY_CLASSES) if bool(cfg.typology_head) else 0,
        "n_pooled_tokens": int(cfg.n_pooled_tokens),
        "dropout": float(cfg.dropout),
        "use_edge_features": (
            bool(cfg.use_edge_features) if use_edge_features is None else use_edge_features
        ),
    }

    available = {
        "num_layers": int(cfg.layers),
        "heads": int(cfg.heads),
        "concat_heads": bool(cfg.concat_heads),
        "residual": bool(cfg.residual),
    }
    specific = {k: v for k, v in available.items() if k in _ARM_KWARGS[arch]}
    return ARMS[arch](**shared, **specific)


def count_parameters(model: BaseEncoder) -> int:
    """Return the number of trainable parameters.

    Reported per arm because a capacity difference is an alternative explanation for a
    performance difference, and a comparison that does not publish it invites the
    question.

    Args:
        model: Any arm.

    Returns:
        Total trainable parameter count.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
