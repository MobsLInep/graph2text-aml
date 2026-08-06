"""Bronze: deterministic template narratives, faithful by construction.

The floor every learned system is measured against, the first baseline in the paper
(B1/B2), the warm-start data for epoch 1 of fine-tuning, and the input Silver rewrites
rather than inventing from scratch.
"""

from g2t_aml.corpus.bronze.renderer import (
    MAX_TOKENS,
    MIN_TOKENS,
    RENDERER_VERSION,
    RenderError,
    SubstrateViolation,
    VariantInapplicable,
    render_bronze,
    select_family,
    select_variant,
)
from g2t_aml.corpus.bronze.templates import FAMILIES, Family, Segment, Variant, family_for

__all__ = [
    "FAMILIES",
    "MAX_TOKENS",
    "MIN_TOKENS",
    "RENDERER_VERSION",
    "Family",
    "RenderError",
    "Segment",
    "SubstrateViolation",
    "Variant",
    "VariantInapplicable",
    "family_for",
    "render_bronze",
    "select_family",
    "select_variant",
]
