"""The fusion contract: what every variant produces, and the projectors they share.

Phase 8's job is one sentence long. The encoder emits ``[B, k, graph_dim]`` pooled graph
tokens; the language model consumes ``[B, T, lm_dim]`` embeddings; something has to carry
the first into the second. That something is the technical novelty of this project, so it
is built behind an interface with more than one implementation and a control, in the same
shape as Phase 7's six arms: a difference between two fusion variants must be a difference
in *how structure enters the embedding space* and not an accidental difference in width,
initialisation or normalisation.

**The projector trains in fp32 and is never quantised.** It is a randomly-initialised map
learning to land inside a specific, already-trained embedding distribution. Quantising it
to nf4 discretises the very parameters that have to move precisely for that to converge,
and the failure mode is not a crash — it is a run that trains for fourteen hours and
produces soft tokens the language model reads as noise. :func:`assert_projector_is_fp32`
exists so that this is checked rather than remembered, and Phase 9 calls it after model
construction.

**Scale matters more than it looks.** A freshly initialised linear map emits activations
whose RMS has nothing to do with the RMS of the token embeddings it is being spliced among.
If the soft tokens arrive an order of magnitude larger, the first attention layer attends
to nothing else and the gradient signal is dominated by getting the magnitude back down; an
order of magnitude smaller and they are ignored. :class:`ProjectorSpec` therefore carries
``target_rms``, and Phase 9 fills it from the base model's own input-embedding statistics
rather than from a constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import torch
from torch import Tensor, nn

__all__ = [
    "FUSION_DTYPE",
    "FusionOutput",
    "GraphFusion",
    "ProjectorKind",
    "ProjectorSpec",
    "assert_projector_is_fp32",
    "build_projector",
    "embedding_rms",
]

#: The dtype every fusion parameter is held in, regardless of the base model's dtype or
#: quantisation. See the module docstring; Phase 9 asserts this after construction.
FUSION_DTYPE: torch.dtype = torch.float32

#: The projector architectures. ``linear`` is the G-Retriever-style single map, ``mlp``
#: adds one hidden layer, ``perceiver`` resamples to a fixed token budget with learned
#: latents and is the only one whose output length differs from its input length.
ProjectorKind = Literal["linear", "mlp", "perceiver"]


@dataclass
class FusionOutput:
    """Soft tokens and the diagnostics Phase 9 logs every N steps.

    Attributes:
        soft_tokens: ``[B, n_tokens, lm_dim]`` embeddings to splice into the language
            model's input sequence. Always fp32; the caller casts to the LM's compute
            dtype at the splice point, so the fusion parameters themselves never see a
            reduced-precision gradient.
        gate: ``[B, n_tokens]`` per-token gate values in ``(0, 1)``, or None for a variant
            with no gate. **A gate trajectory that collapses to zero means the model
            learned to ignore the graph**, which is a Phase 9 acceptance criterion and the
            cheapest early warning that S1 will not beat A1.
        pre_gate_rms: Scalar RMS of the projected tokens before gating, for comparison
            against the language model's own embedding RMS. Drift here explains a run that
            trains but generates nothing graph-conditioned.
        token_norms: ``[B, n_tokens]`` per-token L2 norms after gating. A variant that
            drives a subset of tokens to zero is doing token selection, and this is where
            that shows up.
    """

    soft_tokens: Tensor
    gate: Tensor | None = None
    pre_gate_rms: Tensor | None = None
    token_norms: Tensor | None = None


@runtime_checkable
class GraphFusion(Protocol):
    """The interface every fusion variant and the shuffled control satisfy."""

    #: How many soft tokens the variant emits per case. Phase 9 needs this before it sees
    #: a batch, because it reserves exactly this many positions in the prompt and masks
    #: them out of the loss.
    n_tokens: int

    def forward(self, pooled_tokens: Tensor) -> FusionOutput:
        """Project pooled graph tokens into the language model's embedding space.

        Args:
            pooled_tokens: ``[B, k, graph_dim]`` from
                :attr:`g2t_aml.models.encoder.base.EncoderOutput.pooled_tokens`.

        Returns:
            The soft tokens and their diagnostics.
        """
        ...


@dataclass(frozen=True)
class ProjectorSpec:
    """Everything needed to build a projector, resolved from config.

    Attributes:
        kind: Which architecture.
        graph_dim: Input width — the encoder arm's ``hidden_dim``.
        lm_dim: Output width — the language model's hidden size.
        n_input_tokens: ``k``, the encoder's pooled-token count.
        n_output_tokens: Soft-token count. Equals ``n_input_tokens`` for ``linear`` and
            ``mlp``; the perceiver resampler may differ.
        hidden: Hidden width of the ``mlp`` projector, ignored otherwise.
        dropout: Dropout applied to the projected tokens.
        layer_norm: Whether to normalise the projected tokens before scaling.
        target_rms: The RMS to scale projected tokens to, normally the base model's input
            embedding RMS. None leaves the projector's natural scale alone.
    """

    kind: ProjectorKind
    graph_dim: int
    lm_dim: int
    n_input_tokens: int
    n_output_tokens: int
    hidden: int = 2048
    dropout: float = 0.05
    layer_norm: bool = True
    target_rms: float | None = None


class _LinearProjector(nn.Module):
    """A single affine map per token. The G-Retriever-style baseline."""

    def __init__(self, spec: ProjectorSpec) -> None:
        """Build the projector.

        Args:
            spec: The resolved specification.
        """
        super().__init__()
        self.project = nn.Linear(spec.graph_dim, spec.lm_dim)
        self.n_output_tokens = spec.n_input_tokens

    def forward(self, x: Tensor) -> Tensor:
        """Project each graph token independently.

        Args:
            x: ``[B, k, graph_dim]``.

        Returns:
            ``[B, k, lm_dim]``.
        """
        return self.project(x)


class _MLPProjector(nn.Module):
    """One hidden layer per token. The default, and what F2 uses."""

    def __init__(self, spec: ProjectorSpec) -> None:
        """Build the projector.

        Args:
            spec: The resolved specification.
        """
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(spec.graph_dim, spec.hidden),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.hidden, spec.lm_dim),
        )
        self.n_output_tokens = spec.n_input_tokens

    def forward(self, x: Tensor) -> Tensor:
        """Project each graph token independently through the hidden layer.

        Args:
            x: ``[B, k, graph_dim]``.

        Returns:
            ``[B, k, lm_dim]``.
        """
        return self.project(x)


class _PerceiverProjector(nn.Module):
    """Cross-attention resampling from ``k`` graph tokens to a fixed soft-token budget.

    The one variant whose output length is a free parameter. It exists because the pooled
    token count is an encoder decision (``k = 16``) and the soft-token budget is a prompt
    decision — at 2048 context every soft token is a token the narrative cannot use — and
    tying them together forces one to be chosen for the other's reasons.
    """

    def __init__(self, spec: ProjectorSpec) -> None:
        """Build the resampler.

        Args:
            spec: The resolved specification.
        """
        super().__init__()
        self.latents = nn.Parameter(torch.randn(spec.n_output_tokens, spec.lm_dim) * 0.02)
        self.to_kv = nn.Linear(spec.graph_dim, spec.lm_dim * 2)
        self.attend = nn.MultiheadAttention(spec.lm_dim, num_heads=8, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(spec.lm_dim, spec.hidden),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.hidden, spec.lm_dim),
        )
        self.n_output_tokens = spec.n_output_tokens

    def forward(self, x: Tensor) -> Tensor:
        """Resample the graph tokens onto the learned latents.

        Args:
            x: ``[B, k, graph_dim]``.

        Returns:
            ``[B, n_output_tokens, lm_dim]``.
        """
        keys, values = self.to_kv(x).chunk(2, dim=-1)
        queries = self.latents.unsqueeze(0).expand(x.size(0), -1, -1)
        attended, _ = self.attend(queries, keys, values, need_weights=False)
        return attended + self.ffn(attended)


_PROJECTORS: dict[str, type[nn.Module]] = {
    "linear": _LinearProjector,
    "mlp": _MLPProjector,
    "perceiver": _PerceiverProjector,
}


def build_projector(spec: ProjectorSpec) -> nn.Module:
    """Construct a projector and place it in fp32.

    Args:
        spec: The resolved specification.

    Returns:
        The projector, on CPU, in :data:`FUSION_DTYPE`.

    Raises:
        ValueError: If ``spec.kind`` is not a known projector.
    """
    if spec.kind not in _PROJECTORS:
        raise ValueError(f"unknown projector {spec.kind!r}; expected one of {sorted(_PROJECTORS)}")
    return _PROJECTORS[spec.kind](spec).to(FUSION_DTYPE)


def embedding_rms(embeddings: Tensor) -> float:
    """Measure the RMS of a language model's input embedding table.

    Phase 9 calls this once on the base model's embedding matrix and feeds the result to
    :attr:`ProjectorSpec.target_rms`, so the soft tokens are born at the scale of the
    tokens they will sit among rather than at whatever scale ``nn.Linear``'s default
    initialisation happens to produce.

    Args:
        embeddings: ``[vocab, lm_dim]`` embedding weights, any dtype.

    Returns:
        The root-mean-square of the per-token embedding norms, as a float.
    """
    with torch.no_grad():
        return float(embeddings.to(torch.float32).pow(2).mean().sqrt())


def assert_projector_is_fp32(module: nn.Module, *, name: str = "fusion") -> None:
    """Assert that every fusion parameter is fp32 and none of them is quantised.

    This is the assertion the Phase 9 brief asks for in place of a comment. It is checked
    after the base model is loaded and wrapped, because that is when the damage happens:
    ``prepare_model_for_kbit_training`` and ``PeftModel`` both walk the module tree casting
    and replacing layers, and a projector that was fp32 at construction can be fp16 or a
    ``bitsandbytes`` 4-bit layer by the time the first batch arrives.

    Args:
        module: The fusion module, or any module containing it.
        name: Name used in the error message.

    Raises:
        TypeError: If any floating-point parameter is not fp32, or if any submodule is a
            ``bitsandbytes`` quantised layer.
    """
    wrong = [
        (param_name, str(param.dtype))
        for param_name, param in module.named_parameters()
        if param.is_floating_point() and param.dtype is not FUSION_DTYPE
    ]
    if wrong:
        listed = ", ".join(f"{n} is {d}" for n, d in wrong)
        raise TypeError(
            f"{name} must train in {FUSION_DTYPE} and does not: {listed}. A quantised or "
            "half-precision projector is the documented way to spend fourteen GPU hours "
            "on a run that never converges; see models/fusion/base.py."
        )
    quantised = [
        child_name
        for child_name, child in module.named_modules()
        if type(child).__module__.startswith("bitsandbytes")
    ]
    if quantised:
        raise TypeError(
            f"{name} contains bitsandbytes layers ({', '.join(quantised)}); the projector "
            "must never be quantised."
        )
