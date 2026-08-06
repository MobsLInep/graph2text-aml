"""The shuffled-graph control. This is the arm the project's central claim rests on.

**Why this file exists.** A model given graph tokens and a text prompt will produce a
faithful-looking narrative whether or not it reads the graph tokens, because the text
prompt already carries the serialised facts. Every metric will look good. Faithfulness
will be high. The soft tokens will receive attention, because attention mass is paid to
every position in a sequence and a diffuse read of sixteen positions is indistinguishable
from an informative one at the level of a single scalar. **None of that is evidence that
the graph encoder contributed anything.**

The only thing that is evidence is a control that keeps everything identical except the
correspondence between the graph and the narrative. That is this module. A1 is S1 with
each case's narrative paired against a *different case's* graph tokens: same encoder, same
projector, same parameter count, same token count, same optimiser, same seed, same
curriculum, same number of gradient steps. If S1 does not beat A1, the fusion layer is
decoration and the paper's contribution is the dataset and the evaluation framework.

**Derangement, not permutation.** ``torch.randperm`` leaves fixed points — at batch size 2
it returns the identity half the time, and a "control" that hands a quarter of its cases
their own graph is not a control, it is a weakened treatment arm that biases the
comparison *towards* the null being rejected. :func:`derange` guarantees no case keeps its
own tokens.

**Batch size one.** ``per_device_batch_size`` is 2, but the last batch of an epoch can be
1, and there is no within-batch derangement of a single element. Falling back to "leave it
alone" would quietly feed the control its own graph on those steps. Instead the control
keeps a small ring buffer of graph tokens from previous batches and draws from it, so a
singleton batch is still paired with a foreign case. If the buffer is empty — the very
first step — the batch is skipped rather than silently un-shuffled, and the count is
reported in :attr:`ShuffledGraphFusion.stats`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from g2t_aml.models.fusion.base import FusionOutput

__all__ = ["SHUFFLE_MODES", "ShuffleMode", "ShuffleStats", "ShuffledGraphFusion", "derange"]

#: The control modes. ``across_batch`` is the one the project's claim rests on.
ShuffleMode = Literal["across_batch", "within_case", "noise"]

SHUFFLE_MODES: tuple[str, ...] = ("across_batch", "within_case", "noise")

#: Smallest batch that can be deranged within itself; below it the ring buffer is used.
_MIN_DERANGEABLE = 2

#: How many previous batches' graph tokens to retain for the singleton-batch fallback.
_BUFFER_BATCHES = 8


@dataclass
class ShuffleStats:
    """What the control actually did, so the paper can state it rather than assume it.

    Attributes:
        n_batches: Batches passed through the control.
        n_cases: Cases shuffled.
        n_fixed_points: Cases that kept their own graph tokens. **Must be zero.** Any
            other value means the control leaked treatment into itself, and the S1-vs-A1
            comparison is biased by exactly that fraction.
        n_unshuffled_batches: Batches that could not be shuffled at all — only possible
            on the first singleton batch, before the buffer has filled.
    """

    n_batches: int = 0
    n_cases: int = 0
    n_fixed_points: int = 0
    n_unshuffled_batches: int = 0

    def to_dict(self) -> dict[str, int | float]:
        """Return the statistics as a loggable mapping.

        Returns:
            The counts plus the fixed-point rate, which is the number that has to be zero.
        """
        return {
            "n_batches": self.n_batches,
            "n_cases": self.n_cases,
            "n_fixed_points": self.n_fixed_points,
            "n_unshuffled_batches": self.n_unshuffled_batches,
            "fixed_point_rate": self.n_fixed_points / self.n_cases if self.n_cases else 0.0,
        }


def derange(n: int, *, generator: torch.Generator | None = None, device: str = "cpu") -> Tensor:
    """Return a permutation of ``0..n-1`` with no fixed point.

    Sampled by rejection, which is the right algorithm here despite looking naive: the
    proportion of permutations that are derangements converges to ``1/e`` from ``n = 2``
    upward, so the expected number of draws is under three for every batch size this
    project uses, and rejection sampling is uniform over derangements where the common
    "swap the fixed points afterwards" fix is not.

    Args:
        n: Number of elements. Must be at least 2 — a single element has no derangement.
        generator: Torch generator, for a reproducible control.
        device: Device to build the index on.

    Returns:
        ``[n]`` index tensor where ``result[i] != i`` for every ``i``.

    Raises:
        ValueError: If ``n < 2``, because no derangement exists. Callers that can see a
            singleton batch must handle it rather than asking for the impossible; see
            :class:`ShuffledGraphFusion`.
    """
    if n < _MIN_DERANGEABLE:
        raise ValueError(
            f"no derangement exists for n={n}; a singleton batch cannot be shuffled "
            "against itself and needs the ring-buffer fallback"
        )
    index = torch.arange(n, device=device)
    while True:
        candidate = torch.randperm(n, generator=generator, device=device)
        if bool((candidate != index).all()):
            return candidate


class ShuffledGraphFusion(nn.Module):
    """Wraps a fusion variant and breaks the graph-to-narrative correspondence.

    Wrapping rather than reimplementing is deliberate. The control must differ from the
    treatment in one place only, and the surest way to guarantee that is for both arms to
    run *the same fusion object* with the same code path, differing solely in whether this
    wrapper permuted the input first.

    The shuffle is applied to the **pooled tokens, before projection**, so the projector
    sees an input drawn from exactly the same distribution in both arms. Shuffling after
    projection would additionally change what the projector is trained on, confounding the
    comparison with a second difference.
    """

    def __init__(
        self,
        fusion: nn.Module,
        *,
        mode: ShuffleMode = "across_batch",
        seed: int | None = None,
    ) -> None:
        """Wrap a fusion variant in the control.

        Args:
            fusion: The variant to wrap, normally the same
                :class:`~g2t_aml.models.fusion.variants.PrefixFusion` configuration the
                treatment arm uses.
            mode: Which control. ``across_batch`` pairs each narrative with another
                case's graph and is the control the central claim rests on.
                ``within_case`` only permutes token order within a case, which tests
                order-sensitivity and is a much weaker null — a model reading the tokens
                as an unordered set passes it while ignoring the graph entirely.
                ``noise`` replaces the tokens with moment-matched Gaussian noise, a
                stronger null than either, kept because it separates "the graph is
                unused" from "any sixteen plausible vectors would do".
            seed: Seed for the control's own generator, so A1 is reproducible
                independently of the model's stream of random numbers.

        Raises:
            ValueError: If ``mode`` is not a known control.
        """
        super().__init__()
        if mode not in SHUFFLE_MODES:
            raise ValueError(f"unknown shuffle mode {mode!r}; expected one of {SHUFFLE_MODES}")
        self.fusion = fusion
        self.mode = mode
        self.stats = ShuffleStats()
        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)
        self._buffer: list[Tensor] = []

    @property
    def n_tokens(self) -> int:
        """Return the soft-token count, unchanged by the control.

        Returns:
            The wrapped variant's token count.
        """
        return int(self.fusion.n_tokens)

    def gate_value(self) -> Tensor | None:
        """Return the wrapped variant's gate, for logging.

        The A1 gate trajectory is worth as much as S1's: a gate that stays open on
        shuffled tokens means the model is not distinguishing signal from noise, which is
        a different diagnosis from a gate that closes on both.

        Returns:
            ``[n_tokens]`` gate values, or None on an ungated variant.
        """
        gate = getattr(self.fusion, "gate_value", None)
        return gate() if callable(gate) else None

    def _shuffled(self, pooled_tokens: Tensor) -> Tensor:
        """Break the correspondence between cases and their graph tokens.

        Args:
            pooled_tokens: ``[B, k, graph_dim]``.

        Returns:
            Tokens of the same shape, with no case retaining its own.
        """
        batch = pooled_tokens.size(0)

        if self.mode == "noise":
            mean = pooled_tokens.mean()
            std = pooled_tokens.std().clamp_min(1e-6)
            noise = torch.randn(
                pooled_tokens.shape, generator=self._generator, dtype=pooled_tokens.dtype
            )
            return noise.to(pooled_tokens.device) * std + mean

        if self.mode == "within_case":
            order = torch.stack(
                [derange(pooled_tokens.size(1), generator=self._generator) for _ in range(batch)]
            ).to(pooled_tokens.device)
            return torch.gather(
                pooled_tokens, 1, order.unsqueeze(-1).expand(-1, -1, pooled_tokens.size(-1))
            )

        if batch >= _MIN_DERANGEABLE:
            order = derange(batch, generator=self._generator).to(pooled_tokens.device)
            return pooled_tokens[order]

        # Singleton batch: draw a foreign case from the ring buffer rather than leaving
        # the one case paired with its own graph.
        if not self._buffer:
            self.stats.n_unshuffled_batches += 1
            return pooled_tokens
        pick = int(torch.randint(len(self._buffer), (1,), generator=self._generator))
        return self._buffer[pick][:1].to(pooled_tokens.device)

    def forward(self, pooled_tokens: Tensor) -> FusionOutput:
        """Shuffle the graph tokens, then run the wrapped fusion variant.

        Args:
            pooled_tokens: ``[B, k, graph_dim]`` from the encoder.

        Returns:
            The wrapped variant's output, computed from tokens belonging to other cases.
        """
        shuffled = self._shuffled(pooled_tokens)

        self.stats.n_batches += 1
        self.stats.n_cases += pooled_tokens.size(0)
        if self.mode == "across_batch" and pooled_tokens.size(0) >= _MIN_DERANGEABLE:
            same = (shuffled == pooled_tokens).flatten(1).all(dim=1)
            self.stats.n_fixed_points += int(same.sum())

        if self.mode == "across_batch":
            self._buffer.append(pooled_tokens.detach().cpu())
            if len(self._buffer) > _BUFFER_BATCHES:
                self._buffer.pop(0)

        return self.fusion(shuffled)
