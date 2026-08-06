"""A tiny language model and tokeniser, so Phase 9 is testable without ``transformers``.

The ``llm`` extra is GPU-only and pulls ~4 GB of wheels; requiring it to test a loss mask
would put every Phase 9 test behind a GPU and, in practice, mean they were never run. The
same argument produced :class:`~g2t_aml.corpus.silver.api_client.ScriptedTeacher` in Phase
5, and the resolution is the same: the real dependency sits behind a narrow protocol
(:class:`~g2t_aml.models.generator.model.CausalLM`), and a stub satisfying it exercises
every code path the harness owns.

What these stubs are **not** is a model. They test wiring: shapes, masks, dtypes, gradient
flow, checkpoint round-trips, the guard's selection order. Whether Llama-3.1 can read a
graph token is not a question a 2-layer randomly-initialised transformer can answer, and
no test here claims otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

__all__ = ["StubCausalLM", "StubConfig", "StubOutput", "StubTokenizer"]


@dataclass
class StubConfig:
    """The two config fields the harness reads.

    Attributes:
        hidden_size: Model width.
        vocab_size: Vocabulary size.
    """

    hidden_size: int
    vocab_size: int


@dataclass
class StubOutput:
    """What a forward pass returns.

    Attributes:
        loss: Cross-entropy over unmasked positions, or None without labels.
        logits: ``[B, T, vocab]``.
        attentions: Per-layer attention, when requested.
    """

    loss: Tensor | None
    logits: Tensor
    attentions: tuple[Tensor, ...] | None = None


class StubTokenizer:
    """A deterministic whitespace tokeniser with a stable vocabulary.

    Deterministic across processes matters: ``hash()`` is salted per interpreter run, so a
    tokeniser built on it produces different ids on a rerun and makes a golden test flap.
    """

    def __init__(self, vocab_size: int = 512) -> None:
        """Build the tokeniser.

        Args:
            vocab_size: Vocabulary size. Ids are assigned modulo this, above the reserved
                specials.
        """
        self.vocab_size = vocab_size
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.soft_token_id = 3
        self._vocab: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Tokenise on whitespace, assigning stable ids in first-seen order.

        Args:
            text: The text.
            add_special_tokens: Ignored; the stub adds none.

        Returns:
            The token ids.
        """
        del add_special_tokens
        ids: list[int] = []
        for word in text.split():
            if word not in self._vocab:
                self._vocab[word] = 4 + len(self._vocab) % (self.vocab_size - 4)
            ids.append(self._vocab[word])
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        """Detokenise back to whitespace-joined words.

        Args:
            ids: The token ids.
            skip_special_tokens: Drop ids below 4.

        Returns:
            The text. Unknown ids render as ``<id>`` so a decoding bug is visible rather
            than silently producing empty text.
        """
        reverse = {v: k for k, v in self._vocab.items()}
        words = []
        for i in ids:
            if skip_special_tokens and i < 4:
                continue
            words.append(reverse.get(i, f"<{i}>"))
        return " ".join(words)


class StubCausalLM(nn.Module):
    """A two-layer transformer that satisfies :class:`CausalLM`.

    Small enough to train to zero loss on twenty examples in a hundred steps on CPU, which
    is what makes the overfit test a real test of the harness rather than a mock of one.
    """

    def __init__(
        self,
        *,
        hidden_size: int = 64,
        vocab_size: int = 512,
        n_layers: int = 2,
        max_positions: int = 1024,
    ) -> None:
        """Build the stub.

        Args:
            hidden_size: Model width.
            vocab_size: Vocabulary size.
            n_layers: How many attention layers.
            max_positions: Positional embedding table size.
        """
        super().__init__()
        self.config = StubConfig(hidden_size=hidden_size, vocab_size=vocab_size)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        # Learned positional embeddings. Without them self-attention is permutation
        # invariant and the model cannot memorise a *sequence* at all — which would make
        # the overfit test fail for a reason that has nothing to do with the harness, and
        # so unable to detect the wiring bugs it exists to catch.
        self.positions = nn.Embedding(max_positions, hidden_size)
        self.layers = nn.ModuleList(
            [
                nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
                for _ in range(n_layers)
            ]
        )
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Linear(hidden_size * 4, hidden_size),
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def get_input_embeddings(self) -> nn.Module:
        """Return the embedding module.

        Returns:
            The embedding layer.
        """
        return self.embed

    def forward(
        self,
        *,
        inputs_embeds: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> StubOutput:
        """Run a causal forward pass.

        Args:
            inputs_embeds: ``[B, T, hidden]``.
            attention_mask: ``[B, T]``, 0 at padded positions.
            labels: ``[B, T]``, ``-100`` where the loss is masked.
            output_attentions: Retain attention weights.
            **kwargs: Ignored, for interface parity with ``transformers``.

        Returns:
            The loss, logits and any retained attentions.
        """
        del kwargs
        length = inputs_embeds.size(1)
        positions = torch.arange(length, device=inputs_embeds.device)
        hidden = inputs_embeds + self.positions(positions).unsqueeze(0)
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
        pad = None
        if attention_mask is not None:
            pad = attention_mask == 0

        collected: list[Tensor] = []
        for layer, ffn in zip(self.layers, self.ffns, strict=True):
            attended, weights = layer(
                hidden,
                hidden,
                hidden,
                attn_mask=causal.to(hidden.device),
                key_padding_mask=pad,
                need_weights=output_attentions,
                average_attn_weights=False,
            )
            hidden = self.norm(hidden + attended)
            hidden = self.norm(hidden + ffn(hidden))
            if output_attentions and weights is not None:
                collected.append(weights)

        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            # Standard causal shift: position t predicts token t+1.
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return StubOutput(
            loss=loss,
            logits=logits,
            attentions=tuple(collected) if output_attentions else None,
        )
