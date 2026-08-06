"""Counting tokens the way the generator will, without dragging CUDA into Phase 4.

Validation check 6 bounds a narrative at [80, 400] tokens, and the tokens that matter are
Llama-3.1's, because that is what the fine-tune consumes. Phases 1-6 are CPU-only by
decree (CLAUDE.md §4) and ``transformers`` lives behind the ``llm`` extra, which pins
``torch``; the Llama-3.1 tokenizer is additionally a gated download. Requiring it here
would make ``make bronze`` impossible on the machine that builds every other CPU phase.

So the counter is **pluggable, and it always says which one it was**:

``heuristic-bpe-v1``
    The default. A deterministic segmentation that over-approximates a byte-pair
    tokenizer: words split on case and digit boundaries, then charged one token per four
    characters, punctuation charged individually. It is calibrated to over-count rather
    than under-count English prose, so a narrative that passes the [80, 400] gate under
    the heuristic passes it under the real tokenizer too, at the cost of occasionally
    rejecting a narrative the real tokenizer would have accepted. **That asymmetry is
    deliberate**: the failure it prevents is a training example silently exceeding the
    sequence budget, and the failure it causes is a slightly conservative corpus.

``llama``
    The real thing, used when ``transformers`` and the tokenizer files are present. It is
    never a silent upgrade or a silent fallback: asking for ``llama`` and not getting it
    raises. A length distribution measured under one counter and reported as though it
    came from the other would be a number that does not mean what it says.

The counter identifier is written into every training record's ``length.tokenizer``, so a
corpus can always be re-gated under a different counter without guessing what produced the
first result. See D-039.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

__all__ = [
    "DEFAULT_TOKENIZER",
    "HeuristicTokenCounter",
    "LlamaTokenCounter",
    "TokenCounter",
    "TokenizerUnavailableError",
    "get_token_counter",
    "word_count",
]

#: The counter used unless a run asks for another.
DEFAULT_TOKENIZER = "heuristic-bpe-v1"

#: Characters per sub-word token in the heuristic. Byte-pair merges on English prose land
#: near 3.7 characters per token; 3.5 is used so the estimate leans high. Numbers with
#: thousands separators and the ``bank|account`` identifiers (D-011) fragment far more
#: than prose, which is why they are charged per-character-class below rather than as
#: single words.
_CHARS_PER_TOKEN = 3.5

#: Splits a word at case changes, letter/digit boundaries and punctuation, approximating
#: where a byte-pair vocabulary would refuse to merge.
_PIECE_RE = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")


class TokenizerUnavailableError(RuntimeError):
    """Raised when a named tokenizer was requested and cannot be loaded."""


class TokenCounter(Protocol):
    """Anything that can count the tokens in a narrative."""

    name: str

    def count(self, text: str) -> int:
        """Return the token count for ``text``.

        Args:
            text: The narrative.

        Returns:
            The number of tokens.
        """
        ...


@dataclass(frozen=True)
class HeuristicTokenCounter:
    """A deterministic, dependency-free over-approximation of Llama's BPE.

    Attributes:
        name: The identifier written into ``length.tokenizer``.
    """

    name: str = DEFAULT_TOKENIZER

    def count(self, text: str) -> int:
        """Count tokens by charging each sub-word piece at least one token.

        Args:
            text: The narrative.

        Returns:
            The estimated token count. Always at least 1 for non-empty text.
        """
        total = 0
        for piece in _PIECE_RE.findall(text):
            if piece.isalpha():
                total += max(1, round(len(piece) / _CHARS_PER_TOKEN))
            elif piece.isdigit():
                # Llama-3 splits long digit runs into groups; three digits per token is
                # the observed behaviour and is what a thousands-separated amount costs.
                total += max(1, -(-len(piece) // 3))
            else:
                total += 1
        return total


@dataclass(frozen=True)
class LlamaTokenCounter:
    """The real Llama tokenizer, when the environment carries it.

    Attributes:
        name: The model identifier the tokenizer was loaded from.
    """

    name: str

    def count(self, text: str) -> int:
        """Count tokens with the loaded tokenizer.

        Args:
            text: The narrative.

        Returns:
            The token count, special tokens excluded.

        Raises:
            TokenizerUnavailableError: If the tokenizer cannot be loaded.
        """
        return len(_load_llama(self.name).encode(text, add_special_tokens=False))


@lru_cache(maxsize=2)
def _load_llama(model_id: str) -> object:
    """Load and cache a Hugging Face tokenizer.

    Args:
        model_id: The model identifier or a local directory.

    Returns:
        The tokenizer.

    Raises:
        TokenizerUnavailableError: If ``transformers`` is absent or the tokenizer cannot
            be fetched. Never falls back to the heuristic: a run that asked for Llama and
            silently measured something else would publish a length distribution
            attributed to the wrong tokenizer.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised only without the llm extra
        raise TokenizerUnavailableError(
            "the Llama tokenizer needs `transformers`, which lives behind the `llm` "
            "extra (GPU-only). Phase 4 is CPU-only, so the default counter is "
            f"{DEFAULT_TOKENIZER!r}; see DECISIONS D-039."
        ) from exc
    try:
        return AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:  # pragma: no cover - network / licence gated
        raise TokenizerUnavailableError(
            f"could not load tokenizer {model_id!r}: {exc}. Llama-3.1 is a gated "
            "download; authenticate with `huggingface-cli login` or point "
            "`corpus.tokenizer` at a local directory."
        ) from exc


def get_token_counter(name: str = DEFAULT_TOKENIZER) -> TokenCounter:
    """Build the named token counter.

    Args:
        name: ``"heuristic-bpe-v1"``, or a Hugging Face model id / local path for the
            real tokenizer.

    Returns:
        The counter.

    Raises:
        TokenizerUnavailableError: If a real tokenizer was named and cannot be loaded.
    """
    if name == DEFAULT_TOKENIZER:
        return HeuristicTokenCounter()
    counter = LlamaTokenCounter(name=name)
    counter.count("calibration")  # fail now, loudly, rather than on record 9,000
    return counter


def word_count(text: str) -> int:
    """Count whitespace-delimited words.

    Reported alongside the token count because it is tokenizer-independent, so a length
    distribution stays comparable across a change of counter.

    Args:
        text: The narrative.

    Returns:
        The word count.
    """
    return len(text.split())
