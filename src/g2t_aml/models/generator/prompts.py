"""Building the prompt, in segments, so that every position's role is known exactly.

**Why segments and not a formatted string.** The loss must be computed on the completion
and on nothing else, and the soft-token positions must be locatable to the exact index.
Both are position-level facts about the tokenised sequence. Rendering one string and then
recovering the boundaries by searching for a marker in the *token* stream is where this
normally goes wrong: a tokeniser merges the marker with adjacent text, the recovered
boundary is off by one, and the model trains on a target shifted by a token — which
produces a loss curve that descends convincingly and a model that generates plausible text
starting one word late.

So the prompt is assembled as a list of :class:`PromptSegment`, each tokenised
independently and each carrying its role. Concatenating them gives the ids; concatenating
their roles gives the mask. No searching, no markers, no off-by-one.

**Truncation removes facts, never the narrative and never the graph.** When a case exceeds
``max_seq_len`` the serialised facts are truncated from the end. Truncating the completion
would teach the model to stop mid-sentence; truncating the soft tokens would change how
many positions the fusion layer must fill and break the splice. Cases that lose facts to
truncation are counted and reported, because a corpus quietly losing its tail is a
different experiment from the one described.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "DEFAULT_SYSTEM_MESSAGE",
    "IGNORE_INDEX",
    "TEXT_MODES",
    "BuiltPrompt",
    "PromptBuilder",
    "PromptSegment",
    "SegmentRole",
    "Tokenizer",
]

#: The label value cross-entropy skips. Named because it appears in four modules and a
#: literal -100 in any of them is indistinguishable from a typo.
IGNORE_INDEX = -100

#: The system message. Deliberately short: it is the prompt-cache prefix and it is
#: repeated on every one of ~10,000 training examples, so every token in it is a token
#: multiplied by the corpus size. It states the task and the two hard constraints the
#: Phase 3 checker enforces, and leaves everything else to the fact record.
DEFAULT_SYSTEM_MESSAGE = (
    "You are a financial-crime analyst drafting the narrative section of a Suspicious "
    "Activity Report. Describe only what the case record supports. Report suspicion, "
    "never guilt, and never assert a fact the record does not contain."
)

#: What accompanies the soft tokens in the prompt.
#:
#: ``full``        instruction + serialised facts + graph (S1, and B8)
#: ``none``        instruction + graph only — the graph is the sole source of case
#:                 information, which is the headline arm S2
#: ``serialised``  instruction + serialised facts, no graph (B7, the text-only baseline)
TEXT_MODES: tuple[str, ...] = ("full", "none", "serialised")


class SegmentRole(str, Enum):
    """What a run of tokens is, and therefore whether the loss sees it."""

    SYSTEM = "system"
    PROMPT = "prompt"
    SOFT = "soft"
    COMPLETION = "completion"

    @property
    def in_loss(self) -> bool:
        """Report whether this role contributes to the training loss.

        Returns:
            True only for :attr:`COMPLETION`. The system message and the prompt are
            inputs, and computing loss over them spends capacity teaching the model to
            reproduce its own instructions. The soft-token positions are *embeddings*,
            not tokens — there is no correct token id at those positions, so a loss there
            trains the model to predict the arbitrary placeholder id.
        """
        return self is SegmentRole.COMPLETION


@runtime_checkable
class Tokenizer(Protocol):
    """The slice of a tokeniser this module uses.

    As with :class:`~g2t_aml.models.generator.model.CausalLM`, kept narrow so the tests
    run against a stub without ``transformers`` installed.
    """

    eos_token_id: int
    pad_token_id: int

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Tokenise a string.

        Args:
            text: The text.
            add_special_tokens: Whether to add BOS/EOS.

        Returns:
            The token ids.
        """
        ...

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        """Detokenise ids back to text.

        Args:
            ids: The token ids.
            skip_special_tokens: Drop specials from the output.

        Returns:
            The text.
        """
        ...


@dataclass(frozen=True)
class PromptSegment:
    """One run of tokens with a single role.

    Attributes:
        role: What this run is.
        ids: Its token ids.
    """

    role: SegmentRole
    ids: list[int]

    def __len__(self) -> int:
        """Return the segment's token count.

        Returns:
            The number of ids.
        """
        return len(self.ids)


@dataclass(frozen=True)
class BuiltPrompt:
    """A tokenised example, with every position's role recoverable.

    Attributes:
        input_ids: The full sequence.
        labels: ``-100`` everywhere but the completion.
        soft_mask: True at exactly the soft-token positions.
        segments: The segments it was built from, in order, for diagnostics.
        n_facts_truncated: How many fact tokens were dropped to fit ``max_seq_len``.
        case_id: The case this example is about.
    """

    input_ids: list[int]
    labels: list[int]
    soft_mask: list[bool]
    segments: tuple[PromptSegment, ...]
    n_facts_truncated: int
    case_id: str

    @property
    def soft_start(self) -> int:
        """Return the index of the first soft token.

        Returns:
            The index, or ``-1`` when there are no soft tokens.
        """
        return self.soft_mask.index(True) if any(self.soft_mask) else -1

    @property
    def completion_start(self) -> int:
        """Return the index of the first completion token.

        Returns:
            The index, or ``len(input_ids)`` when there is no completion. Used to restrict
            the attention-mass diagnostic to the generated span.
        """
        for i, label in enumerate(self.labels):
            if label != IGNORE_INDEX:
                return i
        return len(self.input_ids)


class PromptBuilder:
    """Assembles training and inference prompts from a training record.

    One builder per run, constructed with the tokeniser and the arm's text mode, so that
    training and inference cannot drift apart: the guard and the evaluator build their
    prompts through this same object.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        n_soft_tokens: int,
        soft_token_id: int,
        text_mode: str = "full",
        max_seq_len: int = 2048,
        system_message: str = DEFAULT_SYSTEM_MESSAGE,
    ) -> None:
        """Build the prompt builder.

        Args:
            tokenizer: The tokeniser.
            n_soft_tokens: How many positions to reserve for the graph. 0 on a text-only
                arm.
            soft_token_id: The placeholder id written at reserved positions. Its embedding
                is overwritten before the model sees it, so its identity does not matter —
                but it must be a real vocabulary id so the embedding lookup does not fail.
            text_mode: One of :data:`TEXT_MODES`.
            max_seq_len: Truncation length.
            system_message: The system message.

        Raises:
            ValueError: If ``text_mode`` is unknown, or a graph-bearing mode was given no
                soft tokens.
        """
        if text_mode not in TEXT_MODES:
            raise ValueError(f"unknown text_mode {text_mode!r}; expected one of {TEXT_MODES}")
        if text_mode == "none" and n_soft_tokens == 0:
            raise ValueError(
                "text_mode='none' with no soft tokens leaves the model no case "
                "information at all; it would be trained to hallucinate a whole narrative"
            )
        self.tokenizer = tokenizer
        self.n_soft_tokens = n_soft_tokens
        self.soft_token_id = soft_token_id
        self.text_mode = text_mode
        self.max_seq_len = max_seq_len
        self.system_message = system_message

    def _encode(self, text: str) -> list[int]:
        """Tokenise without special tokens.

        Args:
            text: The text.

        Returns:
            The ids.
        """
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def build(self, record: dict[str, Any], *, for_training: bool = True) -> BuiltPrompt:
        """Turn one training record into a tokenised example.

        Args:
            record: A training record — ``case_id``, ``serialised_facts`` and, when
                training, ``target_narrative``.
            for_training: Append the target narrative as a completion. False builds an
                inference prompt, whose labels are all ``-100``.

        Returns:
            The built prompt.

        Raises:
            KeyError: If the record lacks a field this text mode needs.
            ValueError: If the prompt cannot fit in ``max_seq_len`` even with every fact
                token removed, which means the instruction and the narrative alone exceed
                the budget and no amount of truncation will help.
        """
        case_id = str(record["case_id"])
        segments: list[PromptSegment] = [
            PromptSegment(SegmentRole.SYSTEM, self._encode(f"<|system|>\n{self.system_message}\n"))
        ]

        if self.n_soft_tokens:
            segments.append(PromptSegment(SegmentRole.PROMPT, self._encode("<|user|>\nGraph:\n")))
            segments.append(
                PromptSegment(SegmentRole.SOFT, [self.soft_token_id] * self.n_soft_tokens)
            )
            segments.append(PromptSegment(SegmentRole.PROMPT, self._encode("\n")))
        else:
            segments.append(PromptSegment(SegmentRole.PROMPT, self._encode("<|user|>\n")))

        fact_ids: list[int] = []
        if self.text_mode in {"full", "serialised"}:
            fact_ids = self._encode(f"Case record:\n{record['serialised_facts']}\n")

        tail = PromptSegment(
            SegmentRole.PROMPT,
            self._encode("Write the SAR narrative for this case.\n<|assistant|>\n"),
        )

        completion: list[int] = []
        if for_training:
            completion = [
                *self._encode(str(record["target_narrative"])),
                self.tokenizer.eos_token_id,
            ]

        fixed = sum(len(s) for s in segments) + len(tail) + len(completion)
        budget = self.max_seq_len - fixed
        if budget < 0:
            raise ValueError(
                f"case {case_id} needs {fixed} tokens before any facts are added, over the "
                f"{self.max_seq_len} budget; raise max_seq_len or shorten the narrative"
            )

        n_truncated = max(0, len(fact_ids) - budget)
        if n_truncated:
            fact_ids = fact_ids[:budget]

        if fact_ids:
            segments.append(PromptSegment(SegmentRole.PROMPT, fact_ids))
        segments.append(tail)
        if completion:
            segments.append(PromptSegment(SegmentRole.COMPLETION, completion))

        input_ids: list[int] = []
        labels: list[int] = []
        soft_mask: list[bool] = []
        for segment in segments:
            input_ids.extend(segment.ids)
            soft_mask.extend([segment.role is SegmentRole.SOFT] * len(segment))
            labels.extend(segment.ids if segment.role.in_loss else [-100] * len(segment))

        return BuiltPrompt(
            input_ids=input_ids,
            labels=labels,
            soft_mask=soft_mask,
            segments=tuple(segments),
            n_facts_truncated=n_truncated,
            case_id=case_id,
        )
