"""Turn a rewritten narrative back into checkable claims, without an LLM.

Silver's verification loop runs thousands of times per corpus and sits inside a repair
loop, so its extractor has to be fast, deterministic and free. This module is the fast
deterministic path: **slot alignment against the Bronze spans, plus rule-based extraction
over the controlled vocabulary.**

The reasoning is that the rewrite is a *paraphrase of a known text*. Bronze already carries
a character-span alignment from every load-bearing value back to the fact field it came
from; a faithful rewrite preserves those values even as it moves and rewords them. So:

1. **Align.** Every Bronze slot value that reappears in the rewrite becomes a claim, parsed
   out of the *rewrite's* text at the span where it was found — never read from the record
   and never carried over from Bronze. That direction is the whole point (D-040): a claim
   built from the value the formatter started with compares the record against itself and
   reports any corpus as perfect.
2. **Scrutinise what is left.** A quantity, account identifier or risk descriptor in the
   rewrite that aligns to no Bronze slot is a **candidate addition** — something the model
   introduced. It is emitted as a claim naming no field, which the checker resolves to
   UNVERIFIABLE rather than to SUPPORTED, and enough of them exhaust the 0.05 budget and
   fail the record. Silence here would be the dangerous behaviour: an unaligned number
   that produced no claim at all would make an invented figure *raise* the supported rate.
3. **Report what went missing.** A Bronze slot with no alignment is a dropped fact, not an
   invented one. It is reported as reduced salience coverage rather than as a violation,
   which is the honest reading and the one the Phase 4 notes anticipated.

**Formatting drift is deliberately treated as non-alignment.** Matching is exact, and
nothing else: no unit coercion, no currency aliasing, no
numeric re-parsing to see whether two spellings mean the same amount. A looser matcher
would have to decide that "USD 26,780" and "26,779.82 US Dollar" are the same claim, and
every such decision is a place where a genuine numeric error gets absorbed into a
tolerance. The conservative failure — treating a correct-but-reworded value as an
unaligned candidate — costs unverifiable budget and shows up in the discard log, where a
human can see it. The permissive failure is silent.

Exact matching is only safe because the teacher's output is put through
:func:`canonicalise_narrative` first, once, before it is verified or stored. A model that
hard-wraps a paragraph puts a newline inside ``"26,779.82 US Dollar"``, and without
canonicalisation that correct value would align to nothing and be scored as both a dropped
fact and an invented one. Canonicalising the stored text rather than loosening the matcher
keeps the character spans on the record exact — ``narrative[span] == rendered_value`` has
to keep holding, and it cannot if the matcher is allowed to match text that differs from
what was stored.

**Phase 10 adds an LLM extractor as a cross-check, and the interface is shaped for it
now.** :class:`ClaimExtractor` takes the narrative and the fact record; anything a
particular strategy needs beyond those — the Bronze alignment here, an API client there —
is constructor state. Both implementations then satisfy one protocol and can be run over
the same corpus and compared.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from g2t_aml.corpus.bronze.format import FormatError
from g2t_aml.corpus.claims import claim_from_slot
from g2t_aml.corpus.record import BronzeNarrative, SlotAnnotation
from g2t_aml.facts.checkers import Claim, ClaimType
from g2t_aml.facts.schema import CaseFacts
from g2t_aml.facts.vocab import ControlledVocabulary, load_vocabulary

__all__ = [
    "ClaimExtractor",
    "ExtractionReport",
    "SlotAlignmentExtractor",
    "canonicalise_narrative",
    "extract_report",
]

#: An account identifier as the fact layer keys them: ``"<bank>|<account>"`` (D-011). An
#: identifier in a rewrite that is not in the case's inventory is H1, and it is the
#: hallucination class a paraphrasing model is most likely to produce by transcription.
_ACCOUNT_RE = re.compile(r"\b\d+\|[A-Za-z0-9]+\b")

#: Any numeric token: counts, amounts, shares, densities, with or without separators and a
#: trailing percent sign. Deliberately broad — the alignment pass removes everything the
#: rewrite legitimately carried over, and what remains is precisely what needs scrutiny.
_NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?%?")

#: An ISO-ish timestamp as the formatters render them.
_TIMESTAMP_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2})?)?\b")

#: Number-like tokens that are not quantitative claims about the case. Ordinals and the
#: section-structure numerals a model sometimes writes ("the four paragraphs below") would
#: otherwise each burn unverifiable budget for saying nothing.
_BENIGN_NUMBERS = frozenset({"1", "2", "3", "4", "one", "two", "three", "four"})

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ExtractionReport:
    """What an extraction found, beyond the claims themselves.

    Attributes:
        claims: Every claim, aligned and candidate-addition alike, in document order.
        aligned_paths: Fact fields the rewrite demonstrably preserved.
        dropped_paths: Fact fields Bronze asserted that the rewrite does not carry. A
            dropped *salient* field fails adequacy; a dropped incidental one is a
            legitimate editorial choice, and the two are separated by the salience list
            rather than here.
        added_spans: ``(start, end, text)`` for each candidate addition, so a discard log
            entry can quote the exact words that cost the record its budget.
        unparseable: Slot values that were found in the rewrite but could not be read back
            as claims. A formatter/parser inconsistency, not a model error.
        aligned_slots: The Bronze slots, relocated to where their values appear in *this*
            text. This is the ``target_slots`` block a training record carries, and it is
            emitted here rather than rebuilt by the caller so that there is exactly one
            implementation of the alignment. A second one would have to reproduce
            longest-value-first ordering and the token-boundary guard, and the failure
            mode of getting either subtly wrong is a corpus that scores itself as
            perfect (D-048).
    """

    claims: tuple[Claim, ...]
    aligned_paths: tuple[str, ...]
    dropped_paths: tuple[str, ...]
    added_spans: tuple[tuple[int, int, str], ...]
    unparseable: tuple[str, ...] = ()
    aligned_slots: tuple[SlotAnnotation, ...] = ()

    @property
    def n_added(self) -> int:
        """Return how many candidate additions were found.

        Returns:
            The count.
        """
        return len(self.added_spans)


class ClaimExtractor(Protocol):
    """The interface both the deterministic and the Phase 10 LLM extractor satisfy.

    Narrow on purpose: a strategy's own dependencies — the Bronze alignment, a model
    client — are constructor state, so the call site that runs an extractor over a corpus
    does not change when the strategy does.
    """

    def extract(self, narrative: str, facts: CaseFacts) -> list[Claim]:
        """Extract every checkable claim a narrative makes.

        Args:
            narrative: The narrative text.
            facts: The record it is about. Available for vocabulary resolution and
                availability, **not** as a source of claim values.

        Returns:
            The claims, in document order.
        """
        ...


class SlotAlignmentExtractor:
    """The fast deterministic extractor: Bronze span alignment plus vocabulary rules."""

    def __init__(
        self,
        bronze: BronzeNarrative,
        *,
        vocabulary: ControlledVocabulary | None = None,
    ) -> None:
        """Bind the extractor to the Bronze narrative the rewrite came from.

        Args:
            bronze: The Bronze narrative and its slot alignment. This is the reference the
                rewrite is aligned against.
            vocabulary: The controlled vocabulary, for risk-descriptor phrases. Loaded
                from disk when omitted.
        """
        self.bronze = bronze
        self.vocabulary = vocabulary if vocabulary is not None else load_vocabulary()

    def extract(self, narrative: str, facts: CaseFacts) -> list[Claim]:
        """Extract every checkable claim the rewrite makes.

        Args:
            narrative: The rewritten narrative.
            facts: The fact record, unused as a value source and passed for interface
                conformance and future strategies.

        Returns:
            The claims, in document order.
        """
        return list(self.report(narrative, facts).claims)

    def report(self, narrative: str, facts: CaseFacts) -> ExtractionReport:
        """Extract claims and the alignment diagnostics around them.

        Args:
            narrative: The rewritten narrative.
            facts: The fact record. Not read for values.

        Returns:
            The full report.
        """
        del facts  # interface parity; values must never come from the record (D-040)
        claims: list[Claim] = []
        consumed: list[tuple[int, int]] = []
        aligned: list[str] = []
        aligned_slots: list[SlotAnnotation] = []
        dropped: list[str] = []
        unparseable: list[str] = []

        # Longest value first, NOT document order. A rewrite reorders content, so a short
        # value can otherwise be found inside a longer one that has not been consumed yet:
        # the slot rendering "2" aligns inside "2022-09-02 15:01", the timestamp is then
        # reported as a dropped fact, and its leftover digits come back as invented
        # quantities -- one reordering charged twice. In Bronze's own document order the
        # long value is always reached first, which is why this never surfaced in Phase 4.
        ordered_slots = sorted(self.bronze.slots, key=lambda s: (-len(s.rendered_value), s.span))
        for slot in ordered_slots:
            span = _find_unconsumed(narrative, slot.rendered_value, consumed)
            if span is None:
                dropped.append(slot.field_path)
                continue
            consumed.append(span)
            relocated = _relocated(slot, span)
            try:
                claims.append(claim_from_slot(relocated, narrative))
            except (FormatError, ValueError) as exc:
                unparseable.append(f"{slot.field_path}: {exc}")
                continue
            aligned.append(slot.field_path)
            aligned_slots.append(relocated)

        claims.extend(self._descriptor_claims(narrative, consumed))
        claims.extend(self._regulatory_claims(narrative, consumed))
        claims.extend(self._entity_claims(narrative, consumed))
        additions = self._candidate_additions(narrative, consumed)
        claims.extend(claim for _, claim in additions)

        claims.sort(key=lambda c: c.text_span)
        return ExtractionReport(
            claims=tuple(claims),
            aligned_paths=tuple(dict.fromkeys(aligned)),
            dropped_paths=tuple(dict.fromkeys(dropped)),
            added_spans=tuple(span for span, _ in additions),
            unparseable=tuple(unparseable),
            aligned_slots=tuple(sorted(aligned_slots, key=lambda s: s.span)),
        )

    def _descriptor_claims(self, narrative: str, consumed: list[tuple[int, int]]) -> list[Claim]:
        """Emit a qualitative claim for every controlled risk descriptor in the text.

        A rewrite that introduces "rapid dispersal" where Bronze did not has made a
        quantitative assertion in qualitative clothing, and the vocabulary's binding table
        can adjudicate it exactly. Without this pass such a phrase would be invisible: it
        aligns to no slot and contains no digits.

        Args:
            narrative: The rewritten narrative.
            consumed: Spans already claimed, extended in place.

        Returns:
            One QUALITATIVE claim per descriptor occurrence.
        """
        haystack = narrative.lower()
        claims: list[Claim] = []
        phrases = sorted(
            (
                (phrase, descriptor)
                for descriptor in self.vocabulary.risk_descriptors.values()
                for phrase in descriptor.phrase_variants
            ),
            key=lambda pair: len(pair[0]),
            reverse=True,
        )
        for phrase, _descriptor in phrases:
            start = 0
            while (found := haystack.find(phrase, start)) >= 0:
                span = (found, found + len(phrase))
                start = span[1]
                if _overlaps(span, consumed):
                    continue
                consumed.append(span)
                claims.append(
                    Claim(
                        text_span=span,
                        field_path=None,
                        claim_type=ClaimType.QUALITATIVE,
                        value=narrative[span[0] : span[1]],
                        raw_text=narrative[span[0] : span[1]],
                    )
                )
        return claims

    def _regulatory_claims(self, narrative: str, consumed: list[tuple[int, int]]) -> list[Claim]:
        """Emit a regulatory claim for every whitelisted citation in the text.

        Runs **before** :meth:`_candidate_additions` and consumes the phrase, so the
        figure inside a permitted citation is not also charged as an unbacked quantity.
        Without this pass, "the USD 10,000 reporting threshold" — a reference the
        controlled vocabulary explicitly whitelists as context (``regulatory_references``)
        — leaves ``10,000`` aligned to nothing, and a correct sentence costs a record 6%
        of its unverifiable budget. Found in Phase 6 on a hand-written Gold narrative; the
        same sentence from a Silver teacher would have cost the same.

        A citation *outside* the whitelist is deliberately not matched here. It falls
        through to the candidate-addition pass and to
        :func:`~g2t_aml.facts.checkers.check_narrative_text`, which is where an invented
        rule is caught as H6 — matching only the whitelist means this pass can never
        launder one.

        Args:
            narrative: The narrative.
            consumed: Spans already claimed, extended in place.

        Returns:
            One REGULATORY claim per whitelisted citation occurrence.
        """
        haystack = narrative.lower()
        claims: list[Claim] = []
        phrases = sorted(
            (
                phrase
                for reference in self.vocabulary.regulatory.values()
                for phrase in reference.phrase_variants
            ),
            key=len,
            reverse=True,
        )
        for phrase in phrases:
            start = 0
            while (found := haystack.find(phrase, start)) >= 0:
                span = (found, found + len(phrase))
                start = span[1]
                if _overlaps(span, consumed):
                    continue
                consumed.append(span)
                claims.append(
                    Claim(
                        text_span=span,
                        field_path=None,
                        claim_type=ClaimType.REGULATORY,
                        value=narrative[span[0] : span[1]],
                        raw_text=narrative[span[0] : span[1]],
                    )
                )
        return claims

    def _entity_claims(self, narrative: str, consumed: list[tuple[int, int]]) -> list[Claim]:
        """Emit an entity claim for every account identifier in the text.

        Checked against ``entity_inventory``; an identifier that is not in the case is H1.
        Transcribing an account id slightly wrong is one of the few hallucinations a
        careful paraphrasing model still makes, and it is fully decidable.

        Args:
            narrative: The rewritten narrative.
            consumed: Spans already claimed, extended in place.

        Returns:
            One ENTITY claim per unconsumed identifier occurrence.
        """
        claims: list[Claim] = []
        for match in _ACCOUNT_RE.finditer(narrative):
            span = match.span()
            if _overlaps(span, consumed):
                continue
            consumed.append(span)
            claims.append(
                Claim(
                    text_span=span,
                    field_path=None,
                    claim_type=ClaimType.ENTITY,
                    value=match.group(0),
                    raw_text=match.group(0),
                )
            )
        return claims

    def _candidate_additions(
        self, narrative: str, consumed: list[tuple[int, int]]
    ) -> list[tuple[tuple[int, int, str], Claim]]:
        """Emit a claim for every quantity the rewrite carries that Bronze did not.

        These name no fact field, so the checker returns UNVERIFIABLE — never SUPPORTED.
        That is the correct three-valued answer: an unaligned figure has not been shown to
        be wrong, it has been shown to be unbacked, and UNVERIFIABLE is exactly the bucket
        for compliance-dangerous assertions the graph cannot support. Enough of them
        exceed the 0.05 budget and the record is repaired or discarded.

        Args:
            narrative: The rewritten narrative.
            consumed: Spans already claimed, extended in place.

        Returns:
            ``((start, end, text), claim)`` per candidate addition.
        """
        found: list[tuple[tuple[int, int, str], Claim]] = []
        for pattern, claim_type in (
            (_TIMESTAMP_RE, ClaimType.TEMPORAL),
            (_NUMBER_RE, ClaimType.NUMERIC),
        ):
            for match in pattern.finditer(narrative):
                span = match.span()
                text = match.group(0)
                if _overlaps(span, consumed) or text.strip(",.%") in _BENIGN_NUMBERS:
                    continue
                consumed.append(span)
                found.append(
                    (
                        (span[0], span[1], text),
                        Claim(
                            text_span=span,
                            field_path=None,
                            claim_type=claim_type,
                            value=text,
                            raw_text=text,
                        ),
                    )
                )
        return found


def _relocated(slot: SlotAnnotation, span: tuple[int, int]) -> SlotAnnotation:
    """Return the slot with its span moved to where the value appears in the rewrite.

    The rendered value and the claim type carry over; the span does not, because the
    rewrite moved the words. :func:`g2t_aml.corpus.claims.claim_from_slot` then asserts
    that the new span really does hold that text before parsing it, so a mis-located
    alignment fails loudly instead of producing a claim about the wrong words.

    Args:
        slot: The Bronze annotation.
        span: Where the value was found in the rewrite.

    Returns:
        The relocated annotation.
    """
    return SlotAnnotation(
        field_path=slot.field_path,
        span=span,
        rendered_value=slot.rendered_value,
        raw_value=slot.raw_value,
        claim_type=slot.claim_type,
    )


#: Characters that, sitting immediately either side of a match, mean the match is a
#: fragment of a longer token rather than the value itself. ``12`` inside ``126``, ``09``
#: inside ``2022-09-02``, ``2022`` inside ``2022-09-02``. A trailing ``.`` is deliberately
#: absent: a value at the end of a sentence is followed by one.
_LEFT_BOUNDARY = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz|:-,."
_RIGHT_BOUNDARY = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz|:-"


def _find_unconsumed(
    narrative: str, value: str, consumed: list[tuple[int, int]]
) -> tuple[int, int] | None:
    """Find the first whole-token occurrence of a value that no claim has taken.

    Occurrence-by-occurrence rather than first-match so that a value appearing twice —
    two counts that happen to be equal — produces two claims rather than one claim and one
    spurious candidate addition.

    The token-boundary guard is the second half of the fix that longest-first ordering
    starts: ordering stops a short value being consumed by a longer *slot*, and this stops
    it being consumed by any longer token in the text, including one the rewrite wrote
    itself. Aligning ``12`` inside ``126`` would file a claim about the wrong characters
    and report the real quantity as invented.

    Args:
        narrative: The rewritten narrative.
        value: The Bronze rendered value to locate.
        consumed: Spans already claimed.

    Returns:
        The span, or None when the value does not appear as a whole token.
    """
    if not value:
        return None
    start = 0
    while (found := narrative.find(value, start)) >= 0:
        span = (found, found + len(value))
        before = narrative[found - 1] if found > 0 else ""
        after = narrative[span[1]] if span[1] < len(narrative) else ""
        if (
            not _overlaps(span, consumed)
            and before not in _LEFT_BOUNDARY
            and after not in _RIGHT_BOUNDARY
        ):
            return span
        start = found + 1
    return None


def _overlaps(span: tuple[int, int], consumed: list[tuple[int, int]]) -> bool:
    """Report whether a span intersects any already-claimed span.

    Args:
        span: The candidate span.
        consumed: Spans already claimed.

    Returns:
        True when the candidate overlaps at least one of them.
    """
    return any(span[0] < end and start < span[1] for start, end in consumed)


def canonicalise_narrative(text: str) -> str:
    """Put teacher output into the one form that is verified, stored and trained on.

    Applied **once**, to the raw completion, before anything reads it. Everything
    downstream — extraction, the checker, the character spans on the record, the corpus
    file — then sees the same bytes, which is what allows alignment to insist on exact
    matches without punishing a model for hard-wrapping a paragraph.

    Paragraph structure survives: a blank line is the SAR section boundary and the
    four-part structure is part of what is being generated. Only intra-paragraph
    whitespace runs collapse.

    Args:
        text: The raw completion.

    Returns:
        The canonical narrative: paragraphs separated by exactly one blank line, single
        spaces within a paragraph, no leading or trailing whitespace.
    """
    paragraphs = [
        _WHITESPACE_RE.sub(" ", block).strip() for block in re.split(r"\n\s*\n", text.strip())
    ]
    return "\n\n".join(p for p in paragraphs if p)


def extract_report(
    narrative: str,
    facts: CaseFacts,
    bronze: BronzeNarrative,
    *,
    vocabulary: ControlledVocabulary | None = None,
) -> ExtractionReport:
    """Extract claims and diagnostics for one rewrite.

    Args:
        narrative: The rewritten narrative.
        facts: The fact record.
        bronze: The Bronze narrative it was rewritten from.
        vocabulary: The controlled vocabulary. Loaded from disk when omitted.

    Returns:
        The extraction report.
    """
    return SlotAlignmentExtractor(bronze, vocabulary=vocabulary).report(narrative, facts)
