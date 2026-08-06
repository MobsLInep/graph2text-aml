"""Live validation while the annotator types, and why it never blocks submission.

The interface flags a forbidden phrase, an out-of-inventory account, a masked-substrate
assertion or a length violation as it is written, in place. It does not prevent the
annotator from submitting it.

**That is a deliberate choice and it is the more useful one.** A blocking validator trains
people to write around the checker rather than to write accurately: the phrase gets
rephrased until the highlight disappears, and what is measured afterwards is the
annotator's skill at evading a regex. Worse, it destroys the phase's most valuable
by-product. A flag that a domain expert *overrode* is evidence about the rule, not about
the person — a forbidden-phrase list that fires on correct writing is a list that needs
changing, and the only way to discover that is to let the writing through and count.

So every flag is recorded with the text that raised it and whether the annotator
overrode it, and :mod:`g2t_aml.human.gold_ingest` reports override rates per rule. A rule
overridden by two calibrated annotators on a fifth of items is a finding.

**Submission is not the last gate.** The interface runs the real Phase 3 checker on submit
and shows any CONTRADICTED claim for correction before the item is saved. That check *is*
adversarial — it compares against the record rather than against a word list — and it is
the one an annotator cannot write around, because the only way to clear it is to state
what the record says.

Everything here reuses the vocabulary and taxonomy frozen in Phase 3. No phrase list is
defined in this module: a rule the annotator sees and a rule the metric applies must be
the same rule, and duplicating the list is how they stop being.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from g2t_aml.corpus.tokenization import TokenCounter, get_token_counter
from g2t_aml.facts.salience import field_value
from g2t_aml.facts.schema import CaseFacts, Money, Unavailable
from g2t_aml.facts.taxonomy import HallucinationClass
from g2t_aml.facts.vocab import ControlledVocabulary, load_vocabulary

__all__ = [
    "MAX_TOKENS",
    "MIN_TOKENS",
    "SECTION_HEADINGS",
    "LiveFlag",
    "Severity",
    "ValidationSummary",
    "validate_draft",
]

#: The length gate, the same bounds the ten-point harness applies (check 6).
MIN_TOKENS = 80
MAX_TOKENS = 400

#: The four SAR sections, in order. Matched case-insensitively at the start of a line.
SECTION_HEADINGS: tuple[str, ...] = (
    "SUBJECT & SCOPE",
    "ACTIVITY OBSERVED",
    "PATTERN & TYPOLOGY",
    "BASIS & ACTION",
)

#: An account identifier as the fact layer keys them, ``"<bank>|<account>"`` (D-011).
_ACCOUNT_RE = re.compile(r"\b\d+\|[A-Za-z0-9]+\b")

#: A currency-shaped or amount-shaped token. Used only on a substrate whose flow family is
#: masked, where writing one at all is an assertion the record cannot license.
_AMOUNT_RE = re.compile(r"\b\d[\d,]*\.\d{2}\b|\b(?:USD|EUR|GBP)\b", re.IGNORECASE)

#: A wall-clock time. Used only where the temporal family is masked.
_CLOCK_RE = re.compile(r"\b\d{1,2}:\d{2}\b|\b\d{4}-\d{2}-\d{2}\b")

#: Section-heading detection. A line that is a heading and nothing else.
_HEADING_RE = re.compile(r"^\s*(?:\[\s*(\d)\s*\]\s*)?([A-Za-z &]+?)\s*:?\s*$")


class Severity(str, Enum):
    """How much a flag matters.

    Three levels rather than two, because the interface has three responses: stop and
    read (``critical``), reconsider (``warning``), and be aware (``info``). Collapsing
    the last two makes the panel noisy, and a noisy panel is an ignored panel.
    """

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class LiveFlag:
    """One thing the interface wants the annotator to look at.

    Attributes:
        rule: Stable identifier for the rule that fired, e.g. ``"forbidden:guilt"``.
            Stable because the override rates are aggregated by it.
        severity: How much it matters.
        message: What to tell the annotator, in their language rather than the code's.
        span: ``(start, end)`` character offsets into the draft, for in-place
            highlighting. ``(0, 0)`` for a flag about the draft as a whole.
        hallucination_class: The taxonomy class this would be if it survived to the
            finished narrative, or None for a flag that is not a hallucination — a length
            violation, a missing section.
        excerpt: The exact text that raised it, quoted back so the annotator does not
            have to hunt.
    """

    rule: str
    severity: Severity
    message: str
    span: tuple[int, int] = (0, 0)
    hallucination_class: str | None = None
    excerpt: str = ""

    @property
    def is_critical(self) -> bool:
        """Report whether this flag is in the Critical Error Rate's classes.

        Returns:
            True when the class is H4, H6 or H7.
        """
        if self.hallucination_class is None:
            return False
        return HallucinationClass[self.hallucination_class].is_critical

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised flag.

        Returns:
            A JSON-serialisable mapping, stored on the annotation record.
        """
        return {
            "rule": self.rule,
            "severity": self.severity.value,
            "message": self.message,
            "span": [self.span[0], self.span[1]],
            "hallucination_class": self.hallucination_class,
            "excerpt": self.excerpt,
        }


@dataclass(frozen=True)
class ValidationSummary:
    """Everything the interface shows beneath the writing box.

    Attributes:
        flags: The flags raised, most severe first.
        n_tokens: Token count under the named counter.
        tokenizer: Which counter measured it.
        length_ok: Whether the count is inside the gate.
        sections_found: Which of the four sections were detected, in order.
        salient_mentioned: Salient field paths the draft appears to mention.
        salient_missing: Salient field paths it does not.
    """

    flags: tuple[LiveFlag, ...]
    n_tokens: int
    tokenizer: str
    length_ok: bool
    sections_found: tuple[str, ...]
    salient_mentioned: tuple[str, ...] = ()
    salient_missing: tuple[str, ...] = ()

    @property
    def n_critical(self) -> int:
        """Return how many critical flags were raised.

        Returns:
            The count.
        """
        return sum(1 for f in self.flags if f.is_critical)

    @property
    def sections_complete(self) -> bool:
        """Report whether all four sections are present, in order.

        Returns:
            True when every heading was found in the declared order.
        """
        return self.sections_found == SECTION_HEADINGS

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised summary.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "flags": [f.to_dict() for f in self.flags],
            "n_flags": len(self.flags),
            "n_critical": self.n_critical,
            "n_tokens": self.n_tokens,
            "tokenizer": self.tokenizer,
            "length_ok": self.length_ok,
            "sections_found": list(self.sections_found),
            "sections_complete": self.sections_complete,
            "salient_mentioned": list(self.salient_mentioned),
            "salient_missing": list(self.salient_missing),
        }


_SEVERITY_ORDER = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}


def _forbidden_flags(text: str, vocabulary: ControlledVocabulary) -> list[LiveFlag]:
    """Flag every forbidden phrase in the draft.

    Every occurrence, not the first per group. The checker stops at one hit per group
    because one is enough to fail the record; an annotator needs to see all of them, or
    they fix one and resubmit into the next.

    Args:
        text: The draft.
        vocabulary: The controlled vocabulary.

    Returns:
        One flag per occurrence.
    """
    haystack = text.lower()
    flags: list[LiveFlag] = []
    seen: list[tuple[int, int]] = []
    for group, (hallucination_class, phrases) in vocabulary.forbidden.items():
        for phrase in sorted(phrases, key=len, reverse=True):
            start = 0
            while (found := haystack.find(phrase, start)) >= 0:
                span = (found, found + len(phrase))
                start = span[1]
                if any(span[0] < e and s < span[1] for s, e in seen):
                    continue
                seen.append(span)
                flags.append(
                    LiveFlag(
                        rule=f"forbidden:{group}",
                        severity=(
                            Severity.CRITICAL
                            if HallucinationClass[hallucination_class].is_critical
                            else Severity.WARNING
                        ),
                        message=_forbidden_message(group, phrase),
                        span=span,
                        hallucination_class=hallucination_class,
                        excerpt=text[span[0] : span[1]],
                    )
                )
    return flags


def _forbidden_message(group: str, phrase: str) -> str:
    """Return the annotator-facing explanation for a forbidden phrase.

    Args:
        group: The vocabulary group that owns the phrase.
        phrase: The phrase found.

    Returns:
        A sentence saying what is wrong and what to write instead.
    """
    guidance = {
        "guilt": (
            "A SAR reports suspicion, never a finding of guilt. Use a hedge from the "
            "allowed list — 'appears consistent with', 'warrants further review'."
        ),
        "entity_type": (
            "Neither substrate carries an entity-type column, so there is no evidence "
            "for any business type. Describe what the account did, not what it is."
        ),
        "completeness": (
            "A case holds about 65% of its stream's transactions on average. Describe "
            "the activity observed in this window, not the scheme as a whole."
        ),
        "motive": (
            "Intent is off-graph. Nothing in a subgraph evidences why someone acted; "
            "state the pattern and let the reader draw the inference."
        ),
    }
    return f"{phrase!r}: " + guidance.get(group, "outside the controlled vocabulary.")


def _entity_flags(text: str, facts: CaseFacts) -> list[LiveFlag]:
    """Flag account identifiers that are not in the case.

    Args:
        text: The draft.
        facts: The record, whose ``entity_inventory`` is the closed list.

    Returns:
        One flag per out-of-inventory identifier occurrence.
    """
    inventory = set(facts.entity_inventory.node_ids)
    flags: list[LiveFlag] = []
    for match in _ACCOUNT_RE.finditer(text):
        if match.group(0) in inventory:
            continue
        flags.append(
            LiveFlag(
                rule="entity:not_in_case",
                severity=Severity.CRITICAL,
                message=(
                    f"{match.group(0)!r} is not one of the {len(inventory)} accounts in "
                    "this case. Check the identifier against the subject panel."
                ),
                span=match.span(),
                hallucination_class="H1",
                excerpt=match.group(0),
            )
        )
    return flags


def _masked_family_flags(text: str, facts: CaseFacts) -> list[LiveFlag]:
    """Flag assertions about fact families this substrate cannot support.

    Invariant 4, applied to a draft as it is written. On Elliptic2 there are no amounts
    and no wall-clock times, so a figure with two decimal places or a ``14:20`` in the
    draft is an assertion nothing can verify — and it is the assertion an annotator
    familiar with AMLworld is most likely to make out of habit.

    Args:
        text: The draft.
        facts: The record.

    Returns:
        One flag per occurrence, at most a handful per family.

    """
    flags: list[LiveFlag] = []
    checks = (
        ("flow", _AMOUNT_RE, "amounts or currencies", "H2"),
        ("temporal", _CLOCK_RE, "wall-clock timing", "H3"),
    )
    for attribute, pattern, what, hallucination_class in checks:
        if not isinstance(getattr(facts, attribute), Unavailable):
            continue
        for match in pattern.finditer(text):
            flags.append(
                LiveFlag(
                    rule=f"masked:{attribute}",
                    severity=Severity.CRITICAL,
                    message=(
                        f"This substrate carries no {what}, so {match.group(0)!r} "
                        "asserts something the record cannot support. Describe the "
                        "structure instead."
                    ),
                    span=match.span(),
                    hallucination_class=hallucination_class,
                    excerpt=match.group(0),
                )
            )
    return flags


def _hedging_flags(text: str, facts: CaseFacts, vocabulary: ControlledVocabulary) -> list[LiveFlag]:
    """Flag an inferred typology asserted without a required hedge.

    Args:
        text: The draft.
        facts: The record.
        vocabulary: The controlled vocabulary.

    Returns:
        At most one flag.
    """
    if facts.typology.source != "inferred" or facts.typology.label == "unclassified":
        return []
    haystack = text.lower()
    if any(hedge in haystack for hedge in vocabulary.required_for_inferred):
        return []
    return [
        LiveFlag(
            rule="hedging:inferred_typology",
            severity=Severity.WARNING,
            message=(
                f"The typology {facts.typology.label!r} was inferred from motif "
                "detection, not read from ground truth, so it must appear inside one of: "
                f"{', '.join(vocabulary.required_for_inferred)}."
            ),
            hallucination_class="H5",
        )
    ]


def _sections_in(text: str) -> tuple[str, ...]:
    """Return the four-part section headings found, in the order they appear.

    Args:
        text: The draft.

    Returns:
        The recognised headings. A heading written out of order appears out of order
        here, so ``sections_complete`` fails on it — the four-part structure is an
        ordering as much as a set.
    """
    found: list[str] = []
    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match is None:
            continue
        candidate = match.group(2).strip().upper()
        for heading in SECTION_HEADINGS:
            if candidate == heading and heading not in found:
                found.append(heading)
    return tuple(found)


def _length_flags(n_tokens: int, counter_name: str) -> list[LiveFlag]:
    """Flag a draft outside the length gate.

    Args:
        n_tokens: The measured count.
        counter_name: Which counter measured it.

    Returns:
        At most one flag.
    """
    if MIN_TOKENS <= n_tokens <= MAX_TOKENS:
        return []
    direction = "short" if n_tokens < MIN_TOKENS else "long"
    return [
        LiveFlag(
            rule="length:out_of_bounds",
            severity=Severity.WARNING,
            message=(
                f"{n_tokens} tokens ({counter_name}) is too {direction}; the corpus gate "
                f"is [{MIN_TOKENS}, {MAX_TOKENS}] and a record outside it fails "
                "validation at ingestion."
            ),
        )
    ]


def validate_draft(
    text: str,
    facts: CaseFacts,
    *,
    vocabulary: ControlledVocabulary | None = None,
    token_counter: TokenCounter | None = None,
    salient_fields: tuple[str, ...] = (),
) -> ValidationSummary:
    """Validate a draft narrative as it is being written.

    Args:
        text: The draft, exactly as typed.
        facts: The record it is about.
        vocabulary: The controlled vocabulary. Loaded from disk when omitted.
        token_counter: The counter for the length gate. The heuristic counter when
            omitted, which is what the corpus gate uses by default.
        salient_fields: The case's required salience list, so the panel can show what is
            still missing. Empty disables the coverage report rather than reporting
            everything as missing.

    Returns:
        The summary. **Never raises on bad input**: a half-typed draft is the normal case
        here, and a validator that threw on one would take the editor down mid-sentence.
    """
    vocab = vocabulary if vocabulary is not None else load_vocabulary()
    counter = token_counter if token_counter is not None else get_token_counter()

    flags: list[LiveFlag] = []
    flags += _forbidden_flags(text, vocab)
    flags += _entity_flags(text, facts)
    flags += _masked_family_flags(text, facts)
    flags += _hedging_flags(text, facts, vocab)

    n_tokens = counter.count(text)
    flags += _length_flags(n_tokens, counter.name)

    sections = _sections_in(text)
    if text.strip() and len(sections) < len(SECTION_HEADINGS):
        missing = [h for h in SECTION_HEADINGS if h not in sections]
        flags.append(
            LiveFlag(
                rule="structure:missing_section",
                severity=Severity.INFO,
                message="Not yet written: " + ", ".join(f"[{h}]" for h in missing),
            )
        )

    mentioned, missing_salient = _salience_coverage(text, facts, salient_fields)
    if salient_fields and missing_salient:
        flags.append(
            LiveFlag(
                rule="salience:not_yet_mentioned",
                severity=Severity.INFO,
                message=(
                    "Salient for this typology and not yet mentioned: " + ", ".join(missing_salient)
                ),
            )
        )

    flags.sort(key=lambda f: (_SEVERITY_ORDER[f.severity], f.span))
    return ValidationSummary(
        flags=tuple(flags),
        n_tokens=n_tokens,
        tokenizer=counter.name,
        length_ok=MIN_TOKENS <= n_tokens <= MAX_TOKENS,
        sections_found=sections,
        salient_mentioned=mentioned,
        salient_missing=missing_salient,
    )


def _salience_coverage(
    text: str, facts: CaseFacts, salient_fields: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Guess which salient fields the draft mentions, for the live hint only.

    **This is a hint, not the metric.** It looks for the rendered value as a substring,
    which is a weaker test than the alignment
    :mod:`g2t_aml.corpus.silver.claim_extraction` performs at ingestion. It is here so an
    annotator can see what they have left out while they can still fix it; the number that
    reaches the record is computed at ingestion by the real extractor, and the two are
    deliberately not the same code — a live hint that ran the full alignment on every
    keystroke would be too slow, and one that *was* the metric would let an annotator tune
    to it.

    Args:
        text: The draft.
        facts: The record.
        salient_fields: The required field paths.

    Returns:
        ``(mentioned, missing)``.
    """
    if not salient_fields:
        return (), ()
    haystack = text.lower()
    mentioned: list[str] = []
    missing: list[str] = []
    for path in salient_fields:
        value = field_value(facts, path)
        needle = _needle_for(value)
        (mentioned if needle and needle in haystack else missing).append(path)
    return tuple(mentioned), tuple(missing)


def _needle_for(value: object) -> str:
    """Return the lower-cased string to look for when hinting at a salient field.

    Args:
        value: The value read from the record.

    Returns:
        The needle, or an empty string when the value has no usable surface form.
    """
    if value is None or isinstance(value, Unavailable):
        return ""
    if isinstance(value, Money):
        return f"{value.value:,.2f}".lower()
    if isinstance(value, bool):
        return ""
    if isinstance(value, float):
        return f"{value:.1f}".rstrip("0").rstrip(".").lower()
    if isinstance(value, tuple | list):
        return str(value[0]).replace("_", " ").lower() if value else ""
    return str(value).replace("_", " ").lower()
