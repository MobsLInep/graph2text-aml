"""Render a fact record into a Bronze narrative, deterministically and faithfully.

The contract is narrow and the guarantees are strong:

- **Deterministic.** Family follows from ``facts.typology.label`` and a small number of
  documented rules; variant follows from a hash of ``case_id``. The same record renders
  the same text on every machine, every run, at any seed. The seed is recorded but is
  deliberately *not* consulted for variant choice — a corpus that changed shape with the
  global seed could not be regenerated from a manifest.
- **Faithful by construction, and provably so.** Every number written comes from the
  record, is formatted inside the checker's tolerance for its claim type, and is annotated
  with the field it came from. Running :mod:`g2t_aml.facts.checkers` over the result must
  yield zero CONTRADICTED. It is not asserted here; it is measured, per record, by
  :mod:`g2t_aml.corpus.validate`, and a failure is a bug in this module rather than a
  reason to loosen the gate.
- **Substrate-safe, with a hard error rather than a quiet omission.** Two layers: the
  family declares the availability flags it needs and is refused outright on a record that
  lacks one, and every individual slot re-checks. A slot whose value is absent *for this
  case* — no inbound transfers, a multi-currency aggregate that has no defined sum — drops
  its sentence, because that is an honest local absence. A slot whose value is absent
  *because the substrate cannot carry it* raises :class:`SubstrateViolation`, because
  reaching for it at all means the template pack and the mask have gone out of step, and
  the next such attempt might be one the record happens to be able to satisfy.

**The distinction between those two absences is read from the record, not from a table
here.** ``Unavailable.reason`` already encodes it: ``substrate_has_no_*`` is a mask fact,
``no_transfers_in_this_direction`` is a case fact. Keeping the rule in one line that
consults the record means a new sentinel in ``facts/`` cannot silently land on the wrong
side of it.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from g2t_aml.corpus.bronze import format as fmt
from g2t_aml.corpus.bronze.templates import (
    FAMILY_FOR_TYPOLOGY,
    MINIMAL_FAMILY,
    MINIMAL_NODE_CEILING,
    NO_FINDING_FAMILY,
    PHASE_DISPLAY,
    ROLE_DISPLAY,
    SALIENCE_SENTENCES,
    TOPOLOGY_FAMILY,
    TYPOLOGY_DISPLAY,
    UNCLASSIFIED_SUSPICIOUS_FAMILY,
    Family,
    Segment,
    Variant,
    family_for,
)
from g2t_aml.corpus.record import BronzeNarrative, SlotAnnotation
from g2t_aml.corpus.tokenization import TokenCounter, get_token_counter
from g2t_aml.facts.salience import field_value, required_fields
from g2t_aml.facts.schema import CaseFacts, Money, Unavailable
from g2t_aml.facts.vocab import ControlledVocabulary, RiskDescriptor, load_vocabulary

__all__ = [
    "MAX_TOKENS",
    "MIN_TOKENS",
    "RENDERER_VERSION",
    "RenderError",
    "SubstrateViolation",
    "VariantInapplicable",
    "render_bronze",
    "select_family",
    "select_variant",
]

#: Bumped whenever a change alters rendered text, so a corpus is attributable to an exact
#: renderer. Distinct from the schema versions: the format can stay frozen while the
#: prose changes, and both facts matter when a result is reproduced.
RENDERER_VERSION = "0.1.0"

#: Length bounds, in tokens of whatever counter the run configures. 80 is the floor below
#: which a "narrative" is a caption and would inflate surface metrics against Gold without
#: containing a report; 400 is the ceiling that keeps the fact block, the prompt and the
#: target inside a comfortable fine-tuning sequence budget with the graph prefix attached.
MIN_TOKENS = 80
MAX_TOKENS = 400

#: Per-case target band the length controller trims toward, well inside the hard ceiling.
#: **A corpus in which every narrative runs to the ceiling is a worse corpus**, for two
#: reasons that both show up in the paper: a generator trained on it learns that a report
#: is always maximal, and near-identical length with near-identical section structure is
#: exactly what drives self-BLEU up. The target is drawn deterministically from the case
#: id, so length varies across the corpus while staying reproducible per case.
_TARGET_MIN_TOKENS = 170
_TARGET_MAX_TOKENS = 300

#: Sections, in the order they appear. The four-part SAR structure.
SECTION_ORDER = ("subject", "activity", "pattern", "basis")

#: Reasons that mark a substrate-level absence rather than a case-level one. The prefix
#: covers every mask sentinel the fact layer emits; the explicit member is the permanent
#: one (D-030), which predates the prefix convention.
_SUBSTRATE_PREFIX = "substrate_has_no_"
_PERMANENT_REASONS = frozenset({"no_substrate_carries_jurisdiction"})

#: Maximum risk descriptors listed in one narrative. Every controlled descriptor that
#: holds is true, but listing nine of them turns the basis section into a checklist and
#: teaches the generator to enumerate rather than to explain.
_MAX_INDICATORS = 3

_PLACEHOLDER_RE = re.compile(r"\{([^{}]+)\}")


class RenderError(RuntimeError):
    """Base class for rendering failures."""


class SubstrateViolation(RenderError):  # noqa: N818 -- named for what it violates
    """Raised on any attempt to fill a slot the substrate cannot support.

    Invariant 4 in executable form. This is a hard error and not a warning: a template
    pack that reaches for a masked fact is wrong about the substrate, and the next reach
    may land on a field the record happens to populate.
    """


class VariantInapplicable(RenderError):  # noqa: N818 -- reads as a condition, not a fault
    """Raised when a variant's required segments cannot all be filled for a record."""


@dataclass(frozen=True)
class _Filled:
    """One resolved placeholder: its text and the annotations it carries.

    Attributes:
        text: The rendered text replacing the placeholder.
        annotations: ``(local_start, local_end, annotation)`` triples, offsets relative
            to :attr:`text`. A set-valued slot yields one annotation per member.
    """

    text: str
    annotations: tuple[tuple[int, int, SlotAnnotation], ...] = ()


@dataclass
class _RenderedSegment:
    """A segment rendered to text, with annotation offsets still segment-local."""

    text: str
    annotations: list[tuple[int, int, SlotAnnotation]]
    optional_detail: bool
    required: bool = False


def _digest(case_id: str, salt: str) -> int:
    """Return a stable integer derived from a case id.

    Args:
        case_id: The case identifier.
        salt: Distinguishes independent draws for the same case.

    Returns:
        A non-negative integer. SHA-256 rather than ``hash()``, which is salted per
        process and would make renders differ between runs.
    """
    return int(hashlib.sha256(f"{salt}:{case_id}".encode()).hexdigest()[:16], 16)


def _substrate_forbids(value: object) -> bool:
    """Report whether an absent value is absent because the substrate cannot carry it.

    Args:
        value: A value resolved from the fact record.

    Returns:
        True for a mask sentinel, False for a case-level sentinel or a measured null.
    """
    if not isinstance(value, Unavailable):
        return False
    return value.reason.startswith(_SUBSTRATE_PREFIX) or value.reason in _PERMANENT_REASONS


def _resolve(facts: CaseFacts, path: str) -> Any:
    """Resolve a field path, raising when the substrate forbids it.

    Args:
        facts: The record.
        path: A dotted field path.

    Returns:
        The value, or None when it is absent for this particular case.

    Raises:
        SubstrateViolation: If the field sits under a substrate-level sentinel.
    """
    value = field_value(facts, path)
    if _substrate_forbids(value):
        reason = value.reason if isinstance(value, Unavailable) else "unknown"
        raise SubstrateViolation(
            f"template reached for {path!r} on substrate {facts.dataset!r}, which cannot "
            f"carry it ({reason}). This is a template/mask disagreement, not a missing "
            "value: fix the family's requires_mask or its segments (invariant 4)."
        )
    if isinstance(value, Unavailable):
        return None
    return value


def _slot(path: str, span_text: str, raw: Any, claim_type: str) -> _Filled:
    """Build a single-annotation filled placeholder.

    Args:
        path: The field path.
        span_text: The rendered text.
        raw: The value before formatting.
        claim_type: The claim type the slot becomes.

    Returns:
        The filled placeholder.
    """
    annotation = SlotAnnotation(
        field_path=path,
        span=(0, len(span_text)),
        rendered_value=span_text,
        raw_value=raw,
        claim_type=claim_type,
    )
    return _Filled(text=span_text, annotations=((0, len(span_text), annotation),))


def _fill_field(  # noqa: PLR0911, PLR0912 -- one branch per slot kind. A dispatch table
    # would move the kind-to-formatter mapping away from the formatting, which is exactly
    # the correspondence a reader is here to check.
    facts: CaseFacts,
    spec: str,
    salt: int,
) -> _Filled | None:
    """Fill one ``{path:kind}`` placeholder.

    Args:
        facts: The record.
        spec: The placeholder body, e.g. ``"flow.total_outflow:money"`` or
            ``"temporal.burst_detected:bool:detected|not detected"``.
        salt: Drives the choice among equivalent surface forms.

    Returns:
        The filled placeholder, or None when the field is absent for this case.

    Raises:
        SubstrateViolation: If the field is masked out for this substrate.
        RenderError: If the kind is unknown, or the record's value has a type the kind
            cannot render — which means the template names a field it has misunderstood.
    """
    path, _, rest = spec.partition(":")
    kind, _, extra = rest.partition(":")
    value = _resolve(facts, path)
    if value is None:
        return None

    if kind == "count":
        return _slot(path, fmt.format_count(value), value, "numeric")
    if kind == "money":
        if not isinstance(value, Money):
            raise RenderError(f"{path} is not a monetary amount; got {value!r}")
        return _slot(path, fmt.format_money(value.value, value.currency), value.value, "numeric")
    if kind == "duration":
        return _slot(path, fmt.format_duration(float(value)), value, "temporal")
    if kind == "share":
        return _slot(path, fmt.format_percent(float(value)), value, "numeric")
    if kind == "density":
        return _slot(path, fmt.format_density(float(value)), value, "numeric")
    if kind == "timestamp":
        if not isinstance(value, datetime):
            raise RenderError(f"{path} is not a timestamp; got {value!r}")
        return _slot(path, fmt.format_timestamp(value), value.isoformat(), "temporal")
    if kind == "entity":
        return _slot(path, str(value), value, "entity")
    if kind == "role":
        forms = ROLE_DISPLAY.get(str(value))
        if not forms:
            raise RenderError(f"role {value!r} has no controlled surface form")
        return _slot(path, forms[salt % len(forms)], value, "categorical")
    if kind == "typology":
        display = TYPOLOGY_DISPLAY.get(str(value))
        if display is None:
            raise RenderError(f"typology {value!r} has no controlled surface form")
        return _slot(path, display, value, "categorical")
    if kind == "ordering":
        phases = tuple(str(p) for p in value)
        if not phases:
            return None
        return _slot(
            path, ", then ".join(PHASE_DISPLAY[p] for p in phases), list(phases), "temporal"
        )
    if kind == "set":
        return _fill_set(path, value)
    if kind == "bool":
        true_text, _, false_text = extra.partition("|")
        text = true_text if value else false_text
        return _slot(path, text, bool(value), "categorical")
    raise RenderError(f"unknown slot kind {kind!r} in placeholder {spec!r}")


def _fill_set(path: str, value: Any) -> _Filled | None:
    """Fill a set-valued slot, annotating each member separately.

    A single annotation over "ACH, Cash and Wire" would be unparseable as a claim, and
    the checker for a set field takes one member at a time. Annotating per member keeps
    each span a claim in its own right.

    Args:
        path: The field path.
        value: The set field's members.

    Returns:
        The filled placeholder, or None when the set is empty.
    """
    members = [str(v) for v in value]
    if not members:
        return None
    text = members[0] if len(members) == 1 else ", ".join(members[:-1]) + " and " + members[-1]
    annotations: list[tuple[int, int, SlotAnnotation]] = []
    cursor = 0
    for member in members:
        start = text.index(member, cursor)
        end = start + len(member)
        cursor = end
        annotations.append(
            (
                start,
                end,
                SlotAnnotation(
                    field_path=path,
                    span=(start, end),
                    rendered_value=member,
                    raw_value=member,
                    claim_type="categorical",
                ),
            )
        )
    return _Filled(text=text, annotations=tuple(annotations))


def _holding_descriptors(
    facts: CaseFacts, vocabulary: ControlledVocabulary
) -> list[tuple[RiskDescriptor, float]]:
    """Return every controlled risk descriptor whose binding holds for this record.

    A descriptor is emitted only when its condition is satisfied, which is what keeps
    Bronze free of CONTRADICTED qualitative claims by construction rather than by luck.
    Descriptors requiring an availability flag the substrate lacks are skipped silently —
    unlike a fact slot, a descriptor is inherently optional, so reaching for one the mask
    forbids is not a template bug.

    Args:
        facts: The record.
        vocabulary: The controlled vocabulary.

    Returns:
        ``(descriptor, bound value)`` pairs, in vocabulary order.
    """
    mask = facts.availability.to_dict()
    holding: list[tuple[RiskDescriptor, float]] = []
    for descriptor in vocabulary.risk_descriptors.values():
        if any(not mask.get(flag, False) for flag in descriptor.requires):
            continue
        value = field_value(facts, descriptor.binds_to)
        if value is None or isinstance(value, Unavailable | bool):
            continue
        if not isinstance(value, int | float):
            continue
        if descriptor.holds_for(value):
            holding.append((descriptor, float(value)))
    return holding


def _fill_indicators(
    facts: CaseFacts, vocabulary: ControlledVocabulary, salt: int
) -> _Filled | None:
    """Fill the ``{~indicators}`` placeholder.

    Args:
        facts: The record.
        vocabulary: The controlled vocabulary.
        salt: Drives which descriptors are chosen when more than the cap hold, and which
            surface phrase each is written with.

    Returns:
        The filled placeholder, or None when no descriptor holds.
    """
    holding = _holding_descriptors(facts, vocabulary)
    if not holding:
        return None
    rotation = salt % len(holding)
    chosen = (holding[rotation:] + holding[:rotation])[:_MAX_INDICATORS]

    parts: list[str] = []
    annotations: list[tuple[int, int, SlotAnnotation]] = []
    cursor = 0
    for i, (descriptor, bound) in enumerate(chosen):
        phrase = descriptor.phrase_variants[(salt + i) % len(descriptor.phrase_variants)]
        if i:
            cursor += len("; ")
        parts.append(phrase)
        annotations.append(
            (
                cursor,
                cursor + len(phrase),
                SlotAnnotation(
                    field_path=descriptor.binds_to,
                    span=(cursor, cursor + len(phrase)),
                    rendered_value=phrase,
                    raw_value=bound,
                    claim_type="qualitative",
                ),
            )
        )
        cursor += len(phrase)
    return _Filled(text="; ".join(parts), annotations=tuple(annotations))


def _fill_threshold(facts: CaseFacts, salt: int) -> _Filled | None:
    """Fill the ``{_threshold_reference}`` placeholder with a whitelisted citation.

    The vocabulary permits a regulatory reference only when the case's currency matches
    the one the threshold is denominated in. AMLworld is synthetic and carries no
    jurisdiction, so the citation is context ("the USD 10,000 reporting threshold") and
    never a finding about a filing obligation — which is exactly how the whitelist frames
    it.

    Args:
        facts: The record.
        salt: Chooses among the whitelisted surface forms.

    Returns:
        The filled placeholder, or None when the case has no transfer in the threshold
        currency and the citation would therefore be irrelevant.
    """
    flow = _resolve(facts, "flow.threshold_currency")
    currencies = _resolve(facts, "flow.currencies_involved")
    if flow is None or currencies is None or str(flow) not in {str(c) for c in currencies}:
        return None
    variants = (
        "the USD 10,000 reporting threshold",
        "the currency transaction reporting threshold",
    )
    phrase = variants[salt % len(variants)]
    return _slot("flow.threshold_reference", phrase, str(flow), "regulatory")


def _fill_plural(facts: CaseFacts, spec: str) -> _Filled | None:
    """Fill a ``{_p:path:singular:plural}`` placeholder. No claim, agreement only.

    Args:
        facts: The record.
        spec: The placeholder body.

    Returns:
        The singular or plural word, or None when the governing count is absent — in
        which case the sentence carrying it is dropped anyway.

    Raises:
        SubstrateViolation: If the governing field is masked out.
    """
    _, path, singular, plural = spec.split(":", 3)
    value = _resolve(facts, path)
    if value is None:
        return None
    return _Filled(text=singular if float(value) == 1 else plural)


def _render_segment(
    segment: Segment,
    facts: CaseFacts,
    vocabulary: ControlledVocabulary,
    salt: int,
) -> _RenderedSegment | None:
    """Render one segment, or report that it cannot be filled.

    Args:
        segment: The segment.
        facts: The record.
        vocabulary: The controlled vocabulary.
        salt: Drives surface-form choices.

    Returns:
        The rendered segment, or None when any placeholder is unfillable for this case.

    Raises:
        SubstrateViolation: If a placeholder names a masked field.
        VariantInapplicable: If a required segment cannot be filled.
        RenderError: If a placeholder is malformed.
    """
    pieces: list[str] = []
    annotations: list[tuple[int, int, SlotAnnotation]] = []
    cursor = 0
    last = 0
    for match in _PLACEHOLDER_RE.finditer(segment.text):
        literal = segment.text[last : match.start()]
        pieces.append(literal)
        cursor += len(literal)
        last = match.end()

        body = match.group(1)
        if body == "~indicators":
            filled = _fill_indicators(facts, vocabulary, salt)
        elif body == "_threshold_reference":
            filled = _fill_threshold(facts, salt)
        elif body.startswith("_p:"):
            filled = _fill_plural(facts, body)
        elif body.startswith("_"):
            raise RenderError(f"unknown computed placeholder {body!r}")
        else:
            filled = _fill_field(facts, body, salt)

        if filled is None:
            if segment.required:
                raise VariantInapplicable(
                    f"required segment could not be filled for case {facts.case_id!r}: "
                    f"{body!r} is absent from this record"
                )
            return None
        pieces.append(filled.text)
        for start, end, annotation in filled.annotations:
            annotations.append((cursor + start, cursor + end, annotation))
        cursor += len(filled.text)

    tail = segment.text[last:]
    pieces.append(tail)
    return _RenderedSegment(
        text="".join(pieces),
        annotations=annotations,
        optional_detail=segment.optional_detail,
        required=segment.required,
    )


def select_family(facts: CaseFacts) -> str:
    """Choose the template family for a record.

    The rules, in order, and each one exists because of something the data does:

    1. A substrate without monetary amounts gets ``topology_only``. It is the only family
       with no monetary or temporal slot, so it is the only one Elliptic2 can render.
    2. A subgraph of at most :data:`~g2t_aml.corpus.bronze.templates.MINIMAL_NODE_CEILING`
       accounts gets ``minimal_activity``. No motif detector can fire below three
       accounts, so any structural claim would be vacuous.
    3. A ground-truth typology selects its own family. After D-036 such a record always
       carries flagged transactions, so the family may cite the scheme and the evidence in
       one breath.
    4. ``unclassified`` **with** flagged transactions gets ``unclassified_suspicious``.
    5. ``unclassified`` **without** flagged transactions gets ``no_finding``, which states
       the absence of a pattern positively rather than papering over it (D-035).

    Args:
        facts: The record.

    Returns:
        The family key.
    """
    if not facts.availability.monetary_amounts:
        return TOPOLOGY_FAMILY
    if facts.structure.n_nodes <= MINIMAL_NODE_CEILING:
        return MINIMAL_FAMILY
    label = facts.typology.label
    if label in FAMILY_FOR_TYPOLOGY:
        return FAMILY_FOR_TYPOLOGY[label]
    labels = facts.labels
    flagged = 0 if isinstance(labels, Unavailable) else labels.n_illicit_transactions
    return UNCLASSIFIED_SUSPICIOUS_FAMILY if flagged else NO_FINDING_FAMILY


def select_variant(case_id: str, family: Family) -> int:
    """Choose a surface realisation, deterministically and independently of the seed.

    The index encodes all four section choices at once (see
    :meth:`~g2t_aml.corpus.bronze.templates.Family.realisation`), so one integer in the
    training record identifies the narrative's form exactly.

    Args:
        case_id: The case identifier.
        family: The family whose realisations are being chosen among.

    Returns:
        The realisation index, in ``[0, family.n_realisations)``.
    """
    return _digest(case_id, "variant") % family.n_realisations


def _assemble(
    sections: dict[str, list[_RenderedSegment]],
) -> tuple[str, dict[str, str], tuple[SlotAnnotation, ...]]:
    """Join rendered segments into a narrative, rebasing every annotation span.

    Args:
        sections: Rendered segments per section, in :data:`SECTION_ORDER`.

    Returns:
        ``(text, section texts, annotations)`` with annotation spans absolute and in
        document order.
    """
    section_texts: dict[str, str] = {}
    annotations: list[SlotAnnotation] = []
    parts: list[str] = []
    offset = 0
    for name in SECTION_ORDER:
        segments = sections.get(name) or []
        if not segments:
            continue
        if parts:
            offset += 2  # the blank line between sections
        local_parts: list[str] = []
        local_offset = 0
        for segment in segments:
            if local_parts:
                local_offset += 1  # the space between sentences
            for start, end, annotation in segment.annotations:
                absolute = (offset + local_offset + start, offset + local_offset + end)
                annotations.append(
                    SlotAnnotation(
                        field_path=annotation.field_path,
                        span=absolute,
                        rendered_value=annotation.rendered_value,
                        raw_value=annotation.raw_value,
                        claim_type=annotation.claim_type,
                    )
                )
            local_parts.append(segment.text)
            local_offset += len(segment.text)
        section_text = " ".join(local_parts)
        section_texts[name] = section_text
        parts.append(section_text)
        offset += len(section_text)
    text = "\n\n".join(parts)
    annotations.sort(key=lambda a: a.span)
    return text, section_texts, tuple(annotations)


def _salience_segments(
    facts: CaseFacts,
    vocabulary: ControlledVocabulary,
    mentioned: set[str],
    salt: int,
) -> list[_RenderedSegment]:
    """Render one sentence per required salient field the narrative has not mentioned.

    Bronze therefore reaches 100% salience coverage by construction, which is the ceiling
    Phase 10 scores learned systems against. Each sentence carries the field as an
    annotated slot: a fallback that mentioned a field without annotating it would raise
    coverage and lower verifiability at the same time.

    Args:
        facts: The record.
        vocabulary: The controlled vocabulary.
        mentioned: Field paths already annotated in the narrative.
        salt: Drives surface-form choices.

    Returns:
        The extra segments, possibly empty.
    """
    required, _ = required_fields(facts, vocabulary)
    extra: list[_RenderedSegment] = []
    for path in required:
        if path in mentioned or path not in SALIENCE_SENTENCES:
            continue
        rendered = _render_segment(Segment(SALIENCE_SENTENCES[path]), facts, vocabulary, salt)
        if rendered is not None:
            extra.append(rendered)
            mentioned.add(path)
    return extra


def _padding_segments(
    facts: CaseFacts,
    vocabulary: ControlledVocabulary,
    mentioned: set[str],
    salt: int,
) -> list[_RenderedSegment]:
    """Render further factual sentences, for a narrative that falls below the floor.

    Only fields the record actually supports and the narrative has not already used, each
    fully annotated. Padding that repeated a fact, or stated one without a slot, would buy
    length at the cost of the two properties the corpus exists to have.

    Args:
        facts: The record.
        vocabulary: The controlled vocabulary.
        mentioned: Field paths already annotated.
        salt: Drives surface-form choices.

    Returns:
        Candidate segments, in a deterministic order.
    """
    extra: list[_RenderedSegment] = []
    for path, sentence in SALIENCE_SENTENCES.items():
        if path in mentioned:
            continue
        rendered = _render_segment(Segment(sentence), facts, vocabulary, salt)
        if rendered is not None:
            extra.append(rendered)
            mentioned.add(path)
    return extra


def _annotate(text: str, slots: tuple[SlotAnnotation, ...]) -> str:
    """Produce the annotated form, with every slot marked in place.

    Args:
        text: The plain narrative.
        slots: The annotations, in document order.

    Returns:
        The narrative with each slot written ``{field.path|rendered value}``. Nested and
        overlapping spans do not occur, so a single left-to-right splice is exact.
    """
    parts: list[str] = []
    cursor = 0
    for slot in slots:
        start, end = slot.span
        if start < cursor:
            continue
        parts.append(text[cursor:start])
        parts.append(f"{{{slot.field_path}|{slot.rendered_value}}}")
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def render_bronze(
    facts: CaseFacts,
    family: str | None = None,
    variant: int | None = None,
    seed: int = 42,  # noqa: ARG001 -- recorded, deliberately not consulted; see the docstring
    *,
    vocabulary: ControlledVocabulary | None = None,
    token_counter: TokenCounter | None = None,
) -> BronzeNarrative:
    """Render one Bronze narrative from a fact record.

    Args:
        facts: The record to describe.
        family: Force a family. Defaults to :func:`select_family`. Forcing a family the
            substrate cannot support is the documented way to test the guard, and it
            raises.
        variant: Force a surface realisation, as an index into the family's composed
            realisations. Defaults to :func:`select_variant`.
        seed: Recorded on the output for provenance. **Not consulted**: rendering is a
            function of the record alone, so a corpus can be regenerated from a case
            manifest without knowing what seed produced it.
        vocabulary: The controlled vocabulary. Loaded from disk when omitted.
        token_counter: Counts tokens for the length bounds. Defaults to the configured
            heuristic counter.

    Returns:
        The narrative, its slot alignment and the template that produced it.

    Raises:
        SubstrateViolation: If any slot reaches for a fact the substrate cannot carry, or
            the family declares an availability requirement the record does not meet.
        VariantInapplicable: If a required segment cannot be filled for this record.
        KeyError: If ``family`` names no known family.
        IndexError: If ``variant`` is out of range for the family.
    """
    vocab = vocabulary if vocabulary is not None else load_vocabulary()
    counter = token_counter if token_counter is not None else get_token_counter()
    chosen = family_for(family) if family is not None else family_for(select_family(facts))

    mask = facts.availability.to_dict()
    if missing := [flag for flag in chosen.requires_mask if not mask.get(flag, False)]:
        raise SubstrateViolation(
            f"family {chosen.key!r} requires {missing} but substrate {facts.dataset!r} "
            f"does not provide {missing}; a narrative from this family would assert a "
            "fact the data does not carry (invariant 4)"
        )

    index = select_variant(facts.case_id, chosen) if variant is None else variant
    if not 0 <= index < chosen.n_realisations:
        raise IndexError(
            f"realisation {index} is out of range for family {chosen.key!r}, which has "
            f"{chosen.n_realisations}"
        )
    realisation: Variant = chosen.realisation(index)
    salt = _digest(facts.case_id, "surface")

    sections: dict[str, list[_RenderedSegment]] = {}
    for name in SECTION_ORDER:
        rendered: list[_RenderedSegment] = []
        for segment in getattr(realisation, name):
            filled = _render_segment(segment, facts, vocab, salt)
            if filled is not None:
                rendered.append(filled)
        sections[name] = rendered

    mentioned = {
        annotation.field_path
        for segments in sections.values()
        for segment in segments
        for _, _, annotation in segment.annotations
    }
    required, _ = required_fields(facts, vocab)
    protected = frozenset(required)
    sections["activity"].extend(_salience_segments(facts, vocab, mentioned, salt))

    _apply_length_bounds(facts, vocab, sections, mentioned, protected, salt, counter)

    text, section_texts, slots = _assemble(sections)
    return BronzeNarrative(
        case_id=facts.case_id,
        text=text,
        annotated=_annotate(text, slots),
        slots=slots,
        family=chosen.key,
        variant=index,
        sections=section_texts,
        renderer_version=RENDERER_VERSION,
    )


def _apply_length_bounds(
    facts: CaseFacts,
    vocabulary: ControlledVocabulary,
    sections: dict[str, list[_RenderedSegment]],
    mentioned: set[str],
    protected: frozenset[str],
    salt: int,
    counter: TokenCounter,
) -> None:
    """Bring a narrative to its per-case target length, in place.

    Trimming removes whole sentences, last-first, preferring the ones marked
    ``optional_detail``. Two things are never removed: a required segment, and any
    segment carrying a field on this case's salience list. Length control therefore
    cannot make a narrative inadequate, and cannot break the 100% salience coverage that
    is Bronze's ceiling.

    Padding, in the other direction, only ever appends further *annotated* facts the
    record supports and the narrative has not already used. Repeating a fact, or stating
    one without a slot, would buy length at the cost of the two properties the corpus
    exists to have.

    Args:
        facts: The record.
        vocabulary: The controlled vocabulary.
        sections: Rendered segments per section, mutated in place.
        mentioned: Field paths already annotated, mutated in place.
        protected: Field paths that may not be trimmed away.
        salt: Drives surface-form choices.
        counter: The token counter.
    """

    def tokens() -> int:
        return counter.count(_assemble(sections)[0])

    span = _TARGET_MAX_TOKENS - _TARGET_MIN_TOKENS
    target = _TARGET_MIN_TOKENS + _digest(facts.case_id, "length") % (span + 1)

    def droppable() -> list[tuple[int, str, int]]:
        found: list[tuple[int, str, int]] = []
        for name in SECTION_ORDER:
            for i, segment in enumerate(sections.get(name) or []):
                if segment.required:
                    continue
                if any(a.field_path in protected for _, _, a in segment.annotations):
                    continue
                found.append((0 if segment.optional_detail else 1, name, i))
        return sorted(found, key=lambda item: (item[0], SECTION_ORDER.index(item[1]), -item[2]))

    while tokens() > target:
        candidates = droppable()
        if not candidates:
            break
        _, name, i = candidates[0]
        sections[name].pop(i)

    if tokens() >= MIN_TOKENS:
        return
    for segment in _padding_segments(facts, vocabulary, mentioned, salt):
        sections["activity"].append(segment)
        if tokens() >= MIN_TOKENS:
            return
