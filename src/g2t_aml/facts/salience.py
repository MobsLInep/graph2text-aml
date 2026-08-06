"""What an adequate narrative must mention, per typology.

**Defined now, before a single narrative has been generated.** That timing is the whole
point. Salience decided after inspecting model output is not a standard, it is a
description of whatever the model happened to produce, and every subsequent "adequacy"
number measured against it is circular. The lists live in ``schemas/vocab_v1.yaml``, are
committed alongside the annotation guidelines, and changing one is a reviewed decision with
its own entry in ``DECISIONS.md``. See D-032.

**Availability excuses omission.** An Elliptic2 fan-out narrative is not penalised for
failing to mention ``flow.total_outflow``: the substrate has no amounts, so the field is
under a sentinel and no narrative could mention it faithfully. :func:`required_fields`
therefore filters the list against the record before anything is scored, which is why
adequacy is comparable across substrates that support different fact families.

This module computes *which fields are required and which are present*. It does not decide
whether a narrative mentions them — that needs the narrative, and lives with the checker.
"""

from __future__ import annotations

from dataclasses import dataclass

from g2t_aml.facts.schema import CaseFacts, Unavailable
from g2t_aml.facts.vocab import ControlledVocabulary, load_vocabulary

__all__ = [
    "SalienceReport",
    "field_value",
    "is_field_available",
    "required_fields",
    "salience_report",
]


@dataclass(frozen=True)
class SalienceReport:
    """Which salient fields this record can support, and which it cannot.

    Attributes:
        typology: The typology the lists were selected for.
        required: Field paths an adequate narrative must mention, already filtered to
            those the record actually supports.
        excused: Field paths in the typology's list that are under an availability
            sentinel or otherwise absent, and are therefore not required.
    """

    typology: str
    required: tuple[str, ...]
    excused: tuple[str, ...]

    @property
    def coverage_denominator(self) -> int:
        """Return the number of fields adequacy is scored out of.

        Returns:
            The count of required fields.
        """
        return len(self.required)


def field_value(facts: CaseFacts, path: str) -> object:
    """Resolve a dotted field path against a fact record.

    Traverses the dataclass tree, so ``"motifs.fan_out.width"`` reaches the descriptor
    inside the motif result rather than needing the serialised dict.

    Args:
        facts: The record to read.
        path: A dotted path such as ``"flow.total_inflow"``.

    Returns:
        The value at that path, an :class:`~g2t_aml.facts.schema.Unavailable` when any
        ancestor is a sentinel, or None when the path does not exist. None and a sentinel
        are deliberately different returns: the first is a broken path (a bug), the
        second is a masked fact (a fact).
    """
    current: object = facts
    for part in path.split("."):
        if isinstance(current, Unavailable):
            return current
        if hasattr(current, part):
            current = getattr(current, part)
            continue
        # Motif descriptors live in a dict on MotifResult rather than as attributes.
        descriptors = getattr(current, "descriptors", None)
        if isinstance(descriptors, dict) and part in descriptors:
            current = descriptors[part]
            continue
        return None
    return current


def is_field_available(facts: CaseFacts, path: str) -> bool:
    """Report whether a record supports a claim about a field at all.

    Args:
        facts: The record to read.
        path: A dotted field path.

    Returns:
        False when the path is missing, sits under an availability sentinel, or resolves
        to ``None`` — the last because a null descriptor means the motif is absent, and a
        narrative cannot be required to mention a width that does not exist.
    """
    value = field_value(facts, path)
    return value is not None and not isinstance(value, Unavailable)


def required_fields(
    facts: CaseFacts, vocabulary: ControlledVocabulary | None = None
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split a typology's salience list into required and excused fields.

    Args:
        facts: The record whose typology selects the list and whose availability filters
            it.
        vocabulary: The controlled vocabulary. Loaded from disk when omitted.

    Returns:
        ``(required, excused)`` field paths, each in the vocabulary's declared order.
    """
    vocab = vocabulary if vocabulary is not None else load_vocabulary()
    declared = vocab.salient_fields(facts.typology.label)
    required = tuple(p for p in declared if is_field_available(facts, p))
    excused = tuple(p for p in declared if not is_field_available(facts, p))
    return required, excused


def salience_report(
    facts: CaseFacts, vocabulary: ControlledVocabulary | None = None
) -> SalienceReport:
    """Build the salience report for one record.

    Args:
        facts: The record to report on.
        vocabulary: The controlled vocabulary. Loaded from disk when omitted.

    Returns:
        The report, carrying the typology, the required fields and the excused ones.
    """
    required, excused = required_fields(facts, vocabulary)
    return SalienceReport(typology=facts.typology.label, required=required, excused=excused)
