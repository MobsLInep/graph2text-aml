"""Turn a narrative's slot annotations back into checkable claims.

This is the hinge between generation and measurement, and the direction it runs in is the
whole point.

**Claims are parsed out of the rendered text. They are never read from the record.** A
slot carries both ``rendered_value`` (the string in the narrative) and ``raw_value`` (what
the record held before formatting); only the first is used here. Building a claim from
``raw_value`` would compare the fact record against itself and report every corpus ever
generated as 100% SUPPORTED — which is precisely the circularity Phase 3 discovered the
hard way, when three injected extractor bugs left the round trip at 100% because the probe
rendered its claims from the record it was verifying against (D-034).

Parsing the text instead makes the check real. If ``format_money`` dropped a thousands
factor, if a duration were rendered in hours and read as days, if the phase display map
were not a bijection — each of those produces a claim that disagrees with the record, and
the case comes back CONTRADICTED. Bronze passing at 100% is then a measurement rather than
a tautology.

The one thing this module may consult the record for is the *inverse of a display map*:
``"fan-out"`` has to become ``"fan_out"`` before ``check_typology`` can compare it, and
``"the originating account"`` has to become ``"originator"``. Those maps are bijections
over closed vocabularies, asserted as such by the template tests, so inverting one recovers
information from the text rather than importing it from the record.
"""

from __future__ import annotations

from g2t_aml.corpus.bronze.format import (
    FormatError,
    parse_count,
    parse_density,
    parse_duration,
    parse_money,
    parse_percent,
    parse_timestamp,
)
from g2t_aml.corpus.bronze.templates import PHASE_DISPLAY, ROLE_DISPLAY, TYPOLOGY_DISPLAY
from g2t_aml.corpus.record import SlotAnnotation
from g2t_aml.facts.checkers import Claim, ClaimType, DurationClaim

__all__ = ["ClaimParseError", "claim_from_slot", "claims_from_slots"]

#: Inverse of the display maps, built once. A collision here would mean a surface form
#: that maps back to two vocabulary members, which would make the parse ambiguous; the
#: builders below raise rather than pick one.
_ROLE_INVERSE: dict[str, str] = {}
_PHASE_INVERSE: dict[str, str] = {}
_TYPOLOGY_INVERSE: dict[str, str] = {}


def _invert(mapping: dict[str, tuple[str, ...] | str], target: dict[str, str], what: str) -> None:
    """Populate an inverse map, refusing to build an ambiguous one.

    Args:
        mapping: Vocabulary member to its surface form or forms.
        target: The inverse map to fill.
        what: Noun for the error message.

    Raises:
        ValueError: If two members share a surface form.
    """
    for member, forms in mapping.items():
        for form in (forms,) if isinstance(forms, str) else forms:
            if form in target and target[form] != member:
                raise ValueError(
                    f"{what} surface form {form!r} maps back to both {target[form]!r} "
                    f"and {member!r}; a display map must be a bijection or a claim "
                    "parsed from the text is ambiguous"
                )
            target[form] = member


_invert(dict(ROLE_DISPLAY), _ROLE_INVERSE, "role")
_invert(dict(PHASE_DISPLAY), _PHASE_INVERSE, "phase")
_invert(dict(TYPOLOGY_DISPLAY), _TYPOLOGY_INVERSE, "typology")

#: Field paths whose categorical surface form needs inverting before comparison.
_ROLE_PATHS = frozenset({"focal_entity.role"})
_TYPOLOGY_PATHS = frozenset({"typology.label"})
_ORDERING_PATHS = frozenset({"temporal.event_ordering"})

#: Field paths carrying a timestamp rather than a duration, both of which are TEMPORAL.
_TIMESTAMP_PATHS = frozenset(
    {
        "temporal.window_start",
        "temporal.window_end",
        "temporal.burst_start",
        "focal_entity.first_seen",
        "focal_entity.last_seen",
    }
)

#: Numeric field paths rendered as a percentage rather than as a count or an amount.
_SHARE_PATHS = frozenset({"labels.illicit_inflow_share"})

#: Numeric field paths rendered as a bare decimal.
_DECIMAL_PATHS = frozenset(
    {"structure.density", "structure.reciprocity", "motifs.bipartite.score", "typology.confidence"}
)

#: Field paths whose value is a monetary amount with its currency.
_MONEY_PATHS = frozenset(
    {
        "flow.total_inflow",
        "flow.total_outflow",
        "flow.retained",
        "flow.max_single_transfer",
    }
)


class ClaimParseError(ValueError):
    """Raised when a slot's rendered text cannot be read back as a claim.

    Always a bug in the renderer or the formatter, never in the data: every slot in a
    Bronze narrative was written by code in this package, so a string it cannot read back
    is a string it should not have written.
    """


def claim_from_slot(slot: SlotAnnotation, narrative: str) -> Claim:
    """Build one checkable claim from one slot annotation.

    Args:
        slot: The annotation.
        narrative: The narrative the span indexes into. The claim's value is parsed from
            ``narrative[span]``, and a disagreement between that and
            ``slot.rendered_value`` is itself an error — a slot whose span has drifted is
            a slot that would align a claim to the wrong words in Phase 10.

    Returns:
        The claim, with the tolerance rule selected by the slot's claim type and the
        field it names.

    Raises:
        ClaimParseError: If the span does not hold the rendered value, or the rendered
            value cannot be parsed under the slot's claim type.
    """
    start, end = slot.span
    actual = narrative[start:end]
    if actual != slot.rendered_value:
        raise ClaimParseError(
            f"slot for {slot.field_path!r} claims span {slot.span} holds "
            f"{slot.rendered_value!r}, but the narrative holds {actual!r}"
        )
    path = slot.field_path
    text = slot.rendered_value

    try:
        value, claim_type = _parse(path, text, slot.claim_type)
    except (FormatError, KeyError, ValueError) as exc:
        raise ClaimParseError(
            f"cannot read {text!r} back as a claim about {path!r} " f"({slot.claim_type}): {exc}"
        ) from exc

    return Claim(
        text_span=(start, end),
        field_path=None if claim_type in (ClaimType.ENTITY, ClaimType.QUALITATIVE) else path,
        claim_type=claim_type,
        value=value,
        raw_text=text,
    )


def _parse(  # noqa: PLR0911, PLR0912 -- one branch per claim type and per specially
    # formatted field. Collapsing them behind a dispatch table would hide which field
    # parses which way, and that mapping is the thing a reader needs to check.
    path: str,
    text: str,
    declared: str,
) -> tuple[object, ClaimType]:
    """Parse a rendered value into a claim value and its type.

    Args:
        path: The field path the slot names.
        text: The rendered text.
        declared: The slot's declared claim type.

    Returns:
        ``(value, claim_type)``.

    Raises:
        FormatError: If the text does not parse under the expected format.
        KeyError: If a surface form is outside its display map.
        ValueError: If the declared claim type is unknown.
    """
    if declared == "entity":
        return text, ClaimType.ENTITY
    if declared == "qualitative":
        return text, ClaimType.QUALITATIVE
    if declared == "regulatory":
        return text, ClaimType.REGULATORY

    if declared == "temporal":
        if path in _TIMESTAMP_PATHS:
            return parse_timestamp(text), ClaimType.TEMPORAL
        if path in _ORDERING_PATHS:
            return [_PHASE_INVERSE[p] for p in text.split(", then ")], ClaimType.TEMPORAL
        value, unit = parse_duration(text)
        return DurationClaim(value=value, unit=unit), ClaimType.TEMPORAL

    if declared == "numeric":
        if path in _MONEY_PATHS:
            return parse_money(text), ClaimType.NUMERIC
        if path in _SHARE_PATHS:
            return parse_percent(text), ClaimType.NUMERIC
        if path in _DECIMAL_PATHS:
            return parse_density(text), ClaimType.NUMERIC
        return parse_count(text), ClaimType.NUMERIC

    if declared == "categorical":
        if path in _ROLE_PATHS:
            return _ROLE_INVERSE[text], ClaimType.CATEGORICAL
        if path in _TYPOLOGY_PATHS:
            return _TYPOLOGY_INVERSE[text], ClaimType.CATEGORICAL
        if text in ("detected", "present", "yes"):
            return True, ClaimType.CATEGORICAL
        if text in ("not detected", "absent", "no"):
            return False, ClaimType.CATEGORICAL
        return text, ClaimType.CATEGORICAL

    raise ValueError(f"unknown claim type {declared!r} on a slot for {path!r}")


def claims_from_slots(
    slots: tuple[SlotAnnotation, ...] | list[SlotAnnotation], narrative: str
) -> list[Claim]:
    """Build every claim a narrative's slots make.

    Args:
        slots: The annotations.
        narrative: The narrative they index into.

    Returns:
        The claims, in document order.

    Raises:
        ClaimParseError: On the first slot that cannot be read back.
    """
    return [claim_from_slot(slot, narrative) for slot in slots]
