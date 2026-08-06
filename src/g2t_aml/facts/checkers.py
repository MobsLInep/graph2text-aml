"""Three-valued verification: does this claim hold against the fact record?

The reverse direction of the fact layer. Every checker here verifies claims against the
same field paths :mod:`g2t_aml.facts.extractor` writes, with the same semantics, which is
what makes the corpus generator and the faithfulness metric one instrument rather than two
that happen to agree.

**Why three values and not two.** A binary faithful/unfaithful split has to put "the
narrative says the account received USD 480,000 and it received USD 480,000" and "the
narrative says the account is registered in Panama, about which the substrate is silent"
somewhere, and both available answers are wrong. Calling the second faithful licenses every
unsupported assertion a model can invent; calling it unfaithful equates a hedge with a
falsehood and makes the metric punish appropriate caution.

**UNVERIFIABLE is the diagnostically most valuable bucket**, and it must never be collapsed
into either neighbour. It collects exactly the compliance-dangerous claims: assertions about
masked facts, unsupported attributions, vague intensifiers that resolve to no measurement.
A system with high SUPPORTED and high UNVERIFIABLE is not a good system — it is one that
has learned to say impressive things the graph cannot back, which is the failure mode this
whole project exists to detect. Reported separately, always.

**Tolerance policy**, published in the paper and implemented in
:class:`~g2t_aml.facts.config.ToleranceConfig`:

===================  ==========================================================
Claim type           Tolerance
===================  ==========================================================
Counts               Exact.
Monetary amounts     Within 1% relative, with an absolute floor for tiny values.
Durations            Within one unit of the granularity the CLAIM states.
Categorical          Exact, against the controlled vocabulary.
Qualitative          Resolved through the risk-descriptor binding table, else
                     UNVERIFIABLE.
Entity references    Must appear in ``entity_inventory``, else H1.
Regulatory           Must match a whitelisted reference, else H6.
===================  ==========================================================

**Leniency is a bug, not a convenience.** A checker that returns SUPPORTED because a claim
was hard to resolve inflates the headline number by exactly the amount it gives away. When
a claim cannot be verified the answer is UNVERIFIABLE, and the test suite asserts that on
the boundary cases where the temptation is strongest.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from g2t_aml.facts.config import FactConfig, ToleranceConfig
from g2t_aml.facts.salience import field_value
from g2t_aml.facts.schema import CaseFacts, Money, Unavailable
from g2t_aml.facts.taxonomy import HallucinationClass
from g2t_aml.facts.vocab import ControlledVocabulary, load_vocabulary

__all__ = [
    "CHECKER_REGISTRY",
    "CheckContext",
    "CheckResult",
    "Claim",
    "ClaimType",
    "Verdict",
    "check_claim",
    "check_narrative_text",
    "checkable_field_paths",
    "register",
    "summarise",
]


class Verdict(str, Enum):
    """The three possible outcomes of checking a claim."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"


class ClaimType(str, Enum):
    """What kind of claim this is, which selects the tolerance rule applied."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    QUALITATIVE = "qualitative"
    ENTITY = "entity"
    REGULATORY = "regulatory"


@dataclass(frozen=True)
class Claim:
    """One assertion extracted from a narrative.

    Attributes:
        text_span: ``(start, end)`` character offsets into the narrative, so a finding
            can be shown in place rather than merely counted.
        field_path: The fact field the claim is about, e.g. ``"motifs.fan_out.width"``.
            None for claims that reference no field — a forbidden phrase, an entity
            mention.
        claim_type: Selects the tolerance rule.
        value: The value asserted. Its type depends on ``claim_type``: a number for
            NUMERIC, a string for CATEGORICAL and ENTITY, a
            :class:`~g2t_aml.facts.checkers.DurationClaim` or datetime for TEMPORAL.
        raw_text: The narrative text the claim was read from.
    """

    text_span: tuple[int, int]
    field_path: str | None
    claim_type: ClaimType
    value: Any
    raw_text: str


@dataclass(frozen=True)
class DurationClaim:
    """A duration together with the granularity the narrative stated it at.

    Carrying the granularity is what makes the duration tolerance honest. "About three
    days" claims a precision of one day and is SUPPORTED by anything from 60 to 84 hours;
    "76 hours" claims a precision of one hour and is CONTRADICTED by 80. A checker that
    imposed a single tolerance would either punish appropriate vagueness or wave through
    a genuine error, depending on which it chose.

    Attributes:
        value: The stated duration, in :attr:`unit`.
        unit: ``"hours"``, ``"days"`` or ``"minutes"``.
    """

    value: float
    unit: str

    def to_hours(self) -> float:
        """Return the duration in hours.

        Returns:
            The value converted to hours.

        Raises:
            ValueError: If the unit is not one of the three recognised.
        """
        factors = {"minutes": 1 / 60, "hours": 1.0, "days": 24.0}
        if self.unit not in factors:
            raise ValueError(f"unknown duration unit {self.unit!r}; expected {sorted(factors)}")
        return self.value * factors[self.unit]

    def tolerance_hours(self, tolerance: ToleranceConfig) -> float:
        """Return the permitted error, one unit of the stated granularity.

        Args:
            tolerance: The published tolerance policy.

        Returns:
            The half-width of the accepted interval, in hours.

        Raises:
            ValueError: If the unit is not recognised.
        """
        factors = {"minutes": 1 / 60, "hours": 1.0, "days": 24.0}
        if self.unit not in factors:
            raise ValueError(f"unknown duration unit {self.unit!r}; expected {sorted(factors)}")
        return tolerance.duration_granularity_units * factors[self.unit]


@dataclass(frozen=True)
class CheckResult:
    """The verdict on one claim, with the reason and, when adverse, its class.

    Attributes:
        claim: The claim checked.
        verdict: SUPPORTED, CONTRADICTED or UNVERIFIABLE.
        expected: What the record actually says, or None when it says nothing.
        reason: Human-readable explanation, always populated. A verdict without a reason
            cannot be adjudicated by a human annotator, and Phase 6 needs to.
        hallucination_class: The class, when the verdict is adverse. None for SUPPORTED.
        producer: The graph computation that produced ``expected``, from
            ``provenance.field_producers``. Lets a systematic disagreement be traced to
            one computation rather than to the record as a whole.
    """

    claim: Claim
    verdict: Verdict
    expected: Any = None
    reason: str = ""
    hallucination_class: str | None = None
    producer: str | None = None

    @property
    def is_critical(self) -> bool:
        """Report whether this result counts toward the Critical Error Rate.

        Returns:
            True when the hallucination class is H4, H6 or H7.
        """
        if self.hallucination_class is None:
            return False
        return HallucinationClass[self.hallucination_class].is_critical


@dataclass(frozen=True)
class CheckContext:
    """Everything a checker may read.

    Attributes:
        facts: The fact record the claims are checked against.
        config: Thresholds and the tolerance policy.
        vocabulary: The controlled vocabulary.
    """

    facts: CaseFacts
    config: FactConfig = field(default_factory=FactConfig)
    vocabulary: ControlledVocabulary = field(default_factory=load_vocabulary)

    @property
    def tolerance(self) -> ToleranceConfig:
        """Return the published tolerance policy.

        Returns:
            The policy from the configuration.
        """
        return self.config.tolerance

    def producer_of(self, path: str | None) -> str | None:
        """Return the computation that produced a field.

        Args:
            path: A field path, or None.

        Returns:
            The producer name from ``provenance.field_producers``, or None.
        """
        if path is None or self.facts.provenance is None:
            return None
        return self.facts.provenance.field_producers.get(path)


#: Field path to the checker that verifies it. Populated by :func:`register` at import
#: time; ``tests/unit/test_checker_coverage.py`` asserts every checkable schema field has
#: an entry, so a field added to the schema without a checker fails the suite rather than
#: silently becoming unverifiable.
#: Length of a ``(value, currency)`` claim pair.
_MONEY_PAIR_LEN = 2

#: Timestamp tolerance, in seconds. One minute, which is AMLworld's own resolution:
#: a narrative cannot be expected to be more precise than the substrate is.
_TIMESTAMP_TOLERANCE_SECONDS = 60

CHECKER_REGISTRY: dict[str, Callable[[Claim, CheckContext], CheckResult]] = {}


def register(
    *paths: str,
) -> Callable[
    [Callable[[Claim, CheckContext], CheckResult]], Callable[[Claim, CheckContext], CheckResult]
]:
    """Register a checker for one or more field paths.

    Args:
        *paths: The field paths this checker verifies.

    Returns:
        A decorator that records the function and returns it unchanged.

    Raises:
        ValueError: If a path already has a checker. Two checkers for one field would
            make the verdict depend on registration order.
    """

    def decorate(
        fn: Callable[[Claim, CheckContext], CheckResult],
    ) -> Callable[[Claim, CheckContext], CheckResult]:
        for path in paths:
            if path in CHECKER_REGISTRY:
                raise ValueError(f"field path {path!r} already has a registered checker")
            CHECKER_REGISTRY[path] = fn
        return fn

    return decorate


# ------------------------------------------------------------- comparators ---


def _result(
    claim: Claim,
    ctx: CheckContext,
    verdict: Verdict,
    expected: Any,
    reason: str,
    hallucination_class: str | None = None,
) -> CheckResult:
    """Build a result, attaching the field's producer.

    Args:
        claim: The claim checked.
        ctx: The checking context.
        verdict: The outcome.
        expected: What the record says.
        reason: Why.
        hallucination_class: The class, when adverse.

    Returns:
        The populated result.
    """
    return CheckResult(
        claim=claim,
        verdict=verdict,
        expected=expected,
        reason=reason,
        hallucination_class=hallucination_class,
        producer=ctx.producer_of(claim.field_path),
    )


def _unverifiable(claim: Claim, ctx: CheckContext, expected: Any, reason: str) -> CheckResult:
    """Build an UNVERIFIABLE result.

    Every unverifiable claim is H8 (unsupported inference) unless a more specific checker
    overrides it: the narrative asserted something the record cannot speak to.

    Args:
        claim: The claim checked.
        ctx: The checking context.
        expected: The sentinel or absence encountered.
        reason: Why the claim could not be resolved.

    Returns:
        The populated result.
    """
    return _result(claim, ctx, Verdict.UNVERIFIABLE, expected, reason, "H8")


def _resolve(claim: Claim, ctx: CheckContext) -> tuple[Any, CheckResult | None]:
    """Resolve a claim's field, or return the UNVERIFIABLE result that replaces it.

    The single gate through which every field-based checker passes. Centralising it is
    what guarantees a masked fact is never CONTRADICTED: a sentinel means the substrate
    cannot speak, and a narrative cannot contradict silence.

    Args:
        claim: The claim, whose ``field_path`` is resolved.
        ctx: The checking context.

    Returns:
        ``(value, None)`` when the field resolves to something checkable, or
        ``(None, result)`` when it does not.
    """
    if claim.field_path is None:
        return None, _unverifiable(claim, ctx, None, "claim references no fact field")
    value = field_value(ctx.facts, claim.field_path)
    if isinstance(value, Unavailable):
        return None, _unverifiable(
            claim,
            ctx,
            value.to_dict(),
            f"{claim.field_path} is unavailable for this substrate ({value.reason}); a "
            "narrative cannot contradict a fact the data does not carry",
        )
    if value is None:
        return None, _unverifiable(
            claim, ctx, None, f"{claim.field_path} is not present in this record"
        )
    return value, None


def _numeric_verdict(
    claim: Claim, ctx: CheckContext, actual: float, stated: float, tolerance: float, kind: str
) -> CheckResult:
    """Compare two numbers under an absolute tolerance.

    Args:
        claim: The claim checked.
        ctx: The checking context.
        actual: The recorded value.
        stated: The asserted value.
        tolerance: Permitted absolute difference.
        kind: Noun used in the reason text, e.g. ``"count"``.

    Returns:
        SUPPORTED within tolerance, CONTRADICTED (H2) otherwise.
    """
    if abs(stated - actual) <= tolerance:
        return _result(claim, ctx, Verdict.SUPPORTED, actual, f"{kind} {stated} matches {actual}")
    return _result(
        claim,
        ctx,
        Verdict.CONTRADICTED,
        actual,
        f"{kind} stated as {stated}, record says {actual} " f"(tolerance {tolerance})",
        "H2",
    )


def _as_number(value: Any) -> float | None:
    """Coerce a claim value to a float, without accepting a bool.

    ``True`` is an ``int`` in Python, and silently reading it as 1 would let a
    categorical claim be checked as a numeric one.

    Args:
        value: The claimed value.

    Returns:
        The number, or None when the value is not numeric.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


# ------------------------------------------------------- generic checkers ---


def check_count(claim: Claim, ctx: CheckContext) -> CheckResult:
    """Verify an exact-count claim.

    Counts get no tolerance. "Nine accounts" against eight is wrong in a way no rounding
    convention redeems.

    Args:
        claim: A NUMERIC claim about a count field.
        ctx: The checking context.

    Returns:
        SUPPORTED on an exact match, CONTRADICTED (H2) otherwise, UNVERIFIABLE when the
        field is masked or absent.
    """
    actual, early = _resolve(claim, ctx)
    if early is not None:
        return early
    stated = _as_number(claim.value)
    if stated is None:
        return _unverifiable(claim, ctx, actual, f"claim value {claim.value!r} is not a number")
    return _numeric_verdict(claim, ctx, float(actual), stated, 0.0, "count")


def check_ratio(claim: Claim, ctx: CheckContext) -> CheckResult:
    """Verify a share, ratio or bounded score.

    Args:
        claim: A NUMERIC claim about a [0, 1] field.
        ctx: The checking context.

    Returns:
        SUPPORTED within ``share_absolute``, CONTRADICTED (H2) otherwise.
    """
    actual, early = _resolve(claim, ctx)
    if early is not None:
        return early
    stated = _as_number(claim.value)
    if stated is None:
        return _unverifiable(claim, ctx, actual, f"claim value {claim.value!r} is not a number")
    return _numeric_verdict(
        claim, ctx, float(actual), stated, ctx.tolerance.share_absolute, "share"
    )


def check_money(claim: Claim, ctx: CheckContext) -> CheckResult:
    """Verify a monetary claim under the 1% relative tolerance.

    A claim may state a bare number or a ``(value, currency)`` pair. When it states a
    currency and the record disagrees, the verdict is CONTRADICTED regardless of the
    amount: "USD 480,000" against 480,000 Euro is not a rounding difference.

    Args:
        claim: A NUMERIC claim about a monetary field.
        ctx: The checking context.

    Returns:
        SUPPORTED within tolerance, CONTRADICTED (H2) otherwise, UNVERIFIABLE when the
        amount is under a sentinel — which is what a multi-currency aggregate produces.
    """
    actual, early = _resolve(claim, ctx)
    if early is not None:
        return early
    if not isinstance(actual, Money):
        return _unverifiable(
            claim, ctx, actual, f"{claim.field_path} does not hold a monetary amount"
        )

    stated_currency: str | None = None
    stated_value: float | None
    if isinstance(claim.value, tuple | list) and len(claim.value) == _MONEY_PAIR_LEN:
        stated_value = _as_number(claim.value[0])
        stated_currency = str(claim.value[1])
    elif isinstance(claim.value, Money):
        stated_value, stated_currency = claim.value.value, claim.value.currency
    else:
        stated_value = _as_number(claim.value)

    if stated_value is None:
        return _unverifiable(
            claim, ctx, actual.to_dict(), f"claim value {claim.value!r} is not an amount"
        )
    if stated_currency is not None and stated_currency != actual.currency:
        return _result(
            claim,
            ctx,
            Verdict.CONTRADICTED,
            actual.to_dict(),
            f"currency stated as {stated_currency!r}, record says {actual.currency!r}",
            "H2",
        )

    tolerance = max(
        abs(actual.value) * ctx.tolerance.monetary_relative,
        ctx.tolerance.monetary_absolute_floor,
    )
    return _numeric_verdict(claim, ctx, actual.value, stated_value, tolerance, "amount")


def check_duration(claim: Claim, ctx: CheckContext) -> CheckResult:
    """Verify a duration within one unit of the granularity the claim states.

    Args:
        claim: A TEMPORAL claim whose value is a :class:`DurationClaim`, or a bare number
            read as hours.
        ctx: The checking context.

    Returns:
        SUPPORTED within one stated unit, CONTRADICTED (H3) otherwise.
    """
    actual, early = _resolve(claim, ctx)
    if early is not None:
        return early
    actual_hours = _as_number(actual)
    if actual_hours is None:
        return _unverifiable(claim, ctx, actual, f"{claim.field_path} is not a duration")

    if isinstance(claim.value, DurationClaim):
        duration = claim.value
    else:
        stated = _as_number(claim.value)
        if stated is None:
            return _unverifiable(
                claim, ctx, actual, f"claim value {claim.value!r} is not a duration"
            )
        duration = DurationClaim(value=stated, unit="hours")

    tolerance = duration.tolerance_hours(ctx.tolerance)
    stated_hours = duration.to_hours()
    if abs(stated_hours - actual_hours) <= tolerance:
        return _result(
            claim,
            ctx,
            Verdict.SUPPORTED,
            actual_hours,
            f"{duration.value} {duration.unit} ({stated_hours:.4g}h) matches "
            f"{actual_hours:.4g}h within one {duration.unit[:-1]}",
        )
    return _result(
        claim,
        ctx,
        Verdict.CONTRADICTED,
        actual_hours,
        f"duration stated as {duration.value} {duration.unit} ({stated_hours:.4g}h), "
        f"record says {actual_hours:.4g}h (tolerance {tolerance:.4g}h)",
        "H3",
    )


def check_timestamp(claim: Claim, ctx: CheckContext) -> CheckResult:
    """Verify a timestamp claim against a recorded moment.

    Args:
        claim: A TEMPORAL claim whose value is a datetime.
        ctx: The checking context.

    Returns:
        SUPPORTED within one minute — the substrate's own resolution — CONTRADICTED (H3)
        otherwise.
    """
    actual, early = _resolve(claim, ctx)
    if early is not None:
        return early
    if not isinstance(actual, datetime) or not isinstance(claim.value, datetime):
        return _unverifiable(claim, ctx, actual, "claim or record value is not a timestamp")
    delta = abs((claim.value - actual).total_seconds())
    if delta <= _TIMESTAMP_TOLERANCE_SECONDS:
        return _result(
            claim, ctx, Verdict.SUPPORTED, actual.isoformat(), "timestamp matches the record"
        )
    return _result(
        claim,
        ctx,
        Verdict.CONTRADICTED,
        actual.isoformat(),
        f"timestamp stated as {claim.value.isoformat()}, record says {actual.isoformat()}",
        "H3",
    )


def check_boolean(claim: Claim, ctx: CheckContext) -> CheckResult:
    """Verify a boolean claim exactly.

    Args:
        claim: A CATEGORICAL claim whose value is a bool.
        ctx: The checking context.

    Returns:
        SUPPORTED on a match, CONTRADICTED (H2) otherwise.
    """
    actual, early = _resolve(claim, ctx)
    if early is not None:
        return early
    if not isinstance(claim.value, bool):
        return _unverifiable(claim, ctx, actual, f"claim value {claim.value!r} is not a boolean")
    if bool(actual) == claim.value:
        return _result(claim, ctx, Verdict.SUPPORTED, actual, f"presence {actual} matches")
    return _result(
        claim,
        ctx,
        Verdict.CONTRADICTED,
        actual,
        f"stated as {claim.value}, record says {actual}",
        "H2",
    )


def check_categorical(claim: Claim, ctx: CheckContext) -> CheckResult:
    """Verify a categorical claim exactly, against the controlled vocabulary.

    Args:
        claim: A CATEGORICAL claim whose value is a vocabulary member.
        ctx: The checking context.

    Returns:
        SUPPORTED on an exact match, CONTRADICTED (H5) otherwise.
    """
    actual, early = _resolve(claim, ctx)
    if early is not None:
        return early
    stated = str(claim.value)
    if stated == str(actual):
        return _result(claim, ctx, Verdict.SUPPORTED, actual, f"category {stated!r} matches")
    return _result(
        claim,
        ctx,
        Verdict.CONTRADICTED,
        actual,
        f"category stated as {stated!r}, record says {str(actual)!r}",
        "H5",
    )


def check_sequence(claim: Claim, ctx: CheckContext) -> CheckResult:
    """Verify an ordered sequence claim, such as the phase ordering.

    Order is part of the claim: ``[inflow, outflow]`` and ``[outflow, inflow]`` describe
    opposite cases, so the comparison is on the sequence rather than the set.

    Args:
        claim: A TEMPORAL claim whose value is a sequence of phase names.
        ctx: The checking context.

    Returns:
        SUPPORTED on an exact sequence match, CONTRADICTED (H3) otherwise.
    """
    actual, early = _resolve(claim, ctx)
    if early is not None:
        return early
    if not isinstance(claim.value, list | tuple):
        return _unverifiable(claim, ctx, actual, f"claim value {claim.value!r} is not a sequence")
    stated = tuple(str(v) for v in claim.value)
    recorded = tuple(str(v) for v in actual)
    if stated == recorded:
        return _result(claim, ctx, Verdict.SUPPORTED, list(recorded), "ordering matches")
    return _result(
        claim,
        ctx,
        Verdict.CONTRADICTED,
        list(recorded),
        f"ordering stated as {list(stated)}, record says {list(recorded)}",
        "H3",
    )


def check_string_set(claim: Claim, ctx: CheckContext) -> CheckResult:
    """Verify membership in an unordered set field, such as the currencies involved.

    Args:
        claim: A CATEGORICAL claim naming one member.
        ctx: The checking context.

    Returns:
        SUPPORTED when the claimed member is present, CONTRADICTED (H2) otherwise.
    """
    actual, early = _resolve(claim, ctx)
    if early is not None:
        return early
    members = {str(v) for v in actual}
    stated = str(claim.value)
    if stated in members:
        return _result(claim, ctx, Verdict.SUPPORTED, sorted(members), f"{stated!r} is present")
    return _result(
        claim,
        ctx,
        Verdict.CONTRADICTED,
        sorted(members),
        f"{stated!r} does not appear; record has {sorted(members)}",
        "H2",
    )


def check_entity(claim: Claim, ctx: CheckContext) -> CheckResult:
    """Verify that a named entity appears in the case subgraph.

    Args:
        claim: An ENTITY claim whose value is an account identifier.
        ctx: The checking context.

    Returns:
        SUPPORTED when the identifier is in ``entity_inventory.node_ids``, CONTRADICTED
        (H1) otherwise. Never UNVERIFIABLE: the inventory is always complete, so an
        entity is either in the case or fabricated.
    """
    stated = str(claim.value)
    inventory = ctx.facts.entity_inventory.node_ids
    if stated in inventory:
        return _result(
            claim, ctx, Verdict.SUPPORTED, stated, f"{stated!r} appears in the case subgraph"
        )
    return _result(
        claim,
        ctx,
        Verdict.CONTRADICTED,
        list(inventory[:5]),
        f"{stated!r} does not appear among the case's {len(inventory)} accounts",
        "H1",
    )


def check_regulatory(claim: Claim, ctx: CheckContext) -> CheckResult:
    """Verify a regulatory citation against the whitelist.

    Args:
        claim: A REGULATORY claim whose value is a citation phrase or reference id.
        ctx: The checking context.

    Returns:
        SUPPORTED when the citation matches a whitelisted reference, CONTRADICTED (H6)
        otherwise. An invented rule is Critical, so it is never softened to UNVERIFIABLE.
    """
    stated = str(claim.value).strip().lower()
    for reference in ctx.vocabulary.regulatory.values():
        if stated == reference.ident.lower() or stated in reference.phrase_variants:
            return _result(
                claim,
                ctx,
                Verdict.SUPPORTED,
                reference.citation,
                f"matches whitelisted reference {reference.ident!r}",
            )
    return _result(
        claim,
        ctx,
        Verdict.CONTRADICTED,
        sorted(ctx.vocabulary.regulatory),
        f"{stated!r} is not a whitelisted regulatory reference",
        "H6",
    )


def check_qualitative(claim: Claim, ctx: CheckContext) -> CheckResult:
    """Resolve a qualitative intensifier through the risk-descriptor binding table.

    This is what makes "rapid dispersal" checkable. The phrase names a descriptor, the
    descriptor binds to a field and a condition, and the verdict is whether the condition
    holds of the recorded value.

    Args:
        claim: A QUALITATIVE claim whose value is a descriptor name or a surface phrase.
        ctx: The checking context.

    Returns:
        SUPPORTED when the binding's condition holds, CONTRADICTED (H2) when it fails,
        and UNVERIFIABLE when the phrase is not a controlled descriptor, when the
        substrate lacks a required availability flag, or when the bound field is masked
        or null.
    """
    stated = str(claim.value).strip().lower()
    descriptor = ctx.vocabulary.risk_descriptors.get(stated)
    if descriptor is None:
        descriptor = ctx.vocabulary.descriptor_for_phrase(stated)
    if descriptor is None:
        return _unverifiable(
            claim,
            ctx,
            None,
            f"{stated!r} is not a controlled risk descriptor, so it resolves to no " "measurement",
        )

    mask = ctx.facts.availability.to_dict()
    if missing := [f for f in descriptor.requires if not mask.get(f, False)]:
        return _unverifiable(
            claim,
            ctx,
            None,
            f"descriptor {descriptor.name!r} requires {missing}, which this substrate "
            "does not support",
        )

    bound = Claim(
        text_span=claim.text_span,
        field_path=descriptor.binds_to,
        claim_type=ClaimType.NUMERIC,
        value=claim.value,
        raw_text=claim.raw_text,
    )
    value, early = _resolve(bound, ctx)
    if early is not None:
        return _unverifiable(
            claim,
            ctx,
            early.expected,
            f"descriptor {descriptor.name!r} binds to {descriptor.binds_to}, which is "
            f"not measurable here: {early.reason}",
        )
    number = _as_number(value)
    if number is None:
        return _unverifiable(
            claim,
            ctx,
            value,
            f"{descriptor.binds_to} is not numeric, so the binding cannot " "be evaluated",
        )

    holds = descriptor.holds_for(number)
    detail = f"{descriptor.binds_to} = {number} against condition {descriptor.condition!r}"
    if holds:
        return _result(claim, ctx, Verdict.SUPPORTED, number, f"{stated!r} holds: {detail}")
    return _result(
        claim,
        ctx,
        Verdict.CONTRADICTED,
        number,
        f"{stated!r} does not hold: {detail}",
        "H2",
    )


def check_typology(claim: Claim, ctx: CheckContext) -> CheckResult:
    """Verify a typology claim, accounting for how the label was obtained.

    A ground-truth label is checkable exactly. An **inferred** label is not a fact about
    the case, it is this system's own motif detector talking, so asserting it as though
    it were established is H5 even when the label matches. The narrative must hedge, and
    the hedge is checked by :func:`check_narrative_text` against
    ``hedging.required_for_inferred``.

    Args:
        claim: A CATEGORICAL claim naming a typology.
        ctx: The checking context.

    Returns:
        SUPPORTED on a ground-truth match, CONTRADICTED (H5) on a mismatch, and
        UNVERIFIABLE when the label is inferred or absent — the record cannot establish
        what only a detector proposed.
    """
    typology = ctx.facts.typology
    stated = str(claim.value)
    if typology.source == "ground_truth":
        if stated == typology.label:
            return _result(
                claim,
                ctx,
                Verdict.SUPPORTED,
                typology.label,
                f"typology {stated!r} matches the substrate's ground truth",
            )
        return _result(
            claim,
            ctx,
            Verdict.CONTRADICTED,
            typology.label,
            f"typology stated as {stated!r}, ground truth says {typology.label!r}",
            "H5",
        )
    if stated == typology.label:
        return _unverifiable(
            claim,
            ctx,
            typology.label,
            f"typology {stated!r} was inferred from motif detection (confidence "
            f"{typology.confidence}), not established by ground truth",
        )
    return _result(
        claim,
        ctx,
        Verdict.CONTRADICTED,
        typology.label,
        f"typology stated as {stated!r}; motif detection inferred {typology.label!r}",
        "H5",
    )


def check_unavailable_only(claim: Claim, ctx: CheckContext) -> CheckResult:
    """Reject any claim about a permanently unavailable field.

    Registered for ``flow.cross_border``, which no substrate can ever license. The field
    exists in the schema precisely so this checker can fire: without it a cross-border
    assertion would match no field and be quietly dropped instead of counted.

    Args:
        claim: Any claim about the field.
        ctx: The checking context.

    Returns:
        Always UNVERIFIABLE, with the sentinel's reason.
    """
    value = field_value(ctx.facts, claim.field_path or "")
    reason = value.reason if isinstance(value, Unavailable) else "field is never available"
    return _unverifiable(
        claim,
        ctx,
        value.to_dict() if isinstance(value, Unavailable) else None,
        f"{claim.field_path} can never be established on any substrate ({reason})",
    )


# --------------------------------------------------------------- registry ---

_COUNT_FIELDS = (
    "structure.n_nodes",
    "structure.n_edges",
    "structure.n_components",
    "structure.diameter",
    "structure.max_in_degree",
    "structure.max_out_degree",
    "structure.n_self_loops",
    "focal_entity.in_degree",
    "focal_entity.out_degree",
    "focal_entity.n_transactions_in",
    "focal_entity.n_transactions_out",
    "temporal.burst_txn_count",
    "temporal.n_transactions",
    "flow.n_transfers_near_threshold",
    "labels.n_illicit_counterparties",
    "labels.n_licit_counterparties",
    "labels.n_unknown_counterparties",
    "labels.n_counterparties",
    "labels.min_hops_to_known_illicit",
    "labels.n_illicit_transactions",
    "motifs.fan_in.width",
    "motifs.fan_out.width",
    "motifs.chain.max_length",
    "motifs.cycle.length",
    "motifs.bipartite.left_size",
    "motifs.bipartite.right_size",
    "motifs.stack.depth",
    "motifs.gather_scatter.gather_width",
    "motifs.gather_scatter.scatter_width",
    "motifs.scatter_gather.width",
)

_RATIO_FIELDS = (
    "structure.density",
    "structure.reciprocity",
    "labels.illicit_inflow_share",
    "motifs.bipartite.score",
    "typology.confidence",
    "model_signal.gnn_risk_score",
    "model_signal.score_percentile",
)

_MONEY_FIELDS = (
    "flow.total_inflow",
    "flow.total_outflow",
    "flow.retained",
    "flow.max_single_transfer",
    "flow.threshold_reference",
)

_DURATION_FIELDS = (
    "temporal.span_hours",
    "temporal.burst_window_hours",
    "motifs.fan_in.window_hours",
    "motifs.fan_out.window_hours",
)

_TIMESTAMP_FIELDS = (
    "temporal.window_start",
    "temporal.window_end",
    "temporal.burst_start",
    "focal_entity.first_seen",
    "focal_entity.last_seen",
)

_BOOLEAN_FIELDS = (
    "temporal.burst_detected",
    "flow.cross_institution",
    "labels.focal_is_illicit",
    "motifs.fan_in.present",
    "motifs.fan_out.present",
    "motifs.chain.present",
    "motifs.cycle.present",
    "motifs.bipartite.present",
    "motifs.stack.present",
    "motifs.gather_scatter.present",
    "motifs.scatter_gather.present",
)

_ENTITY_FIELDS = (
    "entity_inventory.focal_id",
    "focal_entity.id",
    "motifs.fan_in.hub",
    "motifs.fan_out.hub",
    "motifs.gather_scatter.hub",
    "motifs.scatter_gather.origin",
    "motifs.scatter_gather.destination",
)

_CATEGORICAL_FIELDS = (
    "focal_entity.role",
    "focal_entity.selection_rule",
    "typology.source",
    "typology.scope",
    "flow.threshold_currency",
    "model_signal.model_version",
)

_SET_FIELDS = (
    "flow.currencies_involved",
    "flow.payment_formats",
    "entity_inventory.node_ids",
    "motifs.stack.layer_widths",
    "flow.inflow_by_currency",
    "flow.outflow_by_currency",
    "model_signal.top_contributing_nodes",
)

register(*_COUNT_FIELDS)(check_count)
register(*_RATIO_FIELDS)(check_ratio)
register(*_MONEY_FIELDS)(check_money)
register(*_DURATION_FIELDS)(check_duration)
register(*_TIMESTAMP_FIELDS)(check_timestamp)
register(*_BOOLEAN_FIELDS)(check_boolean)
register(*_ENTITY_FIELDS)(check_entity)
register(*_CATEGORICAL_FIELDS)(check_categorical)
register(*_SET_FIELDS)(check_string_set)
register("temporal.event_ordering")(check_sequence)
register("typology.label")(check_typology)
register("flow.cross_border")(check_unavailable_only)
register("flow.n_distinct_banks")(check_count)
register("flow.threshold_band_fraction")(check_ratio)


def checkable_field_paths() -> tuple[str, ...]:
    """Return every field path with a registered checker.

    Returns:
        The paths, sorted.
    """
    return tuple(sorted(CHECKER_REGISTRY))


# ------------------------------------------------------------- entrypoints ---


def check_claim(claim: Claim, ctx: CheckContext) -> CheckResult:
    """Check one claim against the fact record.

    Dispatches on ``claim.claim_type`` first for the types that do not name a field
    (QUALITATIVE, ENTITY, REGULATORY), then on ``field_path`` through the registry.

    Args:
        claim: The claim to check.
        ctx: The checking context.

    Returns:
        The verdict. UNVERIFIABLE when the claim names a field with no registered
        checker — never SUPPORTED, because an unrecognised claim has not been verified
        and reporting it as verified is exactly the leniency this module forbids.
    """
    if claim.claim_type is ClaimType.QUALITATIVE:
        return check_qualitative(claim, ctx)
    if claim.claim_type is ClaimType.REGULATORY:
        return check_regulatory(claim, ctx)
    if claim.claim_type is ClaimType.ENTITY and claim.field_path is None:
        return check_entity(claim, ctx)

    if claim.field_path is None:
        return _unverifiable(claim, ctx, None, "claim names no fact field")
    checker = CHECKER_REGISTRY.get(claim.field_path)
    if checker is None:
        return _unverifiable(
            claim,
            ctx,
            None,
            f"no checker is registered for {claim.field_path!r}, so the claim is " "unverified",
        )
    return checker(claim, ctx)


def check_narrative_text(text: str, ctx: CheckContext) -> list[CheckResult]:
    """Scan raw narrative text for forbidden phrases and missing hedges.

    Complements the claim-level checks: a narrative can be arithmetically perfect and
    still assert guilt, name a business type, or claim to describe a complete scheme.
    Those are properties of the *text*, not of any single extracted claim, and this is
    where they are caught.

    Args:
        text: The narrative.
        ctx: The checking context.

    Returns:
        One CONTRADICTED result per forbidden-phrase hit, plus one when the record's
        typology is inferred and the text carries none of the required hedges. Empty when
        the text is clean.
    """
    results: list[CheckResult] = []
    haystack = text.lower()

    for group, (hallucination_class, phrases) in ctx.vocabulary.forbidden.items():
        for phrase in sorted(phrases, key=len, reverse=True):
            start = haystack.find(phrase)
            if start < 0:
                continue
            claim = Claim(
                text_span=(start, start + len(phrase)),
                field_path=None,
                claim_type=ClaimType.QUALITATIVE,
                value=phrase,
                raw_text=text[start : start + len(phrase)],
            )
            results.append(
                _result(
                    claim,
                    ctx,
                    Verdict.CONTRADICTED,
                    None,
                    f"forbidden phrase {phrase!r} ({group}) is outside the controlled "
                    "vocabulary",
                    hallucination_class,
                )
            )
            break  # one finding per group is enough to fail it

    if (
        ctx.facts.typology.source == "inferred"
        and ctx.facts.typology.label != "unclassified"
        and not any(hedge in haystack for hedge in ctx.vocabulary.required_for_inferred)
    ):
        unhedged = Claim(
            text_span=(0, len(text)),
            field_path="typology.label",
            claim_type=ClaimType.CATEGORICAL,
            value=ctx.facts.typology.label,
            raw_text=text,
        )
        results.append(
            _result(
                unhedged,
                ctx,
                Verdict.CONTRADICTED,
                ctx.facts.typology.label,
                "typology was inferred from motif detection but the narrative carries "
                "none of the required hedges "
                f"({list(ctx.vocabulary.required_for_inferred)})",
                "H5",
            )
        )
    return results


def summarise(results: list[CheckResult]) -> dict[str, Any]:
    """Aggregate check results into the numbers reported in the paper.

    The critical-error rate is computed over H4/H6/H7 and reported **separately** from
    faithfulness, for the reasons in :mod:`g2t_aml.facts.taxonomy`.

    Args:
        results: Every result for one narrative, or for a whole evaluation set.

    Returns:
        Counts and rates by verdict, the per-class breakdown, and the critical-error
        count and rate. Rates are 0.0 over an empty input rather than undefined, so an
        aggregation over a case with no claims does not raise.
    """
    total = len(results)
    by_verdict = {v.value: sum(1 for r in results if r.verdict is v) for v in Verdict}
    by_class: dict[str, int] = {}
    for result in results:
        if result.hallucination_class is not None:
            by_class[result.hallucination_class] = by_class.get(result.hallucination_class, 0) + 1
    critical = sum(1 for r in results if r.is_critical)
    return {
        "n_claims": total,
        "by_verdict": by_verdict,
        "by_hallucination_class": dict(sorted(by_class.items())),
        "supported_rate": by_verdict[Verdict.SUPPORTED.value] / total if total else 0.0,
        "contradicted_rate": by_verdict[Verdict.CONTRADICTED.value] / total if total else 0.0,
        "unverifiable_rate": by_verdict[Verdict.UNVERIFIABLE.value] / total if total else 0.0,
        "n_critical": critical,
        "critical_error_rate": critical / total if total else 0.0,
    }
