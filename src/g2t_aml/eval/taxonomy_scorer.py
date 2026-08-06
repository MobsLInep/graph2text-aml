"""Which *kind* of error, not merely how many. H1—H9 and the Critical Error Rate.

Every CONTRADICTED and UNVERIFIABLE claim is assigned one of the nine classes defined in
:mod:`g2t_aml.facts.taxonomy`, and the per-class rates are cross-tabulated against
typology and system. **That table is the most quotable thing this project produces**: the
existing agentic-SAR literature reports aggregate quality and does not report what its
systems get wrong, so a per-class error breakdown across sixteen systems is a
contribution on its own.

**The Critical Error Rate is reported separately and never folded into faithfulness.**
H4 (attribution fabrication), H6 (regulatory fabrication) and H7 (guilt overclaim) share
a mechanism — each is an assertion the substrate cannot license at all, rather than a
value read off it wrongly — and they share a consequence: a narrative carrying one, filed
as-is, exposes the institution. A mean that puts them beside a rounding error lets a
system with a 2% critical rate look identical to one with 0%, and those are not the same
product.

**Most classes are assigned by the checker, not here.** ``check_entity`` already returns
H1, ``check_typology`` H5, the forbidden-phrase scan H4/H6/H7. This module refines the
residue — the generic H8 the checker returns when nothing more specific applies — and
adds the one class no claim-level check can produce.

**H9 is detected by absence, which is why it needs its own pass.** Omitting a fact that
weakens the suspicion is a real failure of a SAR narrative and there is no claim to
attach it to: the narrative's sin is the sentence it did not write. :func:`omissions`
walks a small, explicit table of exculpatory conditions and fires only when the record
carries one and the narrative says nothing about it.

**The automatic assignment is validated by hand.** :func:`sample_for_hand_labelling`
draws errors for a human to classify and :func:`validate_against_hand_labels` reports the
agreement. An automatic classifier whose agreement with a human has not been measured
produces a table of numbers, not a table of findings.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from g2t_aml.eval.claim_extraction.agreement import cohens_kappa, interpret_kappa
from g2t_aml.eval.layer2_faithfulness import CaseFaithfulness
from g2t_aml.eval.types import ScoredCase
from g2t_aml.facts.checkers import CheckResult, ClaimType, Verdict
from g2t_aml.facts.salience import field_value, is_field_available
from g2t_aml.facts.schema import CaseFacts
from g2t_aml.facts.taxonomy import CRITICAL_CLASSES, HallucinationClass, class_by_id

__all__ = [
    "CRITICAL_IDS",
    "EXCULPATORY_FACTS",
    "HAND_LABEL_SAMPLE_SIZE",
    "ClassifiedError",
    "ExculpatoryFact",
    "TaxonomyReport",
    "ValidationReport",
    "classify",
    "omissions",
    "sample_for_hand_labelling",
    "score_taxonomy",
    "validate_against_hand_labels",
]

#: The three critical class identifiers, derived from the taxonomy rather than restated,
#: so adding a critical class in one place cannot leave this list behind.
CRITICAL_IDS: frozenset[str] = frozenset(h.ident for h in CRITICAL_CLASSES)

#: How many errors are drawn for hand labelling. Two hundred, per the Phase 10 brief:
#: enough for a per-class agreement to mean something on the four or five classes that
#: carry most of the mass, and honestly too few for the tail — which the validation
#: report says out loud rather than implying otherwise by quoting a single overall κ.
HAND_LABEL_SAMPLE_SIZE = 200


@dataclass(frozen=True)
class ExculpatoryFact:
    """A recorded fact that materially weakens the suspicion, and how to detect it.

    Attributes:
        name: Stable identifier, reported so an H9 finding says which fact was omitted.
        field_path: The field a narrative would mention to discharge it. Also what the
            omission check looks for among the narrative's claims.
        description: One sentence, quoted into the finding's reason.
        applies: Predicate over the record. Returns True when this record actually
            carries the exculpatory condition — a licit majority only counts as
            exculpatory when there *is* one.
    """

    name: str
    field_path: str
    description: str
    applies: Callable[[CaseFacts], bool]


def _greater(path_a: str, path_b: str) -> Callable[[CaseFacts], bool]:
    """Build a predicate comparing two numeric fields.

    Args:
        path_a: The field that must be greater.
        path_b: The field it must exceed.

    Returns:
        A predicate that is False whenever either field is masked, absent or
        non-numeric — an unavailable fact cannot be exculpatory, and requiring a
        narrative to mention it would violate invariant 4.
    """

    def predicate(facts: CaseFacts) -> bool:
        if not (is_field_available(facts, path_a) and is_field_available(facts, path_b)):
            return False
        left, right = field_value(facts, path_a), field_value(facts, path_b)
        if isinstance(left, bool) or isinstance(right, bool):
            return False
        if not isinstance(left, int | float) or not isinstance(right, int | float):
            return False
        return left > right

    return predicate


def _equals(path: str, target: object) -> Callable[[CaseFacts], bool]:
    """Build a predicate testing a field against a literal.

    Args:
        path: The field path.
        target: The value it must equal.

    Returns:
        A predicate, False whenever the field is masked or absent.
    """

    def predicate(facts: CaseFacts) -> bool:
        if not is_field_available(facts, path):
            return False
        return bool(field_value(facts, path) == target)

    return predicate


#: The exculpatory conditions H9 is checked against. Deliberately short and explicit.
#: Every entry is a fact an investigator reading the narrative would want to know and
#: whose omission changes how the case reads; a longer list assembled by intuition would
#: turn H9 into "the narrative did not mention everything", which is what Fact Coverage
#: already measures and measures better.
EXCULPATORY_FACTS: tuple[ExculpatoryFact, ...] = (
    ExculpatoryFact(
        name="licit_counterparty_majority",
        field_path="labels.n_licit_counterparties",
        description=(
            "most of the subject's counterparties are labelled licit, which weakens an "
            "inference drawn from the illicit ones"
        ),
        applies=_greater("labels.n_licit_counterparties", "labels.n_illicit_counterparties"),
    ),
    ExculpatoryFact(
        name="no_illicit_counterparty",
        field_path="labels.n_illicit_counterparties",
        description=(
            "no counterparty in the case carries an illicit label at all, so the "
            "suspicion rests on structure alone"
        ),
        applies=_equals("labels.n_illicit_counterparties", 0),
    ),
    ExculpatoryFact(
        name="no_burst_detected",
        field_path="temporal.burst_detected",
        description=(
            "no burst of activity was detected, so the timing carries none of the "
            "urgency a rapid-dispersal reading would need"
        ),
        applies=_equals("temporal.burst_detected", False),
    ),
    ExculpatoryFact(
        name="focal_not_illicit",
        field_path="labels.focal_is_illicit",
        description=(
            "the subject account itself carries no illicit label, so the case is about "
            "its counterparties rather than about it"
        ),
        applies=_equals("labels.focal_is_illicit", False),
    ),
)


@dataclass(frozen=True)
class ClassifiedError:
    """One adverse finding, with its class and where the class came from.

    Attributes:
        case_id: The case.
        system: The arm.
        typology: The case's typology, for the cross-tabulation.
        hallucination_class: The assigned class, ``"H1"``..``"H9"``.
        verdict: CONTRADICTED or UNVERIFIABLE.
        source: ``"checker"`` when the class came from
            :func:`~g2t_aml.facts.checkers.check_claim`, ``"refined"`` when this module
            replaced a generic H8, ``"omission"`` for H9. Carried so the hand-label
            validation can report agreement separately for the classes this module
            decides and the classes it merely passes through — a κ pooled over both would
            be dominated by the checker's work and would flatter the refinement.
        field_path: The fact field, when the claim named one.
        text: The narrative text the finding is about.
        reason: The checker's explanation.
        span: Character offsets into the narrative.
    """

    case_id: str
    system: str
    typology: str
    hallucination_class: str
    verdict: str
    source: str
    field_path: str | None
    text: str
    reason: str
    span: tuple[int, int]

    @property
    def is_critical(self) -> bool:
        """Report whether this finding counts toward the Critical Error Rate.

        Returns:
            True for H4, H6 and H7.
        """
        return self.hallucination_class in CRITICAL_IDS

    def to_dict(self) -> dict[str, Any]:
        """Return the finding as a JSON-serialisable mapping.

        Returns:
            Every field, plus the criticality flag.
        """
        return {
            "case_id": self.case_id,
            "system": self.system,
            "typology": self.typology,
            "hallucination_class": self.hallucination_class,
            "verdict": self.verdict,
            "source": self.source,
            "field_path": self.field_path,
            "text": self.text,
            "reason": self.reason,
            "span": list(self.span),
            "is_critical": self.is_critical,
        }


def classify(result: CheckResult) -> tuple[str, str] | None:  # noqa: PLR0911 -- one
    # return per class in the ladder; collapsing them hides which rule assigned what.
    """Assign a hallucination class to one check result.

    The checker's own class is kept whenever it gave a specific one. The refinement below
    applies only to the generic H8 that :func:`~g2t_aml.facts.checkers._unverifiable`
    returns by default, and only where the claim's own shape decides the class beyond
    doubt. It deliberately does **not** promote an unverifiable quantity to H2: H2 is a
    number that disagrees with the record, and a number the record cannot speak to has
    not disagreed with anything. Softening that distinction would move the largest bucket
    of unverifiable claims into the hallucination count and roughly double every reported
    hallucination rate for no reason but a classification choice.

    Args:
        result: A check result.

    Returns:
        ``(class id, source)``, or None when the verdict is SUPPORTED and there is
        nothing to classify.
    """
    if result.verdict is Verdict.SUPPORTED:
        return None
    assigned = result.hallucination_class
    if assigned is not None and assigned != "H8":
        return assigned, "checker"

    claim = result.claim
    path = claim.field_path or ""
    if claim.claim_type is ClaimType.ENTITY:
        return "H1", "refined"
    if claim.claim_type is ClaimType.REGULATORY:
        return "H6", "refined"
    if path == "typology.label":
        return "H5", "refined"
    if claim.claim_type is ClaimType.TEMPORAL or path.startswith("temporal."):
        return "H3", "refined"
    return "H8", "checker" if assigned == "H8" else "refined"


def omissions(case: ScoredCase, results: Sequence[CheckResult]) -> list[ClassifiedError]:
    """Detect H9: exculpatory facts the record carries and the narrative does not mention.

    Args:
        case: The narrative bound to its record.
        results: Every check result for the narrative, read for which fields it claims.

    Returns:
        One H9 finding per exculpatory condition the record carries and the narrative
        omits. Empty when the narrative mentions them all, or when the substrate carries
        none — an unavailable fact is never exculpatory, so invariant 4 holds here too.
    """
    mentioned = {r.claim.field_path for r in results if r.claim.field_path is not None}
    found: list[ClassifiedError] = []
    for exculpatory in EXCULPATORY_FACTS:
        if exculpatory.field_path in mentioned or not exculpatory.applies(case.facts):
            continue
        found.append(
            ClassifiedError(
                case_id=case.case_id,
                system=case.output.system,
                typology=case.typology,
                hallucination_class="H9",
                verdict=Verdict.UNVERIFIABLE.value,
                source="omission",
                field_path=exculpatory.field_path,
                text="",
                reason=(
                    f"the narrative does not mention {exculpatory.field_path}: "
                    f"{exculpatory.description}"
                ),
                span=(0, 0),
            )
        )
    return found


@dataclass(frozen=True)
class TaxonomyReport:
    """Per-class error rates for one slice, and the cross-tabulation.

    Attributes:
        system: The arm, or a label for the slice.
        n_cases: Narratives in the slice.
        n_errors: Adverse findings, H9 included.
        by_class: Class id to count.
        rate_by_class: Class id to the fraction of *narratives* carrying at least one
            finding of that class. Per-narrative rather than per-claim because "12% of
            narratives fabricate an entity" is the sentence a compliance reader acts on,
            and "H1 is 0.4% of claims" is not.
        critical_error_rate: Fraction of narratives carrying at least one H4/H6/H7.
        critical_by_class: The three critical classes, broken out.
        cross_tab: ``class -> typology -> count``, the paper's table.
        errors: Every finding, for the qualitative analysis.
    """

    system: str
    n_cases: int
    n_errors: int
    by_class: Mapping[str, int]
    rate_by_class: Mapping[str, float]
    critical_error_rate: float
    critical_by_class: Mapping[str, float]
    cross_tab: Mapping[str, Mapping[str, int]]
    errors: tuple[ClassifiedError, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return the report without the individual findings.

        Returns:
            The aggregates, JSON-serialisable. The findings are written separately by
            the error-analysis section of the report.
        """
        return {
            "system": self.system,
            "n_cases": self.n_cases,
            "n_errors": self.n_errors,
            "by_class": dict(self.by_class),
            "rate_by_class": dict(self.rate_by_class),
            "critical_error_rate": self.critical_error_rate,
            "critical_by_class": dict(self.critical_by_class),
            "cross_tab": {k: dict(v) for k, v in self.cross_tab.items()},
        }


def score_taxonomy(
    cases: Sequence[tuple[ScoredCase, CaseFaithfulness]],
    *,
    system: str | None = None,
    detect_omissions: bool = True,
) -> TaxonomyReport:
    """Classify every adverse finding across a slice and aggregate.

    Args:
        cases: ``(scored case, per-case faithfulness)`` pairs. The faithfulness object
            carries the check results, so the classification works from exactly the
            results Layer 2 counted rather than re-running the checker and risking a
            different answer.
        system: Label for the slice. Taken from the cases when they agree.
        detect_omissions: Whether to run the H9 pass. Off for a slice whose narratives
            are not meant to be complete SAR reports — a single-sentence probe, say.

    Returns:
        The taxonomy report.
    """
    name = system
    if name is None:
        names = {c.output.system for c, _ in cases}
        name = names.pop() if len(names) == 1 else "mixed"

    errors: list[ClassifiedError] = []
    per_case_classes: list[set[str]] = []

    for scored, faithfulness in cases:
        classes: set[str] = set()
        for result in faithfulness.results:
            assignment = classify(result)
            if assignment is None:
                continue
            class_id, source = assignment
            classes.add(class_id)
            errors.append(
                ClassifiedError(
                    case_id=scored.case_id,
                    system=scored.output.system,
                    typology=scored.typology,
                    hallucination_class=class_id,
                    verdict=result.verdict.value,
                    source=source,
                    field_path=result.claim.field_path,
                    text=result.claim.raw_text,
                    reason=result.reason,
                    span=result.claim.text_span,
                )
            )
        if detect_omissions:
            found = omissions(scored, faithfulness.results)
            errors.extend(found)
            classes.update(e.hallucination_class for e in found)
        per_case_classes.append(classes)

    by_class: dict[str, int] = {h.ident: 0 for h in HallucinationClass}
    cross_tab: dict[str, dict[str, int]] = {h.ident: {} for h in HallucinationClass}
    for error in errors:
        by_class[error.hallucination_class] += 1
        row = cross_tab[error.hallucination_class]
        row[error.typology] = row.get(error.typology, 0) + 1

    n = len(cases)
    rate_by_class = {
        ident: (sum(1 for classes in per_case_classes if ident in classes) / n if n else 0.0)
        for ident in by_class
    }
    critical_rate = (
        sum(1 for classes in per_case_classes if classes & CRITICAL_IDS) / n if n else 0.0
    )

    return TaxonomyReport(
        system=name,
        n_cases=n,
        n_errors=len(errors),
        by_class=dict(sorted(by_class.items())),
        rate_by_class=dict(sorted(rate_by_class.items())),
        critical_error_rate=critical_rate,
        critical_by_class={ident: rate_by_class[ident] for ident in sorted(CRITICAL_IDS)},
        cross_tab={k: dict(sorted(v.items())) for k, v in sorted(cross_tab.items())},
        errors=tuple(errors),
    )


# --------------------------------------------------- hand-label validation ---


def sample_for_hand_labelling(
    errors: Sequence[ClassifiedError],
    *,
    n: int = HAND_LABEL_SAMPLE_SIZE,
    seed: int = 42,
    stratify: bool = True,
) -> list[ClassifiedError]:
    """Draw errors for a human to classify independently.

    Stratified by automatic class by default, and that choice is a compromise worth
    naming: a uniform sample would be almost entirely H2 and H8 and would tell us nothing
    about whether H4 or H6 are assigned correctly, which are the classes the paper leans
    on hardest. Stratifying oversamples the rare classes, so the *overall* κ it produces
    is not an estimate of the classifier's accuracy on the real error distribution — it
    is an estimate of per-class accuracy, and :func:`validate_against_hand_labels`
    reports per class for that reason.

    Args:
        errors: Every classified error.
        n: How many to draw.
        seed: Seeds the draw, so the sample a human labelled can be reconstructed.
        stratify: Whether to balance the draw across automatic classes.

    Returns:
        The sample, in a stable order.
    """
    rng = random.Random(seed)
    pool = list(errors)
    if len(pool) <= n:
        return sorted(pool, key=lambda e: (e.case_id, e.span))

    if not stratify:
        return sorted(rng.sample(pool, n), key=lambda e: (e.case_id, e.span))

    by_class: dict[str, list[ClassifiedError]] = {}
    for error in pool:
        by_class.setdefault(error.hallucination_class, []).append(error)

    per_class = max(1, n // max(len(by_class), 1))
    drawn: list[ClassifiedError] = []
    for ident in sorted(by_class):
        bucket = by_class[ident]
        drawn.extend(rng.sample(bucket, min(per_class, len(bucket))))
    # Top up from whatever is left, so a taxonomy with several empty classes still yields
    # the requested sample size rather than silently returning a smaller one.
    if len(drawn) < n:
        taken = {id(e) for e in drawn}
        remaining = [e for e in pool if id(e) not in taken]
        drawn.extend(rng.sample(remaining, min(n - len(drawn), len(remaining))))
    return sorted(drawn[:n], key=lambda e: (e.case_id, e.span))


@dataclass(frozen=True)
class ValidationReport:
    """Agreement between the automatic classifier and a human labeller.

    Attributes:
        n_labelled: Errors a human classified.
        accuracy: Fraction the classifier got right.
        kappa: Cohen's κ over the nine classes.
        kappa_band: The verbal band for that κ.
        per_class: Class id to ``{"n": …, "correct": …, "accuracy": …}``. The number that
            matters: a high overall κ driven by H2 says nothing about H6.
        confusion: Automatic class to human class to count.
        by_source: Agreement split by whether the class came from the checker, from this
            module's refinement, or from the omission pass.
    """

    n_labelled: int
    accuracy: float
    kappa: float | None
    kappa_band: str | None
    per_class: Mapping[str, Mapping[str, float]]
    confusion: Mapping[str, Mapping[str, int]]
    by_source: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-serialisable mapping.

        Returns:
            Every field.
        """
        return {
            "n_labelled": self.n_labelled,
            "accuracy": self.accuracy,
            "kappa": self.kappa,
            "kappa_band": self.kappa_band,
            "per_class": {k: dict(v) for k, v in self.per_class.items()},
            "confusion": {k: dict(v) for k, v in self.confusion.items()},
            "by_source": dict(self.by_source),
        }


def validate_against_hand_labels(
    sample: Sequence[ClassifiedError], hand_labels: Mapping[str, str]
) -> ValidationReport:
    """Compare the automatic classification against a human's on the same errors.

    Args:
        sample: The errors that were labelled, as the classifier saw them.
        hand_labels: ``"<case_id>:<start>-<end>"`` to the class a human assigned. Keyed
            on the span rather than on an index so a re-drawn sample in a different order
            still lines up, and so a label file can be edited by hand without a fragile
            positional contract.

    Returns:
        The validation report.

    Raises:
        KeyError: If a hand label names a class outside H1—H9. Strict: a typo would
            otherwise create a tenth bucket that silently counts as a disagreement.
    """
    auto: list[str] = []
    human: list[str] = []
    sources: list[str] = []
    for error in sample:
        key = f"{error.case_id}:{error.span[0]}-{error.span[1]}"
        label = hand_labels.get(key)
        if label is None:
            continue
        class_by_id(label)  # raises KeyError on a class outside the nine
        auto.append(error.hallucination_class)
        human.append(label.strip().upper())
        sources.append(error.source)

    if not auto:
        return ValidationReport(
            n_labelled=0,
            accuracy=0.0,
            kappa=None,
            kappa_band=None,
            per_class={},
            confusion={},
            by_source={},
        )

    correct = [a == h for a, h in zip(auto, human, strict=True)]
    kappa = cohens_kappa(auto, human)

    per_class: dict[str, dict[str, float]] = {}
    confusion: dict[str, dict[str, int]] = {}
    for a, h, ok in zip(auto, human, correct, strict=True):
        entry = per_class.setdefault(a, {"n": 0.0, "correct": 0.0, "accuracy": 0.0})
        entry["n"] += 1
        entry["correct"] += float(ok)
        confusion.setdefault(a, {})
        confusion[a][h] = confusion[a].get(h, 0) + 1
    for entry in per_class.values():
        entry["accuracy"] = entry["correct"] / entry["n"] if entry["n"] else 0.0

    by_source: dict[str, float] = {}
    for source in sorted(set(sources)):
        hits = [ok for ok, s in zip(correct, sources, strict=True) if s == source]
        by_source[source] = sum(hits) / len(hits) if hits else 0.0

    return ValidationReport(
        n_labelled=len(auto),
        accuracy=sum(correct) / len(correct),
        kappa=kappa,
        kappa_band=interpret_kappa(kappa),
        per_class={k: dict(v) for k, v in sorted(per_class.items())},
        confusion={k: dict(sorted(v.items())) for k, v in sorted(confusion.items())},
        by_source=by_source,
    )
