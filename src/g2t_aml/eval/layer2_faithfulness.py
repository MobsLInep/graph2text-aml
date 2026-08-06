"""Layer 2: faithfulness. The defensible core of the evaluation.

Layer 1 measures whether a narrative resembles a reference. This measures whether it is
*true of the case*, which is the only property a compliance function can act on and the
only one that survives the absence of a large human-written reference set.

**Zero-Hallucination Rate is the headline number.** Not Fact Precision, not Fact F1 — the
per-narrative binary: what fraction of narratives contain no contradicted claim at all. A
SAR narrative containing one fabricated fact is unusable regardless of how good the rest
of it is; it cannot be filed, and an investigator who finds one error has to re-verify
everything. Averaged precision hides exactly that. A system at 97% Fact Precision could be
one that puts a single error into every narrative — 0% Zero-Hallucination — or one that is
perfect on 97% of cases and badly wrong on 3%. Those are different products, and the mean
cannot tell them apart. See D-077.

**Coverage counts a salient fact as covered only when the narrative gets it right.** A
contradicted mention is not coverage. Scoring it as coverage would make Fact F1
maximisable by asserting every salient field wrongly, which is the precise opposite of
what the metric is for.

**Salient fields come from Phase 3 and are not redefined here.**
:func:`~g2t_aml.facts.salience.salience_report` owns the lists, they were frozen before a
single narrative existed (D-032), and availability already excuses omission on a substrate
that cannot support a field. This module reads that report and nothing else.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from g2t_aml.eval.types import ScoredCase
from g2t_aml.facts.checkers import (
    CheckContext,
    CheckResult,
    Claim,
    ClaimType,
    Verdict,
    check_claim,
    check_narrative_text,
)
from g2t_aml.facts.config import FactConfig
from g2t_aml.facts.salience import salience_report
from g2t_aml.facts.vocab import ControlledVocabulary, load_vocabulary

__all__ = [
    "NUMERIC_FIELD_PREFIXES",
    "CaseFaithfulness",
    "FaithfulnessMetrics",
    "aggregate_faithfulness",
    "score_case",
    "score_cases",
]

#: Field paths whose claims count toward Numeric Accuracy. Quantitative in the sense the
#: metric means: a number an investigator would read off the narrative and act on.
#: Booleans and categoricals are deliberately outside it — "a burst was detected" is not
#: a quantity, and folding it in would let a system raise its numeric accuracy by writing
#: more boolean sentences.
NUMERIC_FIELD_PREFIXES: tuple[str, ...] = (
    "structure.n_",
    "structure.density",
    "structure.diameter",
    "structure.max_",
    "structure.reciprocity",
    "focal_entity.in_degree",
    "focal_entity.out_degree",
    "focal_entity.n_",
    "temporal.span_hours",
    "temporal.burst_",
    "temporal.n_transactions",
    "flow.total_",
    "flow.retained",
    "flow.max_single_transfer",
    "flow.n_",
    "flow.threshold_",
    "labels.n_",
    "labels.illicit_inflow_share",
    "labels.min_hops_to_known_illicit",
    "motifs.",
    "model_signal.",
)


def _is_numeric_claim(claim: Claim) -> bool:
    """Report whether a claim counts toward Numeric Accuracy.

    Args:
        claim: The claim.

    Returns:
        True when it is a NUMERIC claim about a field in
        :data:`NUMERIC_FIELD_PREFIXES`. A numeric claim naming no field is excluded: it
        is an unbacked quantity, already counted in the unverifiable rate, and counting
        it here would make Numeric Accuracy fall as a system invented *more* numbers,
        which it does, but for the wrong reason — the metric would be reporting
        invention where it is meant to report arithmetic.
    """
    if claim.claim_type is not ClaimType.NUMERIC or claim.field_path is None:
        return False
    return claim.field_path.startswith(NUMERIC_FIELD_PREFIXES)


@dataclass(frozen=True)
class CaseFaithfulness:
    """Faithfulness for one narrative.

    Every rate is over this narrative alone. The aggregate is a mean of these, not a
    pooled ratio over all claims: a macro average gives every *narrative* one vote, which
    is the unit a compliance reviewer works in, and stops a single long narrative with
    two hundred claims dominating a set of short ones. The pooled figures are reported
    too, on :class:`FaithfulnessMetrics`, so the difference is visible rather than a
    choice the reader has to take on trust.

    Attributes:
        case_id: The case.
        system: The arm that produced the narrative.
        typology: The case's typology, for the per-typology breakdown.
        dataset: The substrate.
        seed: The generation seed, for the variance report.
        stream: ``"balanced"`` or ``"realistic"``.
        n_claims: Total claims extracted.
        n_supported: Claims the record entails.
        n_contradicted: Claims the record refutes.
        n_unverifiable: Claims the record cannot speak to.
        n_salient_required: Salient fields this record supports.
        n_salient_covered: Of those, how many the narrative asserts without contradicting.
        n_numeric: Quantitative claims.
        n_numeric_correct: Of those, how many match the record exactly.
        typology_correct: True when the narrative names the recorded typology and
            contradicts none, False when it does not, None when the case has no
            ground-truth typology to name.
        ordering_correct: True when every temporal-ordering claim holds, None when the
            narrative makes none.
        results: Every check result, kept so the taxonomy scorer and the error analysis
            can work from the same objects rather than re-deriving them.
    """

    case_id: str
    system: str
    typology: str
    dataset: str
    seed: int | None
    stream: str
    n_claims: int
    n_supported: int
    n_contradicted: int
    n_unverifiable: int
    n_salient_required: int
    n_salient_covered: int
    n_numeric: int
    n_numeric_correct: int
    typology_correct: bool | None
    ordering_correct: bool | None
    results: tuple[CheckResult, ...] = ()

    @property
    def fact_precision(self) -> float:
        """Return supported claims over all claims.

        Returns:
            The precision in [0, 1], and 1.0 for a narrative with no claims — which is
            reported as such and is why coverage exists: a narrative that asserts nothing
            is perfectly precise and useless, and Fact F1 is what refuses to reward it.
        """
        return self.n_supported / self.n_claims if self.n_claims else 1.0

    @property
    def hallucination_rate(self) -> float:
        """Return contradicted claims over all claims.

        Returns:
            The rate in [0, 1], and 0.0 for a narrative with no claims.
        """
        return self.n_contradicted / self.n_claims if self.n_claims else 0.0

    @property
    def unverifiable_rate(self) -> float:
        """Return unverifiable claims over all claims.

        Returns:
            The rate in [0, 1], and 0.0 for a narrative with no claims.
        """
        return self.n_unverifiable / self.n_claims if self.n_claims else 0.0

    @property
    def fact_coverage(self) -> float:
        """Return correctly-asserted salient fields over available salient fields.

        Returns:
            The coverage in [0, 1], and 1.0 when the record supports no salient field at
            all — vacuous coverage rather than zero, because a substrate that cannot
            support any salient fact must not drag a system's score down for it.
        """
        if not self.n_salient_required:
            return 1.0
        return self.n_salient_covered / self.n_salient_required

    @property
    def fact_f1(self) -> float:
        """Return the harmonic mean of precision and coverage.

        Returns:
            The F1 in [0, 1], and 0.0 when either component is zero.
        """
        precision, coverage = self.fact_precision, self.fact_coverage
        if precision + coverage <= 0:
            return 0.0
        return 2 * precision * coverage / (precision + coverage)

    @property
    def numeric_accuracy(self) -> float | None:
        """Return the exact-match rate on quantitative claims.

        Returns:
            The rate in [0, 1], or None when the narrative makes no quantitative claim —
            None rather than 1.0, so a system that avoids numbers does not accumulate
            perfect scores it never earned.
        """
        return self.n_numeric_correct / self.n_numeric if self.n_numeric else None

    @property
    def zero_hallucination(self) -> bool:
        """Report whether this narrative contains no contradicted claim at all.

        **The headline unit.** See the module docstring.

        Returns:
            True when no claim was contradicted.
        """
        return self.n_contradicted == 0

    @property
    def n_critical(self) -> int:
        """Return how many results fall in the critical classes H4, H6 or H7.

        Returns:
            The count.
        """
        return sum(1 for result in self.results if result.is_critical)

    def to_dict(self) -> dict[str, Any]:
        """Return the per-case metrics as a JSON-serialisable mapping.

        Returns:
            The counts and the derived rates. The check results are not included: they
            are large, and the error analysis writes the ones it needs separately.
        """
        return {
            "case_id": self.case_id,
            "system": self.system,
            "typology": self.typology,
            "dataset": self.dataset,
            "seed": self.seed,
            "stream": self.stream,
            "n_claims": self.n_claims,
            "n_supported": self.n_supported,
            "n_contradicted": self.n_contradicted,
            "n_unverifiable": self.n_unverifiable,
            "n_salient_required": self.n_salient_required,
            "n_salient_covered": self.n_salient_covered,
            "n_numeric": self.n_numeric,
            "n_numeric_correct": self.n_numeric_correct,
            "n_critical": self.n_critical,
            "fact_precision": self.fact_precision,
            "hallucination_rate": self.hallucination_rate,
            "unverifiable_rate": self.unverifiable_rate,
            "fact_coverage": self.fact_coverage,
            "fact_f1": self.fact_f1,
            "numeric_accuracy": self.numeric_accuracy,
            "typology_correct": self.typology_correct,
            "ordering_correct": self.ordering_correct,
            "zero_hallucination": self.zero_hallucination,
        }


def score_case(
    case: ScoredCase,
    claims: Sequence[Claim],
    *,
    config: FactConfig | None = None,
    vocabulary: ControlledVocabulary | None = None,
) -> CaseFaithfulness:
    """Check one narrative's claims against its record and derive the per-case metrics.

    The text-level scan (:func:`~g2t_aml.facts.checkers.check_narrative_text`) runs here
    as well as the claim-level checks, and its findings are counted as claims. They have
    to be: a narrative can be arithmetically perfect and still assert guilt or name a
    business type, and those are the two most damaging things it can do. Leaving them out
    of the denominator *and* the numerator would let a system with a guilt overclaim in
    every narrative report 100% Zero-Hallucination.

    Args:
        case: The narrative bound to its fact record.
        claims: The claims an extractor found in the narrative.
        config: Thresholds and the tolerance policy. Defaults are the published policy.
        vocabulary: The controlled vocabulary. Loaded from disk when omitted.

    Returns:
        The per-case faithfulness.
    """
    ctx = CheckContext(
        facts=case.facts,
        config=config if config is not None else FactConfig(),
        vocabulary=vocabulary if vocabulary is not None else load_vocabulary(),
    )
    results = [check_claim(claim, ctx) for claim in claims]
    results.extend(check_narrative_text(case.output.narrative, ctx))

    n_supported = sum(1 for r in results if r.verdict is Verdict.SUPPORTED)
    n_contradicted = sum(1 for r in results if r.verdict is Verdict.CONTRADICTED)
    n_unverifiable = sum(1 for r in results if r.verdict is Verdict.UNVERIFIABLE)

    salience = salience_report(case.facts, ctx.vocabulary)
    # A salient field counts as covered when the narrative claims it and is not wrong
    # about it. Counting a contradicted mention would make coverage maximisable by
    # asserting every salient field incorrectly.
    covered = {
        r.claim.field_path
        for r in results
        if r.claim.field_path is not None and r.verdict is not Verdict.CONTRADICTED
    }
    n_covered = sum(1 for path in salience.required if path in covered)

    numeric = [r for r in results if _is_numeric_claim(r.claim)]
    n_numeric_correct = sum(1 for r in numeric if r.verdict is Verdict.SUPPORTED)

    typology_results = [r for r in results if r.claim.field_path == "typology.label"]
    typology_correct: bool | None = None
    if case.facts.typology.label != "unclassified":
        typology_correct = bool(typology_results) and all(
            r.verdict is Verdict.SUPPORTED for r in typology_results
        )

    ordering_results = [r for r in results if r.claim.field_path == "temporal.event_ordering"]
    ordering_correct: bool | None = (
        all(r.verdict is Verdict.SUPPORTED for r in ordering_results) if ordering_results else None
    )

    return CaseFaithfulness(
        case_id=case.case_id,
        system=case.output.system,
        typology=case.typology,
        dataset=case.dataset,
        seed=case.output.seed,
        stream=case.output.stream,
        n_claims=len(results),
        n_supported=n_supported,
        n_contradicted=n_contradicted,
        n_unverifiable=n_unverifiable,
        n_salient_required=len(salience.required),
        n_salient_covered=n_covered,
        n_numeric=len(numeric),
        n_numeric_correct=n_numeric_correct,
        typology_correct=typology_correct,
        ordering_correct=ordering_correct,
        results=tuple(results),
    )


@dataclass(frozen=True)
class FaithfulnessMetrics:
    """Layer 2 aggregated over a set of narratives.

    Attributes:
        system: The arm, or a label for whatever slice this aggregates.
        n_cases: Narratives aggregated.
        zero_hallucination_rate: **The headline.** Fraction of narratives with no
            contradicted claim.
        fact_precision: Mean per-narrative precision.
        hallucination_rate: Mean per-narrative hallucination rate.
        unverifiable_rate: Mean per-narrative unverifiable rate.
        fact_coverage: Mean per-narrative coverage.
        fact_f1: Mean per-narrative F1. The mean of the per-case F1s, not the F1 of the
            means: the second is not a property any narrative has.
        numeric_accuracy: Mean over narratives that made a quantitative claim.
        typology_accuracy: Fraction of cases with a ground-truth typology whose narrative
            names it correctly. None when no case in the slice has one — which is every
            Elliptic2 slice, because the substrate carries no typology ground truth.
        ordering_accuracy: Fraction of narratives whose temporal-ordering claims all hold.
        critical_error_rate: Fraction of narratives carrying at least one H4/H6/H7
            finding. Per-narrative for the same reason the headline is: one fabricated
            regulation makes a report unfileable.
        pooled_fact_precision: Supported claims over all claims, pooled across
            narratives. Reported beside the macro mean so the difference between the two
            is visible.
        pooled_hallucination_rate: Contradicted claims over all claims, pooled.
        n_claims: Total claims across the slice.
        n_narratives_with_no_claims: Narratives from which no claim could be extracted.
            Reported prominently: a narrative with no claims scores perfect precision and
            perfect zero-hallucination, so a rising count here is the signature of an
            extractor failure masquerading as a quality improvement.
    """

    system: str
    n_cases: int
    zero_hallucination_rate: float
    fact_precision: float
    hallucination_rate: float
    unverifiable_rate: float
    fact_coverage: float
    fact_f1: float
    numeric_accuracy: float | None
    typology_accuracy: float | None
    ordering_accuracy: float | None
    critical_error_rate: float
    pooled_fact_precision: float
    pooled_hallucination_rate: float
    n_claims: int
    n_narratives_with_no_claims: int
    per_case: tuple[CaseFaithfulness, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return the aggregate as a JSON-serialisable mapping.

        Returns:
            Every aggregate field. The per-case detail is not included; it is written to
            its own file so the summary stays readable.
        """
        return {
            "system": self.system,
            "n_cases": self.n_cases,
            "zero_hallucination_rate": self.zero_hallucination_rate,
            "fact_precision": self.fact_precision,
            "hallucination_rate": self.hallucination_rate,
            "unverifiable_rate": self.unverifiable_rate,
            "fact_coverage": self.fact_coverage,
            "fact_f1": self.fact_f1,
            "numeric_accuracy": self.numeric_accuracy,
            "typology_accuracy": self.typology_accuracy,
            "ordering_accuracy": self.ordering_accuracy,
            "critical_error_rate": self.critical_error_rate,
            "pooled_fact_precision": self.pooled_fact_precision,
            "pooled_hallucination_rate": self.pooled_hallucination_rate,
            "n_claims": self.n_claims,
            "n_narratives_with_no_claims": self.n_narratives_with_no_claims,
        }


def _mean(values: Sequence[float]) -> float:
    """Return a mean, or 0.0 over an empty sequence.

    Args:
        values: The values.

    Returns:
        The arithmetic mean.
    """
    return statistics.fmean(values) if values else 0.0


def _optional_mean(values: Sequence[float]) -> float | None:
    """Return a mean, or None over an empty sequence.

    Used where an empty sequence means *not measured* rather than *measured as zero*.

    Args:
        values: The values.

    Returns:
        The mean, or None.
    """
    return statistics.fmean(values) if values else None


def aggregate_faithfulness(
    cases: Sequence[CaseFaithfulness], *, system: str | None = None
) -> FaithfulnessMetrics:
    """Aggregate per-case faithfulness into the reported metrics.

    Args:
        cases: The per-case results. May span systems, in which case ``system`` should
            name the slice.
        system: Label for the aggregate. Taken from the cases when they agree, and
            ``"mixed"`` when they do not.

    Returns:
        The aggregate. Every rate is 0.0 over an empty input rather than undefined, and
        ``n_cases`` says so.
    """
    name = system
    if name is None:
        names = {case.system for case in cases}
        name = names.pop() if len(names) == 1 else "mixed"

    typology_scored = [c.typology_correct for c in cases if c.typology_correct is not None]
    ordering_scored = [c.ordering_correct for c in cases if c.ordering_correct is not None]
    numeric_scored = [c.numeric_accuracy for c in cases if c.numeric_accuracy is not None]

    total_claims = sum(c.n_claims for c in cases)
    total_supported = sum(c.n_supported for c in cases)
    total_contradicted = sum(c.n_contradicted for c in cases)

    return FaithfulnessMetrics(
        system=name,
        n_cases=len(cases),
        zero_hallucination_rate=_mean([float(c.zero_hallucination) for c in cases]),
        fact_precision=_mean([c.fact_precision for c in cases]),
        hallucination_rate=_mean([c.hallucination_rate for c in cases]),
        unverifiable_rate=_mean([c.unverifiable_rate for c in cases]),
        fact_coverage=_mean([c.fact_coverage for c in cases]),
        fact_f1=_mean([c.fact_f1 for c in cases]),
        numeric_accuracy=_optional_mean(numeric_scored),
        typology_accuracy=_optional_mean([float(v) for v in typology_scored]),
        ordering_accuracy=_optional_mean([float(v) for v in ordering_scored]),
        critical_error_rate=_mean([float(c.n_critical > 0) for c in cases]),
        pooled_fact_precision=total_supported / total_claims if total_claims else 1.0,
        pooled_hallucination_rate=total_contradicted / total_claims if total_claims else 0.0,
        n_claims=total_claims,
        n_narratives_with_no_claims=sum(1 for c in cases if c.n_claims == 0),
        per_case=tuple(cases),
    )


def score_cases(
    cases: Iterable[tuple[ScoredCase, Sequence[Claim]]],
    *,
    config: FactConfig | None = None,
    vocabulary: ControlledVocabulary | None = None,
) -> list[CaseFaithfulness]:
    """Score many narratives.

    Args:
        cases: ``(scored case, claims)`` pairs. The claims come from an extractor the
            caller chose, so the same aggregation serves Method A at every checkpoint and
            Method B on a sample.
        config: Thresholds and the tolerance policy.
        vocabulary: The controlled vocabulary, loaded once and shared.

    Returns:
        The per-case results, in input order.
    """
    vocab = vocabulary if vocabulary is not None else load_vocabulary()
    cfg = config if config is not None else FactConfig()
    return [score_case(case, claims, config=cfg, vocabulary=vocab) for case, claims in cases]
