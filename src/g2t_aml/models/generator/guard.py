"""The inference-time verification guard: generate several, verify, select, repair, warn.

This is the deployed-system component. An investigator receiving a first-draft SAR needs
to know which sentences are backed by the case record, and a system that emits its best
guess with no verification is not deployable regardless of how good the average is.

::

    generate n=4 candidates at temperature 0.6
    for each: extract claims -> verify against the fact record
    score = w1*(1 - contradiction_rate) + w2*fact_coverage + w3*(1 - unverifiable_rate)
    select the best
    if the best still contradicts:
        one constrained regeneration naming the violations
        if that still contradicts: emit with a machine-readable warning block

**The guarded and unguarded numbers are two different claims and are reported as two
rows.** The raw model's faithfulness is the scientific claim — it is what the architecture
achieved. The guarded system's faithfulness is the application claim — it is what a
deployment would deliver, and it is better partly because a verifier threw away the bad
candidates. Reporting the guarded number as the model's faithfulness credits the
architecture with the verifier's work. :func:`~g2t_aml.models.generator.guard.GuardReport`
carries both, and :attr:`GuardStatistics` records how often the guard actually intervened
so the table can say what it bought.

**Coverage is in the score for a reason.** Contradiction rate alone is maximised by saying
almost nothing: a one-sentence narrative that names the subject account and stops has a
contradiction rate of zero. Weighting fact coverage against it is what stops the guard
from selecting the emptiest candidate, and the weights are recorded in ``DECISIONS.md``
because they are a judgement call rather than a measurement.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from g2t_aml.facts.checkers import CheckContext, Verdict, check_claim, check_narrative_text
from g2t_aml.facts.salience import salience_report
from g2t_aml.facts.schema import CaseFacts

#: Tolerance on the weight-sum check; the weights are config values, not measurements.
_WEIGHT_SUM_TOL = 1e-6

__all__ = [
    "DEFAULT_WEIGHTS",
    "CandidateScore",
    "GuardReport",
    "GuardStatistics",
    "GuardWeights",
    "InferenceGuard",
    "score_candidate",
]


@dataclass(frozen=True)
class GuardWeights:
    """The selection weights.

    Attributes:
        contradiction: Weight on ``1 - contradiction_rate``. Largest of the three: a
            contradicted claim is a false statement in a regulatory filing, which is a
            categorically worse failure than an incomplete one.
        coverage: Weight on salient-fact coverage. Present so the guard cannot win by
            selecting the candidate that says least.
        unverifiable: Weight on ``1 - unverifiable_rate``. Smallest: an unverifiable claim
            is usually a hedge or a stylistic phrase the checker has no field for, not a
            falsehood.
    """

    contradiction: float = 0.5
    coverage: float = 0.35
    unverifiable: float = 0.15

    def __post_init__(self) -> None:
        """Check the weights are usable.

        Raises:
            ValueError: If any weight is negative or they do not sum to 1, which would
                make scores from different runs incomparable.
        """
        total = self.contradiction + self.coverage + self.unverifiable
        if min(self.contradiction, self.coverage, self.unverifiable) < 0:
            raise ValueError("guard weights must be non-negative")
        if abs(total - 1.0) > _WEIGHT_SUM_TOL:
            raise ValueError(f"guard weights must sum to 1.0, got {total}")


#: The configured default, recorded in ``DECISIONS.md``.
DEFAULT_WEIGHTS = GuardWeights()


@dataclass
class CandidateScore:
    """One candidate narrative and what the checker made of it.

    Attributes:
        text: The narrative.
        score: The weighted selection score.
        contradiction_rate: Share of claims the record contradicts.
        coverage: Share of the case's salient facts the narrative mentions.
        unverifiable_rate: Share of claims that could not be checked.
        n_claims: How many claims were extracted.
        violations: Human-readable descriptions of each contradicted claim, which is what
            the constrained regeneration prompt names back to the model.
    """

    text: str
    score: float
    contradiction_rate: float
    coverage: float
    unverifiable_rate: float
    n_claims: int
    violations: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """Report whether this candidate contradicts nothing.

        Returns:
            True when no claim was contradicted.
        """
        return not self.violations

    def to_dict(self) -> dict[str, Any]:
        """Return the score as a JSON-serialisable mapping.

        Returns:
            The fields.
        """
        return asdict(self)


def score_candidate(
    text: str,
    facts: CaseFacts,
    extractor: Any,
    *,
    weights: GuardWeights = DEFAULT_WEIGHTS,
    context: CheckContext | None = None,
) -> CandidateScore:
    """Verify one candidate and score it.

    Args:
        text: The candidate narrative.
        facts: The fact record to verify against.
        extractor: A :class:`~g2t_aml.corpus.silver.claim_extraction.ClaimExtractor`.
        weights: The selection weights.
        context: A prepared checking context, or None to build one.

    Returns:
        The candidate's score and its violations.
    """
    ctx = context if context is not None else CheckContext(facts=facts)
    claims = extractor.extract(text, facts)
    results = [check_claim(claim, ctx) for claim in claims]
    results.extend(check_narrative_text(text, ctx))

    total = len(results)
    contradicted = [r for r in results if r.verdict is Verdict.CONTRADICTED]
    unverifiable = sum(1 for r in results if r.verdict is Verdict.UNVERIFIABLE)

    contradiction_rate = len(contradicted) / total if total else 0.0
    unverifiable_rate = unverifiable / total if total else 0.0

    # Coverage is measured against Phase 3's salience list for this record's typology —
    # the same list the annotation guidelines and the adequacy metric use, so the guard
    # is not selecting against a definition of "adequate" it invented for itself. Fields
    # the record cannot support are already excused by `salience_report`, so a case with
    # masked families is not penalised for staying silent about them (invariant 4).
    required = set(salience_report(facts).required)
    mentioned = {c.field_path for c in claims if c.field_path is not None}
    coverage = len(required & mentioned) / len(required) if required else 1.0

    score = (
        weights.contradiction * (1.0 - contradiction_rate)
        + weights.coverage * coverage
        + weights.unverifiable * (1.0 - unverifiable_rate)
    )
    return CandidateScore(
        text=text,
        score=score,
        contradiction_rate=contradiction_rate,
        coverage=coverage,
        unverifiable_rate=unverifiable_rate,
        n_claims=total,
        violations=[f"{r.claim.field_path or 'text'}: {r.reason}" for r in contradicted],
    )


@dataclass
class GuardStatistics:
    """How often the guard did something, which is a table in the paper.

    A guard that never changes the selection and never regenerates is a guard that costs
    four times the compute for nothing, and that is a finding worth publishing too.

    Attributes:
        n_cases: Cases processed.
        n_selection_changed: Cases where the selected candidate was not the first one
            sampled — that is, where the guard's verification changed the output.
        n_regenerated: Cases that triggered a constrained regeneration.
        n_regeneration_helped: Cases where the regeneration produced a clean narrative
            that the best candidate was not.
        n_warned: Cases emitted with a warning block because nothing clean was found.
        n_clean_first_try: Cases whose first candidate was already clean.
    """

    n_cases: int = 0
    n_selection_changed: int = 0
    n_regenerated: int = 0
    n_regeneration_helped: int = 0
    n_warned: int = 0
    n_clean_first_try: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return the statistics with their rates.

        Returns:
            The counts plus the four rates the results table reports.
        """
        n = self.n_cases or 1
        return {
            **asdict(self),
            "selection_changed_rate": self.n_selection_changed / n,
            "regeneration_rate": self.n_regenerated / n,
            "regeneration_success_rate": (
                self.n_regeneration_helped / self.n_regenerated if self.n_regenerated else 0.0
            ),
            "warning_rate": self.n_warned / n,
        }


@dataclass
class GuardReport:
    """What the guard emitted for one case, and both numbers the paper reports.

    Attributes:
        case_id: The case.
        text: The emitted narrative.
        selected: The winning candidate's score.
        unguarded: The *first* candidate's score — what the raw model would have emitted
            with no guard. **This is the scientific claim's number**, and it is carried
            alongside the guarded one so the two rows of the results table come from the
            same run rather than from two.
        candidates: Every candidate's score, in sampling order.
        regenerated: Whether a constrained regeneration was run.
        warning: The machine-readable warning block, or None when the output is clean.
    """

    case_id: str
    text: str
    selected: CandidateScore
    unguarded: CandidateScore
    candidates: list[CandidateScore] = field(default_factory=list)
    regenerated: bool = False
    warning: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-serialisable mapping.

        Returns:
            The fields, with nested scores flattened to mappings.
        """
        return {
            "case_id": self.case_id,
            "text": self.text,
            "selected": self.selected.to_dict(),
            "unguarded": self.unguarded.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates],
            "regenerated": self.regenerated,
            "warning": self.warning,
        }


class InferenceGuard:
    """Selects the best-verified candidate, repairs once, and warns when it cannot.

    Generation is injected as a callable rather than taken as a model, so the guard is
    testable against a fixture list of candidates — which is how its selection logic is
    checked without a GPU.
    """

    def __init__(
        self,
        *,
        weights: GuardWeights = DEFAULT_WEIGHTS,
        allow_regeneration: bool = True,
        log: Any = None,
    ) -> None:
        """Build the guard.

        Args:
            weights: The selection weights.
            allow_regeneration: Whether to attempt one constrained regeneration. Off makes
                the guard selection-only, which is the ablation that separates what
                selection bought from what repair bought.
            log: Optional logger.
        """
        self.weights = weights
        self.allow_regeneration = allow_regeneration
        self.log = log
        self.stats = GuardStatistics()

    def build_repair_prompt(self, violations: Sequence[str]) -> str:
        """Compose the instruction for the constrained regeneration.

        Naming the specific violations is the whole mechanism. "Try again, be accurate"
        gives the model nothing to act on; "you stated the subject dispersed to 9
        counterparties, the record says 4" identifies the span to fix.

        Args:
            violations: The contradicted claims, as rendered by :func:`score_candidate`.

        Returns:
            The instruction text appended to the original prompt.
        """
        listed = "\n".join(f"- {v}" for v in violations)
        return (
            "Your previous draft contradicted the case record in the following places:\n"
            f"{listed}\n"
            "Rewrite the narrative correcting exactly these statements. Change nothing "
            "else, and do not add any fact the record does not contain."
        )

    def run(
        self,
        case_id: str,
        candidates: Sequence[str],
        facts: CaseFacts,
        extractor: Any,
        *,
        regenerate: Callable[[Sequence[str]], str] | None = None,
        context: CheckContext | None = None,
    ) -> GuardReport:
        """Verify candidates, select the best, repair once, and warn if it must.

        Args:
            case_id: The case.
            candidates: Sampled narratives, in sampling order. The first is treated as
                what the unguarded system would have emitted.
            facts: The fact record.
            extractor: The claim extractor for this case.
            regenerate: Called with the violation list to produce one constrained
                regeneration. None disables repair for this call.
            context: A prepared checking context.

        Returns:
            The report, carrying both the guarded and the unguarded result.

        Raises:
            ValueError: If no candidates were given.
        """
        if not candidates:
            raise ValueError(f"case {case_id} produced no candidates to select between")

        ctx = context if context is not None else CheckContext(facts=facts)
        scored = [
            score_candidate(text, facts, extractor, weights=self.weights, context=ctx)
            for text in candidates
        ]
        best_index = max(range(len(scored)), key=lambda i: scored[i].score)
        best = scored[best_index]

        self.stats.n_cases += 1
        if best_index != 0:
            self.stats.n_selection_changed += 1
        if scored[0].is_clean:
            self.stats.n_clean_first_try += 1

        regenerated = False
        if not best.is_clean and self.allow_regeneration and regenerate is not None:
            regenerated = True
            self.stats.n_regenerated += 1
            repaired_text = regenerate(best.violations)
            repaired = score_candidate(
                repaired_text, facts, extractor, weights=self.weights, context=ctx
            )
            scored.append(repaired)
            if repaired.is_clean:
                self.stats.n_regeneration_helped += 1
                best = repaired
            elif repaired.score > best.score:
                # Kept only on a genuine improvement. A repair that fixes one contradiction
                # while introducing another must not be preferred just for being newer.
                best = repaired

        warning: dict[str, Any] | None = None
        if not best.is_clean:
            self.stats.n_warned += 1
            warning = {
                "case_id": case_id,
                "status": "unverified_claims_present",
                "n_contradicted": len(best.violations),
                "contradicted_claims": list(best.violations),
                "contradiction_rate": best.contradiction_rate,
                "unverifiable_rate": best.unverifiable_rate,
                "guidance": (
                    "This draft contains statements the case record does not support. "
                    "Verify each listed claim against the source data before filing."
                ),
            }
            if self.log is not None:
                self.log.warning(
                    "case %s emitted with %d unresolved contradiction(s)",
                    case_id,
                    len(best.violations),
                )

        return GuardReport(
            case_id=case_id,
            text=best.text,
            selected=best,
            unguarded=scored[0],
            candidates=scored,
            regenerated=regenerated,
            warning=warning,
        )
