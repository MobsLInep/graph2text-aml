"""generate → verify → repair → discard, and the discard log that comes out of it.

This module is the defence against the objection Silver exists to answer. Generating
references with a frontier model and then scoring an 8B student against those same
references measures how well the student imitates the teacher, and a reviewer will say so.
What makes Silver *verified synthetic supervision* rather than distillation is mechanical
and lives here:

- **Two teachers**, assigned deterministically and balanced across typology and split, so
  no stratum is one model's style.
- **Every record verified against the fact record**, by the same checker that produces the
  paper's faithfulness numbers, run in reverse. Teacher agreement is never consulted.
- **Bounded repair, then discard.** Two attempts, then the case is dropped and logged.
- **Evaluation against human-authored Gold**, in Phase 10 — never against Silver.

**The discard log is a deliverable.** "A frontier model produced an unrepairable factual
violation in X% of cases *even when handed a complete structured fact record and a correct
draft*" is a genuine result, it is the direct motivation for a graph-conditioned
architecture with a verifier in the loop, and it is invisible unless the failures are
instrumented on purpose. Every discard carries its case, teacher, typology, attempt count,
per-class violation breakdown and the checker's own summary.

**On not tuning the thresholds to reduce discards.** The unverifiable budget is 0.05
because that is the published standard the whole project reports against; a run that
raised it to make its discard rate look better would have changed the measurement to fit
the result. If the discard rate is high, that is the finding.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from g2t_aml.corpus.record import BronzeNarrative, SlotAnnotation, TrainingRecord
from g2t_aml.corpus.silver.api_client import (
    BudgetExceeded,
    Teacher,
    TeacherError,
    TeacherResponse,
    TeacherSpec,
)
from g2t_aml.corpus.silver.claim_extraction import (
    ExtractionReport,
    SlotAlignmentExtractor,
    canonicalise_narrative,
)
from g2t_aml.corpus.silver.prompts import (
    DEFAULT_MAX_WORDS,
    DEFAULT_MIN_WORDS,
    Violation,
    build_repair_prompt,
    build_rewrite_prompt,
)
from g2t_aml.corpus.tokenization import TokenCounter, get_token_counter, word_count
from g2t_aml.facts.checkers import (
    CheckContext,
    CheckResult,
    Verdict,
    check_claim,
    check_narrative_text,
    summarise,
)
from g2t_aml.facts.salience import salience_report
from g2t_aml.facts.schema import CaseFacts
from g2t_aml.facts.serialiser import serialise_facts
from g2t_aml.facts.vocab import ControlledVocabulary
from g2t_aml.utils.logging import get_logger

__all__ = [
    "MAX_REPAIR_ATTEMPTS",
    "MAX_UNVERIFIABLE_RATE",
    "MIN_SALIENCE_COVERAGE",
    "CaseInput",
    "DiscardRecord",
    "GenerationOutcome",
    "SilverConfig",
    "VerificationVerdict",
    "assign_teachers",
    "generate_one",
    "teacher_balance_report",
    "verify_rewrite",
]

log = get_logger(__name__)

#: Repair attempts before a case is discarded. Two, and not negotiable at the command
#: line. Past the second attempt a model stops repairing and starts contorting: it will
#: satisfy the checker by deleting the sentence that carried the finding, and the result
#: passes verification while reading like nothing a human wrote. An unbounded loop also
#: spends the budget on exactly the cases least likely to yield a usable record.
MAX_REPAIR_ATTEMPTS = 2

#: The unverifiable budget, per record. The same 0.05 the ten-point harness applies, named
#: here rather than imported from :mod:`g2t_aml.corpus.validate` only so this module reads
#: standalone; a test asserts the two agree, because two thresholds that can drift are one
#: bug away from a corpus that passes generation and fails its own gate.
MAX_UNVERIFIABLE_RATE = 0.05

#: Minimum share of the typology's salient fields a rewrite must still carry. Bronze is at
#: 1.000 by construction. This is the ``min_fact_recall`` the Silver config has declared
#: since Phase 0, enforced: a rewrite that silently drops half the findings is fluent and
#: faithful and useless, and neither the contradiction count nor the unverifiable rate can
#: see it — omission is the one hallucination class (H9) detected by absence.
MIN_SALIENCE_COVERAGE = 0.95


@dataclass(frozen=True)
class SilverConfig:
    """Thresholds and knobs for one generation run.

    Attributes:
        max_unverifiable_rate: The unverifiable budget.
        min_salience_coverage: Required salient-field retention.
        max_repair_attempts: Repair attempts before discard.
        min_words: Lower end of the word band offered to the teacher.
        max_words: Upper end.
        tier: Corpus tier written onto the record.
        seed: The run's global seed, recorded on every record (invariant 5). Distinct
            from a teacher's provider seed, which is a decoding parameter and may not
            exist at all.
    """

    max_unverifiable_rate: float = MAX_UNVERIFIABLE_RATE
    min_salience_coverage: float = MIN_SALIENCE_COVERAGE
    max_repair_attempts: int = MAX_REPAIR_ATTEMPTS
    min_words: int = DEFAULT_MIN_WORDS
    max_words: int = DEFAULT_MAX_WORDS
    tier: str = "silver"
    seed: int = 42


@dataclass(frozen=True)
class CaseInput:
    """One case, ready to rewrite.

    Attributes:
        case_id: The case.
        split: Its split, from the frozen Phase 2 manifest.
        facts: The fact record.
        bronze: The Bronze narrative and its slot alignment.
    """

    case_id: str
    split: str
    facts: CaseFacts
    bronze: BronzeNarrative

    @property
    def typology(self) -> str:
        """Return the case's typology label.

        Returns:
            The label, which strata the teacher assignment.
        """
        return self.facts.typology.label


@dataclass(frozen=True)
class VerificationVerdict:
    """The checker's answer on one rewrite, and everything repair or the log needs.

    Attributes:
        n_claims: Claims extracted.
        supported: SUPPORTED count.
        contradicted: CONTRADICTED count.
        unverifiable: UNVERIFIABLE count.
        unverifiable_rate: Share of claims that are UNVERIFIABLE.
        critical_error_rate: Share carrying H4, H6 or H7.
        by_hallucination_class: Per-class counts, the discard log's breakdown.
        salience_coverage: Share of required salient fields the rewrite still carries.
        dropped_salient: Which salient fields it dropped.
        added_spans: Quantities the rewrite introduced that Bronze did not carry.
        violations: The failures, phrased for a repair prompt.
        max_unverifiable_rate: The budget this verdict was judged against, carried on the
            verdict rather than read from a module constant at decision time so a verdict
            can never be re-judged under a threshold it was not produced under.
        min_salience_coverage: The retention floor it was judged against.
    """

    n_claims: int
    supported: int
    contradicted: int
    unverifiable: int
    unverifiable_rate: float
    critical_error_rate: float
    by_hallucination_class: dict[str, int]
    salience_coverage: float
    dropped_salient: tuple[str, ...]
    added_spans: tuple[tuple[int, int, str], ...]
    violations: tuple[Violation, ...]
    max_unverifiable_rate: float = MAX_UNVERIFIABLE_RATE
    min_salience_coverage: float = MIN_SALIENCE_COVERAGE

    @property
    def accepted(self) -> bool:
        """Report whether this rewrite may be written to the corpus.

        Returns:
            True when there are no contradictions, the unverifiable rate is inside its
            budget, and the salient facts survived.
        """
        return not self.failures()

    def failures(self) -> tuple[str, ...]:
        """Return the gate conditions this rewrite fails.

        Returns:
            Zero or more machine-readable reasons, which become the discard log's
            ``reason`` field and let the log be aggregated by cause rather than by
            free text.
        """
        reasons: list[str] = []
        if self.contradicted > 0:
            reasons.append("contradicted_claims")
        if self.unverifiable_rate > self.max_unverifiable_rate:
            reasons.append("unverifiable_rate_exceeded")
        if self.salience_coverage < self.min_salience_coverage:
            reasons.append("salient_facts_dropped")
        return tuple(reasons)

    def summary(self) -> str:
        """Return a one-line human-readable verdict.

        Returns:
            The counts and the failing conditions, for a log line and for the discard
            record's ``final_verdict``.
        """
        classes = ",".join(f"{k}={v}" for k, v in sorted(self.by_hallucination_class.items()))
        return (
            f"{self.n_claims} claims: {self.supported} supported, "
            f"{self.contradicted} contradicted, {self.unverifiable} unverifiable "
            f"(rate {self.unverifiable_rate:.3f}); salience {self.salience_coverage:.3f}"
            + (f"; classes {classes}" if classes else "")
            + (f"; FAILS {'+'.join(self.failures())}" if self.failures() else "")
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the verification block written onto a record.

        Returns:
            A JSON-serialisable mapping. Note the harness recomputes all of this
            independently and does not trust what is written here.
        """
        return {
            "supported": self.supported,
            "contradicted": self.contradicted,
            "unverifiable": self.unverifiable,
            "unverifiable_rate": round(self.unverifiable_rate, 6),
            "n_claims": self.n_claims,
            "critical_error_rate": round(self.critical_error_rate, 6),
            "by_hallucination_class": dict(sorted(self.by_hallucination_class.items())),
        }


def verify_rewrite(
    narrative: str,
    case: CaseInput,
    *,
    vocabulary: ControlledVocabulary,
    config: SilverConfig | None = None,
) -> tuple[VerificationVerdict, ExtractionReport]:
    """Verify one rewrite against its own fact record.

    Runs the Phase 3 checker over claims parsed out of the rewrite, plus the text-level
    scan for forbidden phrases and missing hedges, plus the salience retention check that
    catches omission.

    Args:
        narrative: The canonicalised rewrite.
        case: The case, its facts and its Bronze reference.
        vocabulary: The controlled vocabulary.
        config: Thresholds. Defaults apply when omitted.

    Returns:
        ``(verdict, extraction report)``. The report is returned alongside so the caller
        can log which spans were added without re-running extraction.
    """
    cfg = config if config is not None else SilverConfig()
    context = CheckContext(facts=case.facts, vocabulary=vocabulary)
    extractor = SlotAlignmentExtractor(case.bronze, vocabulary=vocabulary)
    report = extractor.report(narrative, case.facts)

    results = [check_claim(claim, context) for claim in report.claims]
    text_results = check_narrative_text(narrative, context)
    totals = summarise(results + text_results)

    salience = salience_report(case.facts, vocabulary)
    aligned = set(report.aligned_paths)
    required = set(salience.required)
    kept = required & aligned
    coverage = len(kept) / len(required) if required else 1.0
    dropped = tuple(sorted(required - aligned))

    verdict = VerificationVerdict(
        n_claims=totals["n_claims"],
        supported=totals["by_verdict"]["supported"],
        contradicted=totals["by_verdict"]["contradicted"],
        unverifiable=totals["by_verdict"]["unverifiable"],
        unverifiable_rate=totals["unverifiable_rate"],
        critical_error_rate=totals["critical_error_rate"],
        by_hallucination_class=dict(totals["by_hallucination_class"]),
        salience_coverage=coverage,
        dropped_salient=dropped,
        added_spans=report.added_spans,
        violations=_violations_from(results + text_results, dropped, cfg),
        max_unverifiable_rate=cfg.max_unverifiable_rate,
        min_salience_coverage=cfg.min_salience_coverage,
    )
    return verdict, report


def _violations_from(
    results: Sequence[CheckResult], dropped_salient: Sequence[str], config: SilverConfig
) -> tuple[Violation, ...]:
    """Phrase the checker's adverse findings for a repair prompt.

    Args:
        results: Every check result for the rewrite.
        dropped_salient: Salient fields the rewrite no longer carries.
        config: Thresholds, unused beyond documentation of intent.

    Returns:
        One violation per adverse result, plus one per dropped salient field. Capped, so
        a badly degenerate rewrite does not produce a repair prompt longer than the fact
        record — past a dozen violations the useful instruction is "this needs redoing",
        and the attempt limit will get there anyway.
    """
    del config
    violations = [
        Violation(
            field_path=result.claim.field_path,
            quoted=result.claim.raw_text,
            verdict=result.verdict.value,
            hallucination_class=result.hallucination_class,
            reason=result.reason,
        )
        for result in results
        if result.verdict in (Verdict.CONTRADICTED, Verdict.UNVERIFIABLE)
    ]
    violations.extend(
        Violation(
            field_path=path,
            quoted="",
            verdict="omitted",
            hallucination_class="H9",
            reason=(
                "this salient fact is in the record and in the draft, but your rewrite "
                "does not state it. Add it back."
            ),
        )
        for path in dropped_salient
    )
    return tuple(violations[:12])


@dataclass
class DiscardRecord:
    """One case that could not be made to verify. A row in the paper's discard table.

    Attributes:
        case_id: The case.
        teacher: Which teacher was assigned it.
        model: The exact model string.
        typology: Its typology, so the table can be broken down by scheme.
        split: Its split, so a discard-driven imbalance is visible.
        reason: Machine-readable failure conditions.
        attempts: Repair attempts made before giving up.
        by_hallucination_class: Per-class violation counts at the final attempt.
        final_verdict: The checker's own summary line.
        quoted_additions: The spans the model introduced, verbatim, so the failure can be
            read rather than only counted.
        stage: Where it failed — verification, or an API error.
    """

    case_id: str
    teacher: str
    model: str
    typology: str
    split: str
    reason: tuple[str, ...]
    attempts: int
    by_hallucination_class: dict[str, int] = field(default_factory=dict)
    final_verdict: str = ""
    quoted_additions: tuple[str, ...] = ()
    stage: str = "verification"
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Return the discard as a JSONL row.

        Returns:
            The mapping written to ``silver_discards.jsonl``.
        """
        return {
            "case_id": self.case_id,
            "teacher": self.teacher,
            "model": self.model,
            "typology": self.typology,
            "split": self.split,
            "reason": list(self.reason),
            "attempts": self.attempts,
            "by_hallucination_class": dict(sorted(self.by_hallucination_class.items())),
            "final_verdict": self.final_verdict,
            "quoted_additions": list(self.quoted_additions),
            "stage": self.stage,
            "timestamp": self.timestamp,
        }


@dataclass
class GenerationOutcome:
    """What happened to one case.

    Exactly one of ``record`` and ``discard`` is set. Both being None is impossible and
    both being set would mean a case was written and logged as dropped.

    Attributes:
        case_id: The case.
        teacher: The teacher key.
        record: The accepted training record, when it verified.
        discard: The discard row, when it did not.
        attempts: Repair attempts made.
        responses: Every teacher response, for the usage report.
        verdicts: The verdict at each attempt, so a repair that made things worse is
            visible in the run report rather than only in the final state.
    """

    case_id: str
    teacher: str
    record: TrainingRecord | None = None
    discard: DiscardRecord | None = None
    attempts: int = 0
    responses: tuple[TeacherResponse, ...] = ()
    verdicts: tuple[VerificationVerdict, ...] = ()

    @property
    def accepted(self) -> bool:
        """Report whether this case produced a corpus record.

        Returns:
            True when a record was written.
        """
        return self.record is not None


def assign_teachers(cases: Sequence[CaseInput], specs: Sequence[TeacherSpec]) -> dict[str, str]:
    """Assign every case to a teacher, deterministically and balanced per stratum.

    **Not a bare hash mod two.** Hashing balances in expectation over the whole corpus and
    not within a stratum, and this corpus has strata that are small: ``stack`` has 60
    records, ``fan_in`` 70. A 60-case stratum split by a coin flip lands 40/20 often
    enough to matter, and "the open-weights teacher wrote most of the stack cases" is
    precisely the confound a two-teacher design exists to remove.

    So the ordering *within* each ``(typology, split)`` stratum is drawn from a hash — the
    determinism the corpus needs — and assignment then round-robins down that ordering.
    Every stratum is balanced to within one case, on every machine, without a seed.

    **Each stratum starts the round-robin at its own offset.** Starting every stratum at
    teacher zero balances the large strata correctly and hands *every* singleton stratum
    to the same teacher — which on a corpus with many rare typology/split combinations
    reproduces exactly the skew the stratification was introduced to remove. The offset is
    a hash of the stratum, so it is as reproducible as the ordering inside it.

    Args:
        cases: Every case to assign.
        specs: The teachers, in configuration order.

    Returns:
        Case id to teacher key.

    Raises:
        ValueError: If no teachers were given.
    """
    if not specs:
        raise ValueError("cannot assign cases with no teachers configured")

    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for case in cases:
        strata[(case.typology, case.split)].append(case.case_id)

    assignment: dict[str, str] = {}
    for stratum in sorted(strata):
        ordered = sorted(
            strata[stratum],
            key=lambda case_id: hashlib.sha256(f"{case_id}|teacher".encode()).hexdigest(),
        )
        offset = int.from_bytes(
            hashlib.sha256("|".join(stratum).encode()).digest()[:4], "big"
        ) % len(specs)
        for position, case_id in enumerate(ordered):
            assignment[case_id] = specs[(position + offset) % len(specs)].key
    return assignment


def teacher_balance_report(
    assignment: dict[str, str], cases: Sequence[CaseInput], *, kept: set[str] | None = None
) -> dict[str, Any]:
    """Report teacher balance overall and per stratum.

    Args:
        assignment: Case id to teacher key.
        cases: The cases, for their typology and split.
        kept: Case ids that survived verification and filtering. When given, the report
            covers the surviving corpus as well as the assignment — **the balance that
            matters is the one after filtering**, because a teacher whose outputs are
            disproportionately discarded leaves a skewed corpus behind however even the
            assignment was, and that asymmetry is itself a finding.

    Returns:
        Counts by teacher, by typology and by split, before and after filtering, with the
        maximum deviation from an even split.
    """
    by_case = {case.case_id: case for case in cases}

    def tally(ids: list[str]) -> dict[str, Any]:
        overall: Counter[str] = Counter()
        by_typology: dict[str, Counter[str]] = defaultdict(Counter)
        by_split: dict[str, Counter[str]] = defaultdict(Counter)
        for case_id in ids:
            teacher = assignment.get(case_id)
            case = by_case.get(case_id)
            if teacher is None or case is None:
                continue
            overall[teacher] += 1
            by_typology[case.typology][teacher] += 1
            by_split[case.split][teacher] += 1
        total = sum(overall.values())
        n_teachers = len(set(assignment.values())) or 1
        even = total / n_teachers if total else 0.0
        deviation = max((abs(v - even) for v in overall.values()), default=0.0)
        return {
            "n": total,
            "by_teacher": dict(sorted(overall.items())),
            "by_typology": {k: dict(sorted(v.items())) for k, v in sorted(by_typology.items())},
            "by_split": {k: dict(sorted(v.items())) for k, v in sorted(by_split.items())},
            "max_deviation_from_even": round(deviation, 3),
            "max_deviation_share": round(deviation / total, 6) if total else 0.0,
        }

    report: dict[str, Any] = {"assigned": tally(list(assignment))}
    if kept is not None:
        report["kept"] = tally([c for c in assignment if c in kept])
        report["retention_by_teacher"] = _retention(assignment, kept)
    return report


def _retention(assignment: dict[str, str], kept: set[str]) -> dict[str, Any]:
    """Compute per-teacher survival rates through verification and filtering.

    Args:
        assignment: Case id to teacher key.
        kept: Case ids that survived.

    Returns:
        Assigned, kept and retention rate per teacher, plus the spread between the best
        and worst. A large spread means one teacher's work is being dropped far more
        often, which changes what the surviving corpus is and belongs in the phase log.
    """
    assigned: Counter[str] = Counter(assignment.values())
    survived: Counter[str] = Counter(t for c, t in assignment.items() if c in kept)
    rates = {
        teacher: round(survived[teacher] / assigned[teacher], 6) if assigned[teacher] else 0.0
        for teacher in sorted(assigned)
    }
    return {
        "assigned": dict(sorted(assigned.items())),
        "kept": {t: survived[t] for t in sorted(assigned)},
        "retention_rate": rates,
        "retention_spread": round(max(rates.values()) - min(rates.values()), 6) if rates else 0.0,
    }


def generate_one(
    case: CaseInput,
    teacher: Teacher,
    *,
    vocabulary: ControlledVocabulary,
    config: SilverConfig | None = None,
    graph_ref: str = "",
    token_counter: TokenCounter | None = None,
) -> GenerationOutcome:
    """Run generate → verify → repair → discard for one case.

    The loop the phase brief specifies, with the repair bound and the discard as the only
    two exits. A case leaves this function either as a verified record or as a logged
    discard; there is no third state and no path that writes an unverified narrative.

    Args:
        case: The case, its facts and its Bronze draft.
        teacher: The assigned teacher.
        vocabulary: The controlled vocabulary.
        config: Thresholds.
        graph_ref: The ``<case store>#<case_id>`` reference for the record.
        token_counter: Counts tokens for the length block.

    Returns:
        The outcome.

    Raises:
        BudgetExceeded: Propagated so the run halts rather than logging every remaining
            case as a discard. A budget stop is not a data finding and must never be
            mistaken for one in the discard table.
    """
    cfg = config if config is not None else SilverConfig()
    counter = token_counter if token_counter is not None else get_token_counter()
    responses: list[TeacherResponse] = []
    verdicts: list[VerificationVerdict] = []

    prompt = build_rewrite_prompt(
        case.facts,
        case.bronze.text,
        case.bronze.annotated,
        vocabulary=vocabulary,
        min_words=cfg.min_words,
        max_words=cfg.max_words,
    )
    try:
        response = teacher.complete(prompt, case_id=case.case_id, kind="rewrite", attempt=0)
    except BudgetExceeded:
        raise
    except TeacherError as exc:
        return GenerationOutcome(
            case_id=case.case_id,
            teacher=teacher.spec.key,
            discard=_api_discard(case, teacher.spec, exc, attempts=0),
        )

    responses.append(response)
    narrative = canonicalise_narrative(response.text)
    verdict, report = verify_rewrite(narrative, case, vocabulary=vocabulary, config=cfg)
    verdicts.append(verdict)

    attempts = 0
    while not verdict.accepted and attempts < cfg.max_repair_attempts:
        attempts += 1
        repair = build_repair_prompt(
            case.facts,
            narrative,
            list(verdict.violations),
            vocabulary=vocabulary,
            min_words=cfg.min_words,
            max_words=cfg.max_words,
        )
        try:
            response = teacher.complete(
                repair, case_id=case.case_id, kind="repair", attempt=attempts
            )
        except BudgetExceeded:
            raise
        except TeacherError as exc:
            return GenerationOutcome(
                case_id=case.case_id,
                teacher=teacher.spec.key,
                discard=_api_discard(case, teacher.spec, exc, attempts=attempts),
                attempts=attempts,
                responses=tuple(responses),
                verdicts=tuple(verdicts),
            )
        responses.append(response)
        narrative = canonicalise_narrative(response.text)
        verdict, report = verify_rewrite(narrative, case, vocabulary=vocabulary, config=cfg)
        verdicts.append(verdict)

    if not verdict.accepted:
        return GenerationOutcome(
            case_id=case.case_id,
            teacher=teacher.spec.key,
            discard=DiscardRecord(
                case_id=case.case_id,
                teacher=teacher.spec.key,
                model=teacher.spec.model,
                typology=case.typology,
                split=case.split,
                reason=verdict.failures(),
                attempts=attempts,
                by_hallucination_class=dict(verdict.by_hallucination_class),
                final_verdict=verdict.summary(),
                quoted_additions=tuple(text for _, _, text in verdict.added_spans[:8]),
            ),
            attempts=attempts,
            responses=tuple(responses),
            verdicts=tuple(verdicts),
        )

    record = build_silver_record(
        case,
        narrative,
        report,
        verdict,
        teacher.spec,
        prompt_provenance=prompt.to_provenance(),
        attempts=attempts,
        graph_ref=graph_ref,
        vocabulary=vocabulary,
        counter=counter,
        tier=cfg.tier,
        seed=cfg.seed,
    )
    return GenerationOutcome(
        case_id=case.case_id,
        teacher=teacher.spec.key,
        record=record,
        attempts=attempts,
        responses=tuple(responses),
        verdicts=tuple(verdicts),
    )


def _api_discard(
    case: CaseInput, spec: TeacherSpec, exc: Exception, *, attempts: int
) -> DiscardRecord:
    """Build a discard row for a case lost to an API failure rather than a violation.

    Kept in the same log but tagged with a different ``stage``, because the two mean
    opposite things: an API failure is an operational fact about a run, and a
    verification failure is a result about a model. Aggregating them into one discard
    rate would inflate the number the paper reports.

    Args:
        case: The case.
        spec: The teacher.
        exc: The failure.
        attempts: Repair attempts made before it happened.

    Returns:
        The discard row.
    """
    return DiscardRecord(
        case_id=case.case_id,
        teacher=spec.key,
        model=spec.model,
        typology=case.typology,
        split=case.split,
        reason=("api_error",),
        attempts=attempts,
        final_verdict=f"{type(exc).__name__}: {exc}",
        stage="api",
    )


def build_silver_record(
    case: CaseInput,
    narrative: str,
    report: ExtractionReport,
    verdict: VerificationVerdict,
    spec: TeacherSpec,
    *,
    prompt_provenance: dict[str, str],
    attempts: int,
    graph_ref: str,
    vocabulary: ControlledVocabulary,
    counter: TokenCounter,
    tier: str = "silver",
    seed: int = 42,
) -> TrainingRecord:
    """Wrap a verified rewrite in a training record.

    The record carries the same schema Bronze writes — one document for three tiers
    (D-037) — differing only in ``tier`` and ``generator``. The ``generator`` block is
    where the provenance requirement lands: model, exact model served, decoding settings,
    prompt hash, repair count and timestamp, on every record, so a distribution shift can
    later be attributed to a specific prompt and a specific model rather than guessed at.

    Args:
        case: The case.
        narrative: The canonicalised, verified rewrite.
        report: Its extraction report, which supplies the surviving slot alignment.
        verdict: Its verification verdict.
        spec: The teacher.
        prompt_provenance: Prompt name and hashes.
        attempts: Repair attempts made.
        graph_ref: The case-store reference.
        vocabulary: The controlled vocabulary.
        counter: Token counter for the length block.
        tier: Corpus tier.
        seed: The run's global seed.

    Returns:
        The record.
    """
    salience = salience_report(case.facts, vocabulary)
    mentioned = [p for p in salience.required if p in set(report.aligned_paths)]
    slots = _surviving_slots(case.bronze, narrative, report)

    return TrainingRecord(
        case_id=case.case_id,
        dataset=case.facts.dataset,
        split=case.split,
        tier=tier,
        facts=case.facts,
        graph_ref=graph_ref,
        serialised_facts=serialise_facts(case.facts, style="compact"),
        target_narrative=narrative,
        target_slots=slots,
        generator={
            # The frozen training-record schema enumerates method as template |
            # llm_rewrite | human. Silver is llm_rewrite; "verified" is a property of the
            # pipeline, recorded in the verification block, not a fourth method.
            "method": "llm_rewrite",
            "teacher": spec.key,
            "family": spec.family,
            "provider": spec.provider,
            "model": spec.model,
            "source_tier": "bronze",
            "source_family": case.bronze.family,
            "source_variant": case.bronze.variant,
            # Required by the frozen training-record schema, and meaningful here rather
            # than vestigial: a Silver record is a rewrite of a specific Bronze rendering,
            # so the renderer that produced its input is part of its provenance.
            "renderer_version": case.bronze.renderer_version,
            # The run's global seed (invariant 5), an integer on every tier by schema.
            # A teacher's decoding seed is `provider_seed` and is legitimately null.
            "seed": seed,
            "repair_attempts": attempts,
            "generated_at": datetime.now(UTC).isoformat(),
            **spec.sampling_provenance(),
            **prompt_provenance,
        },
        verification=verdict.to_dict(),
        length={
            "n_tokens": counter.count(narrative),
            "n_words": word_count(narrative),
            "n_chars": len(narrative),
            "tokenizer": counter.name,
        },
        salience={
            "required": list(salience.required),
            "excused": list(salience.excused),
            "mentioned": mentioned,
            "coverage": round(verdict.salience_coverage, 6),
        },
    )


def _surviving_slots(
    bronze: BronzeNarrative, narrative: str, report: ExtractionReport
) -> tuple[SlotAnnotation, ...]:
    """Rebuild the slot alignment against the rewrite's own character offsets.

    **A slot the rewrite dropped must not be carried over.** Keeping a Bronze annotation
    whose span no longer holds its value would put a lie in the record: the harness
    asserts ``narrative[span] == rendered_value`` per slot, and Phase 10's faithfulness
    evaluation aligns against exactly these spans. A dropped slot shows up as reduced
    salience coverage, which is the honest signal, rather than as a silent mis-alignment.

    Args:
        bronze: The Bronze narrative, for the slot metadata.
        narrative: The rewrite.
        report: The extraction report, whose claims carry the rewrite's spans.

    Returns:
        The annotations that survived, in document order.
    """
    by_path: dict[str, list[SlotAnnotation]] = defaultdict(list)
    for slot in bronze.slots:
        by_path[slot.field_path].append(slot)

    survivors: list[SlotAnnotation] = []
    used: Counter[str] = Counter()
    for claim in report.claims:
        if claim.field_path is None:
            continue
        candidates = by_path.get(claim.field_path, [])
        index = used[claim.field_path]
        if index >= len(candidates):
            continue
        used[claim.field_path] += 1
        original = candidates[index]
        start, end = claim.text_span
        if narrative[start:end] != original.rendered_value:
            continue
        survivors.append(
            SlotAnnotation(
                field_path=original.field_path,
                span=(start, end),
                rendered_value=original.rendered_value,
                raw_value=original.raw_value,
                claim_type=original.claim_type,
            )
        )
    survivors.sort(key=lambda s: s.span)
    return tuple(survivors)


def write_discards(discards: Sequence[DiscardRecord], path: Path) -> None:
    """Write the discard log.

    Args:
        discards: Every discarded case.
        path: Destination JSONL.
    """
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for discard in discards:
            handle.write(json.dumps(discard.to_dict(), sort_keys=True) + "\n")


def discard_report(discards: Sequence[DiscardRecord], n_attempted: int) -> dict[str, Any]:
    """Aggregate the discard log into the table that goes in the paper.

    Args:
        discards: Every discarded case.
        n_attempted: Cases attempted, the denominator.

    Returns:
        The discard rate and its breakdowns by reason, hallucination class, teacher,
        typology, split and attempt count. The verification rate and the API-failure rate
        are reported separately — one is a result about models, the other is an
        operational fact about a run.
    """
    verification = [d for d in discards if d.stage == "verification"]
    api = [d for d in discards if d.stage == "api"]

    by_reason: Counter[str] = Counter()
    by_class: Counter[str] = Counter()
    by_teacher: Counter[str] = Counter()
    by_typology: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    by_attempts: Counter[int] = Counter()
    for discard in verification:
        by_reason.update(discard.reason)
        by_class.update(discard.by_hallucination_class)
        by_teacher[discard.teacher] += 1
        by_typology[discard.typology] += 1
        by_split[discard.split] += 1
        by_attempts[discard.attempts] += 1

    return {
        "n_attempted": n_attempted,
        "n_discarded": len(discards),
        "discard_rate": round(len(discards) / n_attempted, 6) if n_attempted else 0.0,
        "n_discarded_verification": len(verification),
        "verification_discard_rate": (
            round(len(verification) / n_attempted, 6) if n_attempted else 0.0
        ),
        "n_discarded_api": len(api),
        "api_discard_rate": round(len(api) / n_attempted, 6) if n_attempted else 0.0,
        "by_reason": dict(sorted(by_reason.items())),
        "by_hallucination_class": dict(sorted(by_class.items())),
        "by_teacher": dict(sorted(by_teacher.items())),
        "by_typology": dict(sorted(by_typology.items())),
        "by_split": dict(sorted(by_split.items())),
        "by_attempts": {str(k): v for k, v in sorted(by_attempts.items())},
    }
