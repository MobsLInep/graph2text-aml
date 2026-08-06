"""How much the two extractors agree, measured rather than assumed.

This is the module that makes the faithfulness metric defensible. An automatic metric
validated against nothing is a number with a method section; the κ computed here is what
lets the paper claim that Method A's verdicts are not an artefact of Method A.

**Three agreements are reported, and they are not interchangeable.**

*Verdict agreement* — Cohen's κ over ``{supported, contradicted, unverifiable}`` on
claims the two methods both found. The headline κ, and the weakest of the three to
over-read: it is conditioned on the claims they both found, so a method that finds half
the claims can still show high verdict κ.

*Boundary agreement* — Cohen's κ over token-level claim membership: for each word of the
narrative, did Method A place it inside a claim, and did Method B? This is what catches
the failure verdict κ cannot see, namely two methods that agree on everything they both
look at while looking at different things. Computed per token rather than per span
because a span-level κ needs a chance model over an unbounded set of possible spans,
which does not exist.

*Decision agreement* — per-narrative agreement on the one binary that is actually
reported: does this narrative contain at least one contradicted claim? Zero-Hallucination
Rate is the headline number, so agreement on the decision behind it matters more than
agreement on the claims behind the decision, and the two can differ sharply — a hundred
claim-level disagreements spread across a hundred narratives that all still contain some
other contradiction change the headline not at all.

**Matching is by span overlap, not by text.** Two methods that phrase a claim differently
have still found the same claim if they point at the same words; requiring the restated
text to match would measure paraphrase similarity and call it extractor agreement.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from g2t_aml.eval.claim_extraction.llm_based import AtomicClaim
from g2t_aml.facts.checkers import CheckResult, Verdict

__all__ = [
    "AGREEMENT_SAMPLE_SIZE",
    "MATCH_IOU_THRESHOLD",
    "AgreementCase",
    "AgreementReport",
    "BoundaryAlignment",
    "cohens_kappa",
    "interpret_kappa",
    "measure_agreement",
]

#: The sample the agreement is measured on, fixed by the Phase 10 brief. Three hundred
#: cases: large enough that a κ standard error is a couple of hundredths, small enough
#: that Method B's two calls per case stay affordable.
AGREEMENT_SAMPLE_SIZE = 300

#: Intersection-over-union above which two spans are taken to be the same claim. Set at a
#: half — the two methods must agree on the majority of the words, not merely touch. Fixed
#: here, before any agreement was computed, because a matching threshold tuned after
#: seeing the κ is a threshold chosen to produce a κ.
MATCH_IOU_THRESHOLD = 0.5

#: Word tokens, for the token-level boundary comparison. Deliberately the same shape as
#: the diversity tokenizer: characters would inflate agreement by counting the whitespace
#: both methods trivially agree is outside every claim.
_TOKEN_RE = re.compile(r"\S+")


def cohens_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Compute Cohen's κ between two label sequences.

    Args:
        a: The first rater's labels.
        b: The second rater's labels, index-aligned with ``a``.

    Returns:
        κ in [-1, 1]. Returns 1.0 when the two agree completely on a single label — the
        degenerate case where expected agreement is also 1 and the usual formula is 0/0.
        That convention is stated rather than hidden: a narrative set on which both
        methods find nothing but supported claims genuinely is a set they agree on, and
        reporting NaN there would drop those cases out of a mean and quietly bias it.

    Raises:
        ValueError: If the sequences are different lengths, or empty.
    """
    if len(a) != len(b):
        raise ValueError(f"κ needs paired labels; got {len(a)} and {len(b)}")
    if not a:
        raise ValueError("κ is undefined over an empty sample")

    n = len(a)
    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    labels = set(a) | set(b)
    expected = sum((a.count(label) / n) * (b.count(label) / n) for label in labels)
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def interpret_kappa(kappa: float) -> str:
    """Return the conventional verbal band for a κ.

    Landis and Koch's bands, quoted because a reviewer expects them, and reported beside
    the number rather than instead of it.

    Args:
        kappa: The coefficient.

    Returns:
        One of ``"poor"``, ``"slight"``, ``"fair"``, ``"moderate"``, ``"substantial"``,
        ``"almost perfect"``.
    """
    for threshold, name in (
        (0.0, "poor"),
        (0.20, "slight"),
        (0.40, "fair"),
        (0.60, "moderate"),
        (0.80, "substantial"),
    ):
        if kappa <= threshold:
            return name
    return "almost perfect"


@dataclass(frozen=True)
class AgreementCase:
    """One case scored by both methods.

    Attributes:
        case_id: The case.
        narrative: The narrative both methods read, needed for the token-level boundary
            comparison.
        method_a: Method A's results — claims with the checker's verdict on each.
        method_b: Method B's claims, each carrying the judge's verdict.
    """

    case_id: str
    narrative: str
    method_a: tuple[CheckResult, ...]
    method_b: tuple[AtomicClaim, ...]


@dataclass(frozen=True)
class BoundaryAlignment:
    """How the two methods' claim spans line up on one case.

    Attributes:
        matched: ``(a_index, b_index, iou)`` for each matched pair.
        unmatched_a: Indices of Method A claims with no Method B counterpart.
        unmatched_b: Indices of Method B claims with no Method A counterpart.
        mean_iou: Mean IoU over matched pairs, 0.0 when nothing matched.
    """

    matched: tuple[tuple[int, int, float], ...]
    unmatched_a: tuple[int, ...]
    unmatched_b: tuple[int, ...]
    mean_iou: float

    @property
    def span_f1(self) -> float:
        """Return the F1 of Method B's spans against Method A's.

        Reported alongside the boundary κ because F1 says *how much* was found in common
        and κ says how much of that is beyond chance. Neither substitutes for the other.

        Returns:
            The F1 in [0, 1], and 0.0 when either method found nothing.
        """
        n_matched = len(self.matched)
        n_a = n_matched + len(self.unmatched_a)
        n_b = n_matched + len(self.unmatched_b)
        if not n_a or not n_b or not n_matched:
            return 0.0
        precision, recall = n_matched / n_b, n_matched / n_a
        return 2 * precision * recall / (precision + recall)


def _iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Return the intersection-over-union of two character spans.

    Args:
        a: The first span, half-open.
        b: The second span, half-open.

    Returns:
        The IoU in [0, 1], and 0.0 when either span is empty.
    """
    intersection = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return intersection / union if union > 0 else 0.0


def align_spans(
    a_spans: Sequence[tuple[int, int]],
    b_spans: Sequence[tuple[int, int]],
    *,
    threshold: float = MATCH_IOU_THRESHOLD,
) -> BoundaryAlignment:
    """Match two sets of claim spans greedily by overlap.

    Greedy on descending IoU, and each span may match at most once. Greedy rather than
    optimal because the alternative — a maximum-weight bipartite matching — changes the
    result only when spans overlap ambiguously, and a claim set where that is common has
    a boundary problem the matching algorithm should not paper over.

    Args:
        a_spans: Method A's spans.
        b_spans: Method B's spans.
        threshold: Minimum IoU for a pair to count as the same claim.

    Returns:
        The alignment.
    """
    candidates = sorted(
        (
            (_iou(a, b), i, j)
            for i, a in enumerate(a_spans)
            for j, b in enumerate(b_spans)
            if _iou(a, b) >= threshold
        ),
        key=lambda triple: (-triple[0], triple[1], triple[2]),
    )
    used_a: set[int] = set()
    used_b: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for iou, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched.append((i, j, iou))
    return BoundaryAlignment(
        matched=tuple(sorted(matched)),
        unmatched_a=tuple(i for i in range(len(a_spans)) if i not in used_a),
        unmatched_b=tuple(j for j in range(len(b_spans)) if j not in used_b),
        mean_iou=sum(iou for _, _, iou in matched) / len(matched) if matched else 0.0,
    )


def _token_membership(narrative: str, spans: Sequence[tuple[int, int]]) -> list[str]:
    """Label each word of the narrative as inside or outside a claim.

    Args:
        narrative: The narrative.
        spans: Claim spans, half-open character offsets.

    Returns:
        ``"in"`` or ``"out"`` per whitespace-delimited token, in document order.
    """
    return [
        "in" if any(m.start() < end and start < m.end() for start, end in spans) else "out"
        for m in _TOKEN_RE.finditer(narrative)
    ]


@dataclass(frozen=True)
class AgreementReport:
    """The measured agreement between the two extractors.

    Attributes:
        n_cases: Cases both methods ran on.
        n_matched_claims: Claim pairs the two methods both found.
        n_only_a: Claims only Method A found.
        n_only_b: Claims only Method B found.
        verdict_kappa: Cohen's κ on verdicts over matched claims.
        verdict_observed_agreement: Raw agreement on the same, reported because κ over a
            skewed label distribution can be low while raw agreement is high, and a
            reader who sees only κ will misread that as a broken metric.
        boundary_kappa: Cohen's κ on token-level claim membership.
        span_f1: Mean per-case span F1.
        mean_iou: Mean IoU over matched pairs.
        decision_kappa: κ on the per-narrative "contains at least one contradicted claim"
            decision — the binary behind Zero-Hallucination Rate.
        decision_observed_agreement: Raw agreement on that decision.
        confusion: Method A verdict to Method B verdict to count, over matched claims.
        unlocated_b: Method B claims whose evidence could not be located in the narrative
            and which therefore took no part in boundary agreement.
        sample_size_target: :data:`AGREEMENT_SAMPLE_SIZE`, carried so a report says
            whether it met the sample the protocol calls for.
    """

    n_cases: int
    n_matched_claims: int
    n_only_a: int
    n_only_b: int
    verdict_kappa: float | None
    verdict_observed_agreement: float | None
    boundary_kappa: float | None
    span_f1: float
    mean_iou: float
    decision_kappa: float | None
    decision_observed_agreement: float | None
    confusion: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    unlocated_b: int = 0
    sample_size_target: int = AGREEMENT_SAMPLE_SIZE

    @property
    def meets_sample_target(self) -> bool:
        """Report whether the agreement was measured on the protocol's sample size.

        Returns:
            True when at least :data:`AGREEMENT_SAMPLE_SIZE` cases were scored.
        """
        return self.n_cases >= self.sample_size_target

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-serialisable mapping.

        Returns:
            Every field, plus the verbal band for each κ.
        """
        return {
            "n_cases": self.n_cases,
            "meets_sample_target": self.meets_sample_target,
            "sample_size_target": self.sample_size_target,
            "n_matched_claims": self.n_matched_claims,
            "n_only_a": self.n_only_a,
            "n_only_b": self.n_only_b,
            "unlocated_b": self.unlocated_b,
            "verdict_kappa": self.verdict_kappa,
            "verdict_kappa_band": (
                interpret_kappa(self.verdict_kappa) if self.verdict_kappa is not None else None
            ),
            "verdict_observed_agreement": self.verdict_observed_agreement,
            "boundary_kappa": self.boundary_kappa,
            "boundary_kappa_band": (
                interpret_kappa(self.boundary_kappa) if self.boundary_kappa is not None else None
            ),
            "span_f1": self.span_f1,
            "mean_iou": self.mean_iou,
            "decision_kappa": self.decision_kappa,
            "decision_kappa_band": (
                interpret_kappa(self.decision_kappa) if self.decision_kappa is not None else None
            ),
            "decision_observed_agreement": self.decision_observed_agreement,
            "confusion": {a: dict(row) for a, row in self.confusion.items()},
        }


def measure_agreement(
    cases: Sequence[AgreementCase], *, threshold: float = MATCH_IOU_THRESHOLD
) -> AgreementReport:
    """Measure verdict, boundary and decision agreement across a sample.

    Args:
        cases: The cases both methods scored.
        threshold: IoU above which two spans are the same claim.

    Returns:
        The agreement report. κ values are None when the sample offers nothing to compute
        them on — no matched claims, or no case at all — rather than zero, which would be
        read as a measured disagreement.
    """
    verdicts_a: list[str] = []
    verdicts_b: list[str] = []
    tokens_a: list[str] = []
    tokens_b: list[str] = []
    decisions_a: list[str] = []
    decisions_b: list[str] = []
    confusion: dict[str, dict[str, int]] = {
        v.value: dict.fromkeys((w.value for w in Verdict), 0) for v in Verdict
    }
    n_only_a = n_only_b = 0
    unlocated_b = 0
    span_f1_total = 0.0
    iou_total = 0.0
    n_matched = 0

    for case in cases:
        a_spans = [result.claim.text_span for result in case.method_a]
        located_b = [(i, c) for i, c in enumerate(case.method_b) if c.span is not None]
        unlocated_b += len(case.method_b) - len(located_b)
        b_spans = [c.span for _, c in located_b if c.span is not None]

        alignment = align_spans(a_spans, b_spans, threshold=threshold)
        span_f1_total += alignment.span_f1
        iou_total += alignment.mean_iou * len(alignment.matched)
        n_matched += len(alignment.matched)
        n_only_a += len(alignment.unmatched_a)
        n_only_b += len(alignment.unmatched_b)

        for i, j, _ in alignment.matched:
            verdict_b = located_b[j][1].verdict
            if verdict_b is None:
                continue
            label_a = case.method_a[i].verdict.value
            label_b = verdict_b.value
            verdicts_a.append(label_a)
            verdicts_b.append(label_b)
            confusion[label_a][label_b] += 1

        tokens_a.extend(_token_membership(case.narrative, a_spans))
        tokens_b.extend(_token_membership(case.narrative, b_spans))

        decisions_a.append(
            "hallucinated"
            if any(r.verdict is Verdict.CONTRADICTED for r in case.method_a)
            else "clean"
        )
        decisions_b.append(
            "hallucinated"
            if any(c.verdict is Verdict.CONTRADICTED for c in case.method_b)
            else "clean"
        )

    def _observed(a: Sequence[str], b: Sequence[str]) -> float | None:
        """Return raw agreement, or None over an empty pairing."""
        if not a:
            return None
        return sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(a)

    return AgreementReport(
        n_cases=len(cases),
        n_matched_claims=len(verdicts_a),
        n_only_a=n_only_a,
        n_only_b=n_only_b,
        verdict_kappa=cohens_kappa(verdicts_a, verdicts_b) if verdicts_a else None,
        verdict_observed_agreement=_observed(verdicts_a, verdicts_b),
        boundary_kappa=cohens_kappa(tokens_a, tokens_b) if tokens_a else None,
        span_f1=span_f1_total / len(cases) if cases else 0.0,
        mean_iou=iou_total / n_matched if n_matched else 0.0,
        decision_kappa=cohens_kappa(decisions_a, decisions_b) if decisions_a else None,
        decision_observed_agreement=_observed(decisions_a, decisions_b),
        confusion=confusion,
        unlocated_b=unlocated_b,
    )
