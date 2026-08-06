"""Calibration: ten cases every annotator writes before any of their work counts.

**Skipping this is how you discover in week six that one annotator has been systematically
mislabelling scatter-gather**, and by then their forty items are unusable and there is no
budget to redo them. Ten items costs each annotator about two and a half hours. Forty
wasted items costs ten, plus the schedule.

Each annotator writes the same ten cases; their output is scored against a reference set
written by the project lead. Four dimensions, scored separately and thresholded separately
— **never averaged**, because the failure modes are not interchangeable:

- **Typology agreement.** Do they see the same shape the reference does? The dimension that
  catches the systematic error, because a person who confuses scatter-gather with
  gather-scatter does it every time.
- **Salience coverage.** Do they mention what the case's typology requires? Catches the
  annotator who writes fluently and omits the width.
- **Hedging compliance.** Threshold 1.00, deliberately. A guilt overclaim is a critical
  error, and an annotator who produces one in a ten-item calibration will produce them in
  a corpus. There is no partial credit for mostly not asserting guilt.
- **Factual accuracy.** Does the checker find contradictions against the fact record? The
  only dimension measured against the record rather than against another human, which is
  what stops calibration from being a test of agreeing with the project lead.

The feedback is per-dimension and quotes the specific items, because "your typology
agreement is 0.6" is not something anyone can act on and "you called both of these
gather-scatter; the reference says the collect phase completes before the disperse phase
begins, which makes them scatter-gather" is.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from g2t_aml.facts.checkers import CheckContext, Verdict, check_narrative_text
from g2t_aml.facts.schema import CaseFacts
from g2t_aml.facts.vocab import ControlledVocabulary, load_vocabulary
from g2t_aml.human.agreement import jaccard
from g2t_aml.human.store import Annotation

__all__ = [
    "DIMENSIONS",
    "AnnotatorCalibration",
    "CalibrationItem",
    "CalibrationSet",
    "CalibrationError",
    "DimensionScore",
    "build_calibration_set",
    "score_annotator",
]

#: How many items a confusion must span before it is reported as systematic rather than
#: as a slip. Two is the smallest number that can show a pattern in a ten-item set.
_SYSTEMATIC_CONFUSION_MIN = 2

#: The four dimensions, with their default pass thresholds. Mirrors
#: ``configs/corpus/gold.yaml``; the config is what a run reads, this is the default a
#: caller gets when none is supplied.
DIMENSIONS: dict[str, float] = {
    "typology_agreement": 0.70,
    "salience_coverage": 0.70,
    "hedging_compliance": 1.00,
    "factual_accuracy": 0.95,
}


class CalibrationError(RuntimeError):
    """Raised when a calibration set or a scoring run is not well formed."""


@dataclass(frozen=True)
class CalibrationItem:
    """One calibration case and the project lead's reference answer.

    Attributes:
        case_id: The case.
        reference_typology: The typology the lead assigned.
        reference_narrative: The lead's narrative. **Never shown to an annotator before
            they submit** — it is the answer key, and the interface has no route to it
            (see :mod:`g2t_aml.human.caseloader`). Shown afterwards, with the commentary,
            because that is where the learning is.
        reference_mentioned: Salient fields the reference narrative mentions.
        commentary: Why the reference says what it says. This is the teaching material and
            is the reason a calibration item is worth more than a test item.
    """

    case_id: str
    reference_typology: str
    reference_narrative: str
    reference_mentioned: tuple[str, ...] = ()
    commentary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised item.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "case_id": self.case_id,
            "reference_typology": self.reference_typology,
            "reference_narrative": self.reference_narrative,
            "reference_mentioned": list(self.reference_mentioned),
            "commentary": self.commentary,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CalibrationItem:
        """Rebuild an item from its serialised form.

        Args:
            payload: The mapping.

        Returns:
            The item.

        Raises:
            CalibrationError: If a required field is missing.
        """
        try:
            return cls(
                case_id=str(payload["case_id"]),
                reference_typology=str(payload["reference_typology"]),
                reference_narrative=str(payload["reference_narrative"]),
                reference_mentioned=tuple(payload.get("reference_mentioned") or ()),
                commentary=str(payload.get("commentary", "")),
            )
        except KeyError as exc:
            raise CalibrationError(f"calibration item is missing {exc}") from exc


@dataclass(frozen=True)
class CalibrationSet:
    """The ten cases, and the references for them.

    Attributes:
        items: The calibration items, in the order annotators work them.
        drawn_from: What population they were drawn from, for the record.
        seed: The seed the draw used.
    """

    items: tuple[CalibrationItem, ...]
    drawn_from: str = ""
    seed: int = 0

    def __len__(self) -> int:
        """Return how many calibration items there are.

        Returns:
            The count.
        """
        return len(self.items)

    @property
    def case_ids(self) -> tuple[str, ...]:
        """Return the calibration case ids, in working order.

        Returns:
            The ids.
        """
        return tuple(i.case_id for i in self.items)

    def by_case(self) -> dict[str, CalibrationItem]:
        """Index the items by case.

        Returns:
            Case id to item.
        """
        return {i.case_id: i for i in self.items}

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised set.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "n_items": len(self.items),
            "drawn_from": self.drawn_from,
            "seed": self.seed,
            "items": [i.to_dict() for i in self.items],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CalibrationSet:
        """Rebuild a set from its serialised form.

        Args:
            payload: The mapping.

        Returns:
            The set.
        """
        return cls(
            items=tuple(CalibrationItem.from_dict(i) for i in payload.get("items") or ()),
            drawn_from=str(payload.get("drawn_from", "")),
            seed=int(payload.get("seed", 0)),
        )


def build_calibration_set(
    candidates: list[tuple[str, str]],
    *,
    n_cases: int = 10,
    seed: int = 42,
    drawn_from: str = "",
) -> CalibrationSet:
    """Choose the calibration cases, spread across typologies.

    The set must span typologies rather than sample them proportionally: ten cases drawn
    in proportion to the Gold sample would be eight ``unclassified`` ones and would
    calibrate nobody on the eight typed shapes, which is exactly where the systematic
    errors live.

    Args:
        candidates: ``(case_id, typology)`` for every case eligible for calibration,
            normally the Gold sample itself so calibration exercises the real
            distribution's cases even though it does not follow its proportions.
        n_cases: How many items the set holds.
        seed: Selects the deterministic order inside a typology.
        drawn_from: A note recorded on the set, e.g. the sample file it came from.

    Returns:
        The set, with empty references. The project lead fills them in; a set whose
        references are blank is a set nobody can be scored against, and
        :func:`score_annotator` says so rather than scoring everyone at 1.0.

    Raises:
        CalibrationError: If there are fewer candidates than requested items.
    """
    if len(candidates) < n_cases:
        raise CalibrationError(
            f"{len(candidates)} candidates cannot supply a {n_cases}-item calibration set"
        )

    by_typology: dict[str, list[str]] = {}
    for case_id, typology in candidates:
        by_typology.setdefault(typology, []).append(case_id)
    for pool in by_typology.values():
        pool.sort(key=lambda c: hashlib.sha256(f"{seed}:{c}".encode()).hexdigest())

    chosen: list[tuple[str, str]] = []
    typologies = sorted(by_typology)
    while len(chosen) < n_cases and any(by_typology.values()):
        for typology in typologies:
            if len(chosen) >= n_cases:
                break
            if by_typology[typology]:
                chosen.append((by_typology[typology].pop(0), typology))

    return CalibrationSet(
        items=tuple(
            CalibrationItem(case_id=case_id, reference_typology=typology, reference_narrative="")
            for case_id, typology in chosen
        ),
        drawn_from=drawn_from,
        seed=seed,
    )


@dataclass(frozen=True)
class DimensionScore:
    """One dimension's score for one annotator.

    Attributes:
        name: The dimension.
        score: The value, in [0, 1].
        threshold: What it had to reach.
        detail: Per-item notes, quoted back in the feedback.
    """

    name: str
    score: float
    threshold: float
    detail: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        """Report whether the dimension met its threshold.

        Returns:
            True when the score is at or above the threshold.
        """
        return self.score >= self.threshold

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised score.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "name": self.name,
            "score": round(self.score, 6),
            "threshold": self.threshold,
            "passed": self.passed,
            "detail": list(self.detail),
        }


@dataclass(frozen=True)
class AnnotatorCalibration:
    """One annotator's calibration result.

    Attributes:
        annotator_id: The pseudonym.
        n_items: How many calibration items they completed.
        n_expected: How many the set holds.
        dimensions: The four scores.
        mean_minutes: Their mean time per item — the number the schedule is built on.
        guidance: Targeted feedback, one line per problem found.
    """

    annotator_id: str
    n_items: int
    n_expected: int
    dimensions: tuple[DimensionScore, ...]
    mean_minutes: float = 0.0
    guidance: tuple[str, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        """Report whether the annotator may begin real annotation.

        Every dimension must pass, and the set must be complete. An annotator who did
        eight of ten items has not calibrated, however well they did on the eight.

        Returns:
            True when calibration is complete and every dimension met its threshold.
        """
        return self.n_items >= self.n_expected and all(d.passed for d in self.dimensions)

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised result.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "annotator_id": self.annotator_id,
            "n_items": self.n_items,
            "n_expected": self.n_expected,
            "complete": self.n_items >= self.n_expected,
            "passed": self.passed,
            "mean_minutes": round(self.mean_minutes, 2),
            "dimensions": [d.to_dict() for d in self.dimensions],
            "guidance": list(self.guidance),
        }

    def summary(self) -> str:
        """Return the human-readable report handed back to the annotator.

        Returns:
            A short report.
        """
        verdict = "PASSED" if self.passed else "NOT YET PASSED"
        lines = [
            f"Calibration — {self.annotator_id}: {verdict}",
            f"  {self.n_items} of {self.n_expected} items, "
            f"{self.mean_minutes:.1f} minutes each on average",
            "",
            f"  {'dimension':<22} {'score':>7} {'needs':>7}",
            f"  {'-' * 22} {'-' * 7} {'-' * 7}",
        ]
        for dimension in self.dimensions:
            mark = " " if dimension.passed else "!"
            lines.append(
                f" {mark}{dimension.name:<22} {dimension.score:>7.3f} "
                f"{dimension.threshold:>7.2f}"
            )
        if self.guidance:
            lines += ["", "  What to work on:"]
            lines += [f"    - {g}" for g in self.guidance]
        return "\n".join(lines)


def score_annotator(
    annotator_id: str,
    annotations: list[Annotation],
    calibration: CalibrationSet,
    facts_by_case: dict[str, CaseFacts],
    *,
    mentioned_by_case: dict[str, tuple[str, ...]] | None = None,
    thresholds: dict[str, float] | None = None,
    vocabulary: ControlledVocabulary | None = None,
) -> AnnotatorCalibration:
    """Score one annotator's calibration against the reference set.

    Args:
        annotator_id: The pseudonym.
        annotations: Their calibration submissions. Anything outside the calibration set
            is ignored rather than scored.
        calibration: The reference set.
        facts_by_case: The fact record for each calibration case, for the factual-accuracy
            dimension.
        mentioned_by_case: Salient fields the annotator mentioned per case, from
            ingestion's alignment. When omitted the salience dimension is scored 0.0 and
            said to be unmeasured, never waved through — a dimension that silently passes
            because it was not measured is worse than one that fails.
        thresholds: Per-dimension pass thresholds. :data:`DIMENSIONS` when omitted.
        vocabulary: The controlled vocabulary. Loaded from disk when omitted.

    Returns:
        The result, with targeted guidance.

    Raises:
        CalibrationError: If the reference set has no filled-in references, which would
            score every annotator as perfectly agreeing with nothing.
    """
    references = calibration.by_case()
    if not any(item.reference_narrative.strip() for item in calibration.items):
        raise CalibrationError(
            "the calibration set has no reference narratives. Scoring against blank "
            "references would pass every annotator; the project lead writes the "
            "references before anyone calibrates."
        )
    vocab = vocabulary if vocabulary is not None else load_vocabulary()
    limits = dict(DIMENSIONS) | dict(thresholds or {})

    mine = [a for a in annotations if a.annotator_id == annotator_id and a.case_id in references]
    if not mine:
        return AnnotatorCalibration(
            annotator_id=annotator_id,
            n_items=0,
            n_expected=len(calibration),
            dimensions=tuple(
                DimensionScore(name, 0.0, limits[name], ("no items submitted",))
                for name in DIMENSIONS
            ),
            guidance=("No calibration items have been submitted.",),
        )

    typology_hits: list[bool] = []
    typology_detail: list[str] = []
    salience_scores: list[float] = []
    hedging_hits: list[bool] = []
    hedging_detail: list[str] = []
    accuracy_hits: list[bool] = []
    accuracy_detail: list[str] = []
    confusions: Counter[str] = Counter()

    for annotation in sorted(mine, key=lambda a: a.case_id):
        item = references[annotation.case_id]

        agreed = annotation.typology_assigned == item.reference_typology
        typology_hits.append(agreed)
        if not agreed:
            confusions[f"{annotation.typology_assigned} for {item.reference_typology}"] += 1
            typology_detail.append(
                f"{annotation.case_id}: you assigned {annotation.typology_assigned!r}, "
                f"the reference says {item.reference_typology!r}"
                + (f" — {item.commentary}" if item.commentary else "")
            )

        if mentioned_by_case is not None:
            mine_mentioned = set(mentioned_by_case.get(annotation.case_id, ()))
            reference_mentioned = set(item.reference_mentioned)
            salience_scores.append(jaccard(mine_mentioned, reference_mentioned))

        facts = facts_by_case.get(annotation.case_id)
        if facts is None:
            continue
        context = CheckContext(facts=facts, vocabulary=vocab)
        results = check_narrative_text(annotation.narrative, context)
        adverse = [r for r in results if r.verdict is Verdict.CONTRADICTED]
        critical = [r for r in adverse if r.is_critical]

        hedging_hits.append(not critical)
        for result in critical:
            hedging_detail.append(f"{annotation.case_id}: {result.reason}")
        accuracy_hits.append(not adverse)
        for result in adverse:
            if result not in critical:
                accuracy_detail.append(f"{annotation.case_id}: {result.reason}")

    dimensions = (
        DimensionScore(
            "typology_agreement",
            _mean(typology_hits),
            limits["typology_agreement"],
            tuple(typology_detail),
        ),
        DimensionScore(
            "salience_coverage",
            _mean(salience_scores) if mentioned_by_case is not None else 0.0,
            limits["salience_coverage"],
            () if mentioned_by_case is not None else ("not measured: no alignment supplied",),
        ),
        DimensionScore(
            "hedging_compliance",
            _mean(hedging_hits),
            limits["hedging_compliance"],
            tuple(hedging_detail),
        ),
        DimensionScore(
            "factual_accuracy",
            _mean(accuracy_hits),
            limits["factual_accuracy"],
            tuple(accuracy_detail),
        ),
    )

    return AnnotatorCalibration(
        annotator_id=annotator_id,
        n_items=len(mine),
        n_expected=len(calibration),
        dimensions=dimensions,
        mean_minutes=sum(a.seconds_spent for a in mine) / len(mine) / 60,
        guidance=_guidance(dimensions, confusions),
    )


def _mean(values: list[bool] | list[float]) -> float:
    """Return the mean of a list, 0.0 when it is empty.

    Args:
        values: The values.

    Returns:
        The mean. Zero rather than an error on an empty list: an unmeasured dimension
        must score zero and fail, never pass by default.
    """
    return sum(float(v) for v in values) / len(values) if values else 0.0


def _guidance(dimensions: tuple[DimensionScore, ...], confusions: Counter[str]) -> tuple[str, ...]:
    """Turn failed dimensions into things the annotator can act on.

    Args:
        dimensions: The four scores.
        confusions: ``"assigned for reference"`` to how often, so a *systematic* confusion
            can be named as one rather than listed item by item.

    Returns:
        One line per problem, most systematic first.
    """
    guidance: list[str] = []
    for pair, n in confusions.most_common():
        if n >= _SYSTEMATIC_CONFUSION_MIN:
            assigned, _, reference = pair.partition(" for ")
            guidance.append(
                f"You assigned {assigned!r} where the reference says {reference!r} on "
                f"{n} items. This is a systematic difference, not a slip — re-read Part C "
                f"of the guidelines on {reference}."
            )
    for dimension in dimensions:
        if dimension.passed:
            continue
        if dimension.name == "hedging_compliance":
            guidance.append(
                "A SAR reports suspicion, never guilt, and this threshold is 1.00 because "
                "one overclaim in ten items is one too many. Re-read Part A and the "
                "allowed hedge list, and write the hedge before the finding."
            )
        elif dimension.name == "salience_coverage":
            guidance.append(
                "Your narratives omit facts the typology's salience list requires. The "
                "fact panel marks them; check the marked rows before you submit."
            )
        elif dimension.name == "factual_accuracy":
            guidance.append(
                "Some claims disagree with the fact record. Assert only what the panel "
                "shows, and use the record's own numbers rather than rounding them."
            )
        elif dimension.name == "typology_agreement" and not guidance:
            guidance.append(
                "Your typology judgements differ from the reference without a single "
                "systematic pattern. Re-read Part C and the worked examples."
            )
    return tuple(guidance)
