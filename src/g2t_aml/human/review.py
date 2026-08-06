"""The second-reviewer pass, and how a disagreement gets settled on the record.

Every Gold item is checked against its fact record by someone other than its author, and
nothing enters ``gold.jsonl`` without an accepted review. That is a stronger requirement
than "two annotators agree", and deliberately so: two people can agree about a case and
both be wrong about what the record says, and on a set this small a shared misreading
would become the reference standard the whole evaluation is measured against.

**A disagreement is data, not friction.** Each one is logged with what the reviewer
disputes and the evidence they cite, and the adjudication records who decided, what they
decided and why. Those adjudications are the phase's most useful qualitative output: they
are a written record of where reasonable, trained readers of the same fact record reached
different conclusions, which is precisely the difficulty the automated metric cannot see.

**Double-annotated items need an extra decision**, and it is made here rather than by a
rule. Two independent narratives of one case are both legitimate; exactly one goes into
the corpus, and which one is an adjudication with a reason, not a coin toss and not
"whichever was submitted first". The other is kept in the annotation store, which is where
the agreement analysis reads it from — discarding it would destroy the double-annotation
measurement to save a line in a file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from g2t_aml.utils.io import atomic_path

__all__ = [
    "REVIEW_SCHEMA_VERSION",
    "Adjudication",
    "Review",
    "ReviewError",
    "ReviewLog",
    "ReviewVerdict",
]

#: Version of the review record.
REVIEW_SCHEMA_VERSION = "1.0.0"


class ReviewError(RuntimeError):
    """Raised when a review is malformed or cannot be reconciled with the annotations."""


class ReviewVerdict(str, Enum):
    """What the second reviewer concluded.

    ``REVISE`` and ``REJECT`` are separated because they mean different things to the
    schedule: a revision returns to its author and comes back, a rejection removes the
    item from Gold and the reservation loses a case. Collapsing them would hide how much
    of the sample survived.
    """

    ACCEPT = "accept"
    REVISE = "revise"
    REJECT = "reject"


@dataclass(frozen=True)
class Adjudication:
    """How a disagreement was settled.

    Attributes:
        decided_by: The adjudicator's pseudonym. Never the reviewer's and never the
            author's; an adjudicator who wrote or reviewed the item is deciding their own
            dispute, and :meth:`Review.validate_against` refuses it.
        decision: What was decided, in plain words.
        rationale: Why. The field the phase's qualitative findings come out of, so a blank
            one is refused.
        decided_at: ISO-8601 UTC.
    """

    decided_by: str
    decision: str
    rationale: str
    decided_at: str = ""

    def __post_init__(self) -> None:
        """Stamp the time and refuse an unexplained adjudication.

        Raises:
            ReviewError: If the decision or the rationale is empty. An adjudication
                without a reason settles the item and teaches nothing, which is half of
                what the process is for.
        """
        if not self.decision.strip() or not self.rationale.strip():
            raise ReviewError(
                "an adjudication needs both a decision and a rationale; the rationale is "
                "where this phase's qualitative findings come from"
            )
        if not self.decided_at:
            object.__setattr__(self, "decided_at", datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised adjudication.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "decided_by": self.decided_by,
            "decision": self.decision,
            "rationale": self.rationale,
            "decided_at": self.decided_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Adjudication:
        """Rebuild an adjudication from its serialised form.

        Args:
            payload: The mapping.

        Returns:
            The adjudication.
        """
        return cls(
            decided_by=str(payload["decided_by"]),
            decision=str(payload["decision"]),
            rationale=str(payload["rationale"]),
            decided_at=str(payload.get("decided_at", "")),
        )


@dataclass(frozen=True)
class Review:
    """One second-reviewer pass over one Gold item.

    Attributes:
        case_id: The case.
        reviewer_id: The reviewer's pseudonym.
        verdict: Their conclusion.
        chosen_annotator: For a double-annotated case, whose narrative goes into the
            corpus. Required when the case has two annotations, and it is an adjudicated
            choice rather than a rule.
        disagreements: What the reviewer disputes, one line each, each citing the fact
            the narrative is wrong about.
        adjudication: How the dispute was settled. Required whenever there are
            disagreements or the verdict is not ``ACCEPT``.
        reviewed_at: ISO-8601 UTC.
        schema_version: :data:`REVIEW_SCHEMA_VERSION`.
    """

    case_id: str
    reviewer_id: str
    verdict: ReviewVerdict
    chosen_annotator: str = ""
    disagreements: tuple[str, ...] = ()
    adjudication: Adjudication | None = None
    reviewed_at: str = ""
    schema_version: str = REVIEW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Stamp the time and enforce the adjudication requirement.

        Raises:
            ReviewError: If a disputed or non-accepted review carries no adjudication.
                This is the rule that makes "disagreements are logged and adjudicated with
                the adjudication recorded" a property of the data rather than a promise.
        """
        if (self.disagreements or self.verdict is not ReviewVerdict.ACCEPT) and (
            self.adjudication is None
        ):
            raise ReviewError(
                f"review of {self.case_id!r} records {len(self.disagreements)} "
                f"disagreement(s) and a {self.verdict.value!r} verdict but no "
                "adjudication. Every disagreement is settled on the record, with a reason."
            )
        if not self.reviewed_at:
            object.__setattr__(self, "reviewed_at", datetime.now(UTC).isoformat())

    @property
    def accepted(self) -> bool:
        """Report whether this review admits the item into the corpus.

        Returns:
            True only for an ``ACCEPT`` verdict. A revised item comes back as a new
            annotation and a new review; a review is never upgraded in place.
        """
        return self.verdict is ReviewVerdict.ACCEPT

    def validate_against(self, annotator_ids: tuple[str, ...]) -> None:
        """Check the review is consistent with the annotations it covers.

        Args:
            annotator_ids: Everyone who annotated this case.

        Raises:
            ReviewError: If the reviewer also annotated the case, if a double-annotated
                case names no chosen annotator, if the chosen annotator did not annotate
                it, or if an adjudicator wrote or reviewed the item. Each of these is a
                way for the second pass to stop being independent, and independence is the
                only thing it contributes.
        """
        if self.reviewer_id in annotator_ids:
            raise ReviewError(
                f"{self.reviewer_id!r} both annotated and reviewed {self.case_id!r}; a "
                "second reviewer who wrote the item is not a second reader of it"
            )
        if len(annotator_ids) > 1:
            if not self.chosen_annotator:
                raise ReviewError(
                    f"{self.case_id!r} has {len(annotator_ids)} annotations but the "
                    "review names no chosen_annotator. Which narrative enters the corpus "
                    "is an adjudicated decision, not a default."
                )
            if self.chosen_annotator not in annotator_ids:
                raise ReviewError(
                    f"review of {self.case_id!r} chooses {self.chosen_annotator!r}, who "
                    f"did not annotate it (annotators: {list(annotator_ids)})"
                )
        if self.adjudication is not None:
            decider = self.adjudication.decided_by
            if decider in annotator_ids or decider == self.reviewer_id:
                raise ReviewError(
                    f"{decider!r} adjudicated a dispute on {self.case_id!r} that they "
                    "were party to; an adjudicator must be neither the author nor the "
                    "reviewer"
                )

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised review.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "reviewer_id": self.reviewer_id,
            "verdict": self.verdict.value,
            "chosen_annotator": self.chosen_annotator,
            "disagreements": list(self.disagreements),
            "adjudication": self.adjudication.to_dict() if self.adjudication else None,
            "reviewed_at": self.reviewed_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Review:
        """Rebuild a review from its serialised form.

        Args:
            payload: A line from the review log.

        Returns:
            The review.

        Raises:
            ReviewError: If a required field is missing or the verdict is unknown.
        """
        try:
            adjudication = payload.get("adjudication")
            return cls(
                case_id=str(payload["case_id"]),
                reviewer_id=str(payload["reviewer_id"]),
                verdict=ReviewVerdict(str(payload["verdict"])),
                chosen_annotator=str(payload.get("chosen_annotator", "")),
                disagreements=tuple(payload.get("disagreements") or ()),
                adjudication=Adjudication.from_dict(adjudication) if adjudication else None,
                reviewed_at=str(payload.get("reviewed_at", "")),
                schema_version=str(payload.get("schema_version", REVIEW_SCHEMA_VERSION)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ReviewError(f"malformed review record: {exc}") from exc


@dataclass
class ReviewLog:
    """Append-only storage for second-reviewer passes.

    Attributes:
        path: The JSONL file reviews are written to.
    """

    path: Path
    _cache: list[Review] | None = field(default=None, repr=False)

    def append(self, review: Review) -> Path:
        """Append one review.

        Args:
            review: The review.

        Returns:
            The file written to.

        Raises:
            OSError: If the write fails.
        """
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = path.read_text(encoding="utf-8") if path.is_file() else ""
        line = json.dumps(review.to_dict(), ensure_ascii=False, sort_keys=True)
        with atomic_path(path) as tmp:
            tmp.write_text(existing + line + "\n", encoding="utf-8")
        self._cache = None
        return path

    def read(self) -> list[Review]:
        """Read every review in the log, in order.

        Returns:
            The reviews. Empty when nothing has been reviewed.

        Raises:
            ReviewError: If a line is malformed.
        """
        if self._cache is not None:
            return list(self._cache)
        path = Path(self.path)
        if not path.is_file():
            return []
        reviews: list[Review] = []
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                reviews.append(Review.from_dict(json.loads(line)))
            except (json.JSONDecodeError, ReviewError) as exc:
                raise ReviewError(f"{path}:{n}: {exc}") from exc
        self._cache = reviews
        return list(reviews)

    def latest_by_case(self) -> dict[str, Review]:
        """Return the most recent review per case.

        A revised item is reviewed again, and the later review is the operative one. The
        earlier one stays in the log, because "this needed two passes" is worth knowing.

        Returns:
            Case id to its latest review.
        """
        latest: dict[str, Review] = {}
        for review in self.read():
            latest[review.case_id] = review
        return latest
