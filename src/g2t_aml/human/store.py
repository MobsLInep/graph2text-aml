"""Where annotations are written, and what is recorded besides the text.

The narrative is the deliverable. Everything else in an :class:`Annotation` exists because
some question about the Gold set cannot be answered afterwards without it.

- **Time spent** answers "is fifteen minutes per item real?", which is the number the whole
  recruitment plan is costed against, and it answers "did this annotator's quality fall as
  they sped up?", which is the question a reviewer will ask about a set annotated over six
  weeks.
- **Revision count and the draft's own history** distinguish a considered narrative from a
  first draft submitted unread. It is also the only signal that an item was *hard*, which is
  worth more than the annotator's own difficulty rating because it is not self-reported.
- **Every validation flag, and whether it was overridden**, makes the flag list itself
  measurable. See :mod:`g2t_aml.human.validation` for why overriding is permitted.
- **The panel and graph-view digests** record what the annotator could actually see. A
  narrative that omits a fact hidden by the display cap is not an annotator error, and six
  weeks later there is no other way to tell.
- **What they were shown of the case, never what a system wrote about it.** There is no
  field here for a model narrative, a Bronze rendering or a Silver rewrite, and
  :func:`append_annotation` refuses a record carrying one. Gold's independence is the
  reason it is worth having, and independence enforced by intention is independence until
  the first busy afternoon.

Storage is append-only JSONL through :mod:`g2t_aml.utils.io`'s atomic writer, one file per
annotator. Append-only because an annotation that was revised and an annotation that was
replaced are different histories, and a store that overwrote would keep neither.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from g2t_aml.human.validation import LiveFlag
from g2t_aml.utils.io import atomic_path

__all__ = [
    "ANNOTATION_SCHEMA_VERSION",
    "FORBIDDEN_KEYS",
    "Annotation",
    "AnnotationStore",
    "AnnotationStoreError",
    "FlagOutcome",
]

#: Version of the annotation record. Independent of ``training_record``: this is the raw
#: capture, and ingestion turns it into a training record afterwards.
ANNOTATION_SCHEMA_VERSION = "1.0.0"

#: Keys that must never appear on an annotation. Each names something generated — a model
#: narrative, a template rendering, a rewrite. Their presence would mean the annotator saw
#: system output, which is the one thing Gold cannot survive.
FORBIDDEN_KEYS: frozenset[str] = frozenset(
    {
        "bronze_narrative",
        "silver_narrative",
        "model_narrative",
        "generated_narrative",
        "suggestion",
        "prefill",
        "reference_narrative",
    }
)

#: An annotator identifier. Deliberately restrictive: these are pseudonyms
#: (``annotator-03``), never names or emails, because invariant 8 forbids real-world
#: identifiers in the repository and an annotation file is a repository artifact.
_ANNOTATOR_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,31}$")


class AnnotationStoreError(RuntimeError):
    """Raised when an annotation cannot be accepted or a store cannot be read."""


@dataclass(frozen=True)
class FlagOutcome:
    """One validation flag and what the annotator did about it.

    Attributes:
        flag: The flag as raised.
        overridden: True when the annotator submitted with it still standing. False when
            they changed the text and it stopped firing.
        annotator_note: Their reason, when they gave one. Optional, and worth reading:
            this is where a wrong rule gets explained.
    """

    flag: LiveFlag
    overridden: bool
    annotator_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised outcome.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            **self.flag.to_dict(),
            "overridden": self.overridden,
            "annotator_note": self.annotator_note,
        }


@dataclass(frozen=True)
class Annotation:
    """One submitted item.

    Attributes:
        case_id: The case.
        dataset: Substrate key.
        annotator_id: Pseudonymous annotator identifier.
        narrative: The narrative, exactly as submitted.
        seconds_spent: Wall-clock seconds between opening the item and submitting it.
        revision_count: How many times the draft was substantively changed.
        flags: Every validation flag raised at submission, with its outcome.
        typology_assigned: The typology the annotator judged the case to be, from the
            controlled vocabulary. Recorded separately from the narrative because it is
            what Cohen's kappa is computed over.
        difficulty: The annotator's own 1-5 rating, or None.
        annotator_comment: Free text, for anything the structure does not capture.
        submitted_at: ISO-8601 UTC.
        panel_digest: What the fact panel showed, from
            :meth:`~g2t_aml.human.factpanel.FactPanel.to_dict`.
        graph_digest: What the graph view showed, from
            :meth:`~g2t_aml.human.graphview.GraphView.to_dict`.
        checker_summary: The Phase 3 checker's verdict at submission, when the interface
            ran it. Advisory here; ingestion recomputes it and does not trust this.
        is_calibration: Whether this item is part of the calibration set rather than the
            Gold corpus.
        schema_version: :data:`ANNOTATION_SCHEMA_VERSION`.
    """

    case_id: str
    dataset: str
    annotator_id: str
    narrative: str
    seconds_spent: float
    revision_count: int
    flags: tuple[FlagOutcome, ...] = ()
    typology_assigned: str = "unclassified"
    difficulty: int | None = None
    annotator_comment: str = ""
    submitted_at: str = ""
    panel_digest: dict[str, Any] = field(default_factory=dict)
    graph_digest: dict[str, Any] = field(default_factory=dict)
    checker_summary: dict[str, Any] = field(default_factory=dict)
    is_calibration: bool = False
    schema_version: str = ANNOTATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Normalise and validate the record.

        Raises:
            AnnotationStoreError: If the annotator id is not a pseudonym in the expected
                form, the narrative is empty, or time or revision counts are negative.
        """
        if not _ANNOTATOR_RE.match(self.annotator_id):
            raise AnnotationStoreError(
                f"annotator_id {self.annotator_id!r} is not a valid pseudonym. Use "
                "'annotator-03' style identifiers: invariant 8 forbids real-world "
                "identifiers anywhere in this repository, and that includes who wrote "
                "which narrative."
            )
        if not self.narrative.strip():
            raise AnnotationStoreError(f"the annotation for {self.case_id!r} has no narrative")
        if self.seconds_spent < 0 or self.revision_count < 0:
            raise AnnotationStoreError("time spent and revision count cannot be negative")
        if not self.submitted_at:
            object.__setattr__(self, "submitted_at", datetime.now(UTC).isoformat())

    @property
    def n_overridden(self) -> int:
        """Return how many flags were overridden.

        Returns:
            The count.
        """
        return sum(1 for f in self.flags if f.overridden)

    @property
    def n_critical_overridden(self) -> int:
        """Return how many *critical* flags were overridden.

        The number a reviewer reads first. An overridden H4/H6/H7 is either an annotator
        who needs re-calibrating or a vocabulary entry that is wrong, and it is never
        nothing.

        Returns:
            The count.
        """
        return sum(1 for f in self.flags if f.overridden and f.flag.is_critical)

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised annotation.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "dataset": self.dataset,
            "annotator_id": self.annotator_id,
            "narrative": self.narrative,
            "typology_assigned": self.typology_assigned,
            "seconds_spent": round(self.seconds_spent, 3),
            "revision_count": self.revision_count,
            "difficulty": self.difficulty,
            "annotator_comment": self.annotator_comment,
            "submitted_at": self.submitted_at,
            "flags": [f.to_dict() for f in self.flags],
            "n_flags": len(self.flags),
            "n_overridden": self.n_overridden,
            "n_critical_overridden": self.n_critical_overridden,
            "panel_digest": self.panel_digest,
            "graph_digest": self.graph_digest,
            "checker_summary": self.checker_summary,
            "is_calibration": self.is_calibration,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Annotation:
        """Rebuild an annotation from its serialised form.

        Args:
            payload: A line from an annotation file.

        Returns:
            The annotation.

        Raises:
            AnnotationStoreError: If a required field is missing or invalid.
        """
        try:
            flags = tuple(
                FlagOutcome(
                    flag=LiveFlag(
                        rule=str(f["rule"]),
                        severity=_severity(str(f["severity"])),
                        message=str(f.get("message", "")),
                        span=(int(f["span"][0]), int(f["span"][1])),
                        hallucination_class=f.get("hallucination_class"),
                        excerpt=str(f.get("excerpt", "")),
                    ),
                    overridden=bool(f.get("overridden", False)),
                    annotator_note=str(f.get("annotator_note", "")),
                )
                for f in payload.get("flags") or ()
            )
            return cls(
                case_id=str(payload["case_id"]),
                dataset=str(payload["dataset"]),
                annotator_id=str(payload["annotator_id"]),
                narrative=str(payload["narrative"]),
                seconds_spent=float(payload["seconds_spent"]),
                revision_count=int(payload["revision_count"]),
                flags=flags,
                typology_assigned=str(payload.get("typology_assigned", "unclassified")),
                difficulty=payload.get("difficulty"),
                annotator_comment=str(payload.get("annotator_comment", "")),
                submitted_at=str(payload.get("submitted_at", "")),
                panel_digest=dict(payload.get("panel_digest") or {}),
                graph_digest=dict(payload.get("graph_digest") or {}),
                checker_summary=dict(payload.get("checker_summary") or {}),
                is_calibration=bool(payload.get("is_calibration", False)),
                schema_version=str(payload.get("schema_version", ANNOTATION_SCHEMA_VERSION)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AnnotationStoreError(f"malformed annotation record: {exc}") from exc


def _severity(value: str) -> Any:
    """Return the severity enum for a serialised value.

    Args:
        value: The stored string.

    Returns:
        The :class:`~g2t_aml.human.validation.Severity` member.

    Raises:
        AnnotationStoreError: If the value is not a known severity.
    """
    from g2t_aml.human.validation import Severity

    try:
        return Severity(value)
    except ValueError as exc:
        raise AnnotationStoreError(f"unknown flag severity {value!r}") from exc


@dataclass
class AnnotationStore:
    """Append-only per-annotator storage under one directory.

    Attributes:
        root: Directory holding ``<annotator_id>.jsonl``.
    """

    root: Path

    def path_for(self, annotator_id: str) -> Path:
        """Return the file one annotator's work is written to.

        Args:
            annotator_id: The pseudonym.

        Returns:
            The path.

        Raises:
            AnnotationStoreError: If the identifier is not a valid pseudonym, which would
                otherwise let a name reach a filename.
        """
        if not _ANNOTATOR_RE.match(annotator_id):
            raise AnnotationStoreError(f"annotator_id {annotator_id!r} is not a valid pseudonym")
        return Path(self.root) / f"{annotator_id}.jsonl"

    def append(self, annotation: Annotation, **extra: Any) -> Path:
        """Append one annotation.

        Rewrites the whole file through the atomic writer rather than opening it in
        append mode. At a few hundred items that costs nothing, and it means a process
        killed mid-write leaves the previous complete file rather than a truncated last
        line that :meth:`read` would then refuse — losing the whole annotator's history to
        protect one record.

        Args:
            annotation: The submitted item.
            **extra: Additional keys to record alongside it. Rejected if any names
                generated text — see :data:`FORBIDDEN_KEYS`.

        Returns:
            The file written to.

        Raises:
            AnnotationStoreError: If ``extra`` carries a forbidden key.
            OSError: If the write fails.
        """
        if offending := FORBIDDEN_KEYS & set(extra):
            raise AnnotationStoreError(
                f"refusing to store {sorted(offending)} on an annotation. Gold is the "
                "independent human reference; a record carrying generated text is "
                "evidence the annotator was shown some, and the tier is worthless if "
                "they were."
            )
        payload = annotation.to_dict() | dict(extra)
        path = self.path_for(annotation.annotator_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        existing = path.read_text(encoding="utf-8") if path.is_file() else ""
        with atomic_path(path) as tmp:
            tmp.write_text(existing + line + "\n", encoding="utf-8")
        return path

    def read(self, annotator_id: str) -> list[Annotation]:
        """Read one annotator's submissions in order.

        Args:
            annotator_id: The pseudonym.

        Returns:
            Their annotations, oldest first. Empty when they have submitted nothing.

        Raises:
            AnnotationStoreError: If a line is malformed.
        """
        path = self.path_for(annotator_id)
        if not path.is_file():
            return []
        annotations: list[Annotation] = []
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                annotations.append(Annotation.from_dict(json.loads(line)))
            except (json.JSONDecodeError, AnnotationStoreError) as exc:
                raise AnnotationStoreError(f"{path}:{n}: {exc}") from exc
        return annotations

    def annotators(self) -> tuple[str, ...]:
        """Return every annotator with a file in the store.

        Returns:
            Their pseudonyms, sorted.
        """
        root = Path(self.root)
        if not root.is_dir():
            return ()
        return tuple(sorted(p.stem for p in root.glob("*.jsonl")))

    def read_all(self, *, include_calibration: bool = False) -> list[Annotation]:
        """Read every annotation in the store.

        Only the **latest** submission per (annotator, case) is returned: an annotator who
        revisits an item appends a new line, and the earlier one is history rather than a
        second opinion. Treating it as a second opinion would inflate every agreement
        statistic by pairing an annotator with themselves.

        Args:
            include_calibration: Whether to include calibration items. False by default,
                because calibration is scored against references and is not part of the
                corpus.

        Returns:
            The annotations, sorted by case then annotator.

        Raises:
            AnnotationStoreError: If any file is malformed.
        """
        latest: dict[tuple[str, str], Annotation] = {}
        for annotator_id in self.annotators():
            for annotation in self.read(annotator_id):
                if annotation.is_calibration and not include_calibration:
                    continue
                latest[annotation.case_id, annotation.annotator_id] = annotation
        return [latest[k] for k in sorted(latest)]
