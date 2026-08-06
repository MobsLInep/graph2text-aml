r"""The rating interface: one graph, one fact record, one narrative, and no idea whose.

Run it with::

    uv run streamlit run src/g2t_aml/human/study_ui.py -- \\
        --rater rater-01 --design artifacts/human_study/design.json \\
        --narratives artifacts/human_study/narratives.jsonl

Phase 6's interface asked an annotator to write. This one asks a rater to judge, and the
two differ in what has to be prevented. There, the risk was influence — an annotator shown
a draft writes edits to it. Here the risk is **unblinding**, and it is worse, because it
does not need to be deliberate to work. A rater who notices that the fluent narratives all
arrive early in a session, or that one arm's items share a formatting tic, is scoring the
arm from then on. So:

- :func:`render_item` is handed a :class:`~g2t_aml.human.study_design.StudyItem`, which has
  no system field, and a narrative looked up by item id. There is no code path in this
  module that reads a :class:`~g2t_aml.human.study_design.BlindKey`, and
  ``tests/unit/test_study_blinding.py`` asserts that no rendered payload contains a system
  identifier for any system in the registry.
- The narrative pool is keyed by opaque item id and iterated in the rater's own order, so
  file order carries nothing.
- The response store writes the item id and never a system, which means unblinding happens
  exactly once, in the analysis, against a key held elsewhere.

**The two measurements that matter are the two that a rater cannot self-report.** A Likert
row is what every paper submits and it is the weakest evidence here. Time-to-usable-draft
and edit distance are behavioural: the interface measures them whether or not the rater is
thinking about them, and a reduction in investigator drafting time against the template
baseline is a deployment claim rather than an opinion. Both are therefore instrumented
rather than asked.

**Time is active time.** The timer starts when the narrative first renders, pauses when the
tab loses visibility and resumes when it returns, and stops at submission. A rater who
answers the door mid-item would otherwise contribute a forty-minute reading time, and a
handful of those move a mean far more than a real effect does. Two clocks run: a browser-side
one in :mod:`~g2t_aml.human.study_timer_component` that can see ``visibilitychange``, and a
server-side :class:`BlurAwareTimer` that cannot but always works. Which one produced the
recorded number is stored on the response as :attr:`RatingResponse.timing_source`, because a
session where the component silently failed to load is a session whose times include the
coffee break, and that has to be discoverable afterwards rather than assumed away.

**The edit box starts pre-filled with the presented narrative and both versions are kept.**
The distance between them is computed in :mod:`g2t_aml.human.study_analysis`, not here —
the interface's job is capture, and a metric computed in a UI is a metric that cannot be
recomputed from the released data.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from g2t_aml.human.caseloader import AnnotationCase, CaseSource
from g2t_aml.human.factpanel import FactPanel, build_fact_panel
from g2t_aml.human.graphview import DEFAULT_MAX_NODES, GraphView, build_graph_view
from g2t_aml.human.study_design import StudyDesign, StudyItem, load_design
from g2t_aml.utils.io import atomic_path

__all__ = [
    "LIKERT_DIMENSIONS",
    "RESPONSE_SCHEMA_VERSION",
    "BlurAwareTimer",
    "LikertDimension",
    "RatingResponse",
    "RenderedItem",
    "ResponseStore",
    "ResponseStoreError",
    "assert_no_system_identity",
    "build_parser",
    "load_narratives",
    "main",
    "render_item",
]

#: Version of the response record. Bumping it invalidates the join between a response file
#: and the design that produced it, so the analysis refuses an unexpected version rather
#: than reading fields that may have moved.
RESPONSE_SCHEMA_VERSION = "1.0.0"

#: The ordinal scale's endpoints. Named because they are referenced in five places -- the
#: validator, the widget, the training pack, the ordinal Krippendorff computation and the
#: release schema -- and a scale that changed in four of them is a silent recoding.
LIKERT_MIN = 1
LIKERT_MAX = 7

#: A rater identifier, under the same rule as Phase 6's annotators: pseudonyms only.
#: Invariant 8 forbids real-world identifiers anywhere in the repository, and a response
#: file is a repository artifact that is additionally destined for public release.
_RATER_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,31}$")


@dataclass(frozen=True)
class LikertDimension:
    """One 7-point rating scale, with the anchors that make it mean something.

    The anchors are not decoration. "Factual correctness: 7" is meaningless across raters
    unless 7 has a stated definition, and an inter-rater agreement statistic computed over
    unanchored scales measures how similarly two people use a number line. Every anchor
    here is reproduced verbatim in ``docs/human_study/rater_training.md`` and in the
    interface's own help text, so a rater never has to remember which document was
    authoritative.

    Attributes:
        key: Field name on :class:`RatingResponse`.
        label: What the rater sees.
        question: The question actually being asked, in one sentence.
        anchor_low: What 1 means.
        anchor_mid: What 4 means.
        anchor_high: What 7 means.
    """

    key: str
    label: str
    question: str
    anchor_low: str
    anchor_mid: str
    anchor_high: str

    def to_dict(self) -> dict[str, str]:
        """Return the serialised dimension.

        Returns:
            A JSON-serialisable mapping. Released alongside the data so the scales are
            published with the responses rather than described in prose in a paper.
        """
        return {
            "key": self.key,
            "label": self.label,
            "question": self.question,
            "anchor_1": self.anchor_low,
            "anchor_4": self.anchor_mid,
            "anchor_7": self.anchor_high,
        }


#: The five ordinal dimensions, in presentation order. Factual correctness leads because it
#: is the one the automatic Layer-2 metric claims to predict, and the correlation between
#: the two is what validates that metric.
LIKERT_DIMENSIONS: tuple[LikertDimension, ...] = (
    LikertDimension(
        key="factual_correctness",
        label="Factual correctness",
        question="Is every factual assertion in this narrative supported by the record shown?",
        anchor_low=(
            "1 - At least one assertion is contradicted by the record: a wrong amount, a "
            "wrong count, a wrong direction of flow, or a counterparty that does not appear."
        ),
        anchor_mid=(
            "4 - No assertion is contradicted, but at least one cannot be checked against "
            "the record and would have to be verified before filing."
        ),
        anchor_high=(
            "7 - Every assertion is supported by the record, with no rounding, direction or "
            "attribution errors."
        ),
    ),
    LikertDimension(
        key="completeness",
        label="Completeness",
        question="Are the material facts present?",
        anchor_low=(
            "1 - A fact a reviewer would need to reach a decision is missing: the pattern, "
            "the volume, the counterparties, or the exculpatory context."
        ),
        anchor_mid=(
            "4 - The main pattern is stated but a supporting detail a reviewer would ask "
            "for is absent."
        ),
        anchor_high=(
            "7 - Every material fact is present, including the exculpatory ones. A reviewer "
            "would ask no follow-up question answerable from the record."
        ),
    ),
    LikertDimension(
        key="actionability",
        label="Actionability",
        question="Could an investigator act on this?",
        anchor_low=(
            "1 - Says something happened without saying what, to whom, or in what pattern. "
            "No next step follows from it."
        ),
        anchor_mid=(
            "4 - The activity is identifiable but the investigator would have to return to "
            "the raw data to decide anything."
        ),
        anchor_high=(
            "7 - States the pattern, its participants and its scale precisely enough that "
            "the next investigative step is obvious from the text alone."
        ),
    ),
    LikertDimension(
        key="readability",
        label="Readability and professional register",
        question="Does this read as professional financial-crime writing?",
        anchor_low=(
            "1 - Ungrammatical, repetitive, or written in a register no compliance function "
            "would send: marketing language, chat register, or machine-listing style."
        ),
        anchor_mid=(
            "4 - Clear and correct but flat or formulaic; a reviewer would rewrite it before "
            "sending."
        ),
        anchor_high=("7 - Reads as competent professional prose. Could be sent after a proofread."),
    ),
    LikertDimension(
        key="regulatory_tone",
        label="Regulatory tone appropriateness",
        question="Does it report suspicion rather than assert guilt?",
        anchor_low=(
            "1 - Asserts criminality as fact: calls the activity money laundering, names the "
            "subject as a launderer, or states intent."
        ),
        anchor_mid=(
            "4 - Mostly hedged, with at least one sentence that overstates what the data "
            "supports."
        ),
        anchor_high=(
            "7 - Describes observed activity and why it is consistent with a typology, "
            "without asserting intent, guilt or a legal conclusion anywhere."
        ),
    ),
)


class ResponseStoreError(RuntimeError):
    """Raised when a response cannot be accepted or a response file cannot be read."""


@dataclass
class BlurAwareTimer:
    """Wall-clock time with the periods the tab was not visible removed.

    The server-side clock, and the authority in tests. It is driven by explicit events so
    that timer behaviour is testable without a browser: a test feeds it a sequence of
    timestamps and asserts the arithmetic, which is the only way to know the pause logic is
    right before a rater's session depends on it.

    All timestamps are monotonic seconds supplied by the caller rather than read from a
    clock inside, for the same reason.

    Attributes:
        started_at: When the narrative first rendered, or None before it did.
        stopped_at: When submission happened, or None before it did.
        hidden_since: When the tab last became hidden, or None while visible.
        hidden_seconds: Total time accumulated while hidden.
        n_blurs: How many times the tab lost visibility. Reported because an item rated
            across six interruptions is a different measurement from one rated in a sitting,
            even when the active times match.
    """

    started_at: float | None = None
    stopped_at: float | None = None
    hidden_since: float | None = None
    hidden_seconds: float = 0.0
    n_blurs: int = 0

    def start(self, now: float) -> None:
        """Start the clock when the narrative renders.

        Idempotent: a Streamlit rerun re-executes the whole script, and a start that reset
        the clock on every rerun would report the time since the last widget interaction
        rather than the time since the item opened.

        Args:
            now: Monotonic seconds.
        """
        if self.started_at is None:
            self.started_at = now

    def blur(self, now: float) -> None:
        """Record the tab becoming hidden.

        Args:
            now: Monotonic seconds.
        """
        if self.started_at is None or self.stopped_at is not None:
            return
        if self.hidden_since is None:
            self.hidden_since = now
            self.n_blurs += 1

    def focus(self, now: float) -> None:
        """Record the tab becoming visible again.

        Args:
            now: Monotonic seconds.
        """
        if self.hidden_since is not None:
            self.hidden_seconds += max(0.0, now - self.hidden_since)
            self.hidden_since = None

    def stop(self, now: float) -> None:
        """Stop the clock at submission.

        Closes an open hidden period first, so an item submitted from a keyboard shortcut
        while the tab was reported hidden does not count that period as active.

        Args:
            now: Monotonic seconds.
        """
        if self.started_at is None or self.stopped_at is not None:
            return
        self.focus(now)
        self.stopped_at = now

    def active_seconds(self, now: float | None = None) -> float:
        """Return elapsed active time, excluding hidden periods.

        Args:
            now: Monotonic seconds, for reading the clock while it is still running.
                Ignored once stopped. Required while running.

        Returns:
            Active seconds, never negative. Zero before the clock starts.

        Raises:
            ValueError: If the clock is running and ``now`` was not supplied, which would
                otherwise silently return the time at the last event.
        """
        if self.started_at is None:
            return 0.0
        if self.stopped_at is not None:
            end = self.stopped_at
        elif now is None:
            raise ValueError("active_seconds() needs `now` while the timer is running")
        else:
            end = now
        hidden = self.hidden_seconds
        if self.hidden_since is not None:
            hidden += max(0.0, end - self.hidden_since)
        return max(0.0, (end - self.started_at) - hidden)

    def to_dict(self, now: float | None = None) -> dict[str, Any]:
        """Return the serialised timing record.

        Args:
            now: Monotonic seconds, if the clock is still running.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "active_seconds": round(self.active_seconds(now), 3),
            "hidden_seconds": round(self.hidden_seconds, 3),
            "n_blurs": self.n_blurs,
            "stopped": self.stopped_at is not None,
        }


@dataclass(frozen=True)
class RenderedItem:
    """Everything shown to the rater for one item, and nothing else.

    The blinding boundary is this dataclass. :func:`render_item` builds it from a
    :class:`~g2t_aml.human.study_design.StudyItem` and a narrative pool, neither of which
    carries a system identity, and the Streamlit layer displays only what is here. A test
    walks every field of a rendered item for every registry system id and asserts none
    appears — which is a check on the *data*, not on the display code, and so cannot be
    defeated by a later change to the layout.

    Attributes:
        item_id: The opaque identifier.
        case_id: The case.
        dataset: Substrate key.
        narrative: The narrative under judgement, exactly as the system produced it.
        panel: The fact record, rendered through Bronze's formatters (D-054).
        graph: The graph view.
        position: 0-based position in this rater's sequence, for the progress indicator.
        total: How many items this rater has in all.
    """

    item_id: str
    case_id: str
    dataset: str
    narrative: str
    panel: FactPanel
    graph: GraphView
    position: int
    total: int

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised rendered item.

        Used by the blinding test, which needs the whole payload as data rather than as
        rendered pixels.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "item_id": self.item_id,
            "case_id": self.case_id,
            "dataset": self.dataset,
            "narrative": self.narrative,
            "panel": self.panel.to_dict(),
            "graph": self.graph.to_dict(),
            "position": self.position,
            "total": self.total,
        }


@dataclass(frozen=True)
class RatingResponse:
    """One rater's judgement of one item.

    Attributes:
        item_id: The opaque item identifier. The only handle on which system this was, and
            it resolves only against the blind key.
        rater_id: Pseudonymous rater identifier.
        case_id: The case.
        position: Where this sat in the rater's sequence.
        is_repeat: Whether this is a planted repeat, for intra-rater reliability.
        factual_correctness: 1-7.
        completeness: 1-7.
        actionability: 1-7.
        readability: 1-7.
        regulatory_tone: 1-7.
        would_file: Would you file this after review? The binary, and the closest thing
            here to a decision rather than an opinion.
        seconds_to_usable_draft: Active seconds from first render to submission. The
            headline behavioural measure.
        presented_narrative: What the rater was shown, stored verbatim. Kept even though it
            is recoverable from the narrative pool, because a released response file that
            cannot be read without a second file is a released response file nobody reads.
        corrected_narrative: What the rater edited it to. The edit distance between the two
            is computed at analysis time.
        timing_source: ``"browser"`` when the visibility-aware component supplied the
            number, ``"server"`` when it did not. A session's times mean different things
            in the two cases and the difference must not be silent.
        hidden_seconds: Time excluded because the tab was not visible.
        n_blurs: How many times the tab lost visibility during this item.
        comment: Optional free text. **Scrubbed before release** — see
            :func:`g2t_aml.human.study_release.prepare_release`.
        submitted_at: ISO-8601 UTC.
        schema_version: :data:`RESPONSE_SCHEMA_VERSION`.
    """

    item_id: str
    rater_id: str
    case_id: str
    position: int
    is_repeat: bool
    factual_correctness: int
    completeness: int
    actionability: int
    readability: int
    regulatory_tone: int
    would_file: bool
    seconds_to_usable_draft: float
    presented_narrative: str
    corrected_narrative: str
    timing_source: str = "server"
    hidden_seconds: float = 0.0
    n_blurs: int = 0
    comment: str = ""
    submitted_at: str = ""
    schema_version: str = RESPONSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Normalise and validate the response.

        Raises:
            ResponseStoreError: If the rater id is not a pseudonym, a Likert value is
                outside 1-7, the time is negative, or the presented narrative is empty.
        """
        if not _RATER_RE.match(self.rater_id):
            raise ResponseStoreError(
                f"rater_id {self.rater_id!r} is not a valid pseudonym. Use 'rater-03' style "
                "identifiers: invariant 8 forbids real-world identifiers in this "
                "repository, and these responses are destined for public release."
            )
        for dimension in LIKERT_DIMENSIONS:
            value = getattr(self, dimension.key)
            if not isinstance(value, int) or not LIKERT_MIN <= value <= LIKERT_MAX:
                raise ResponseStoreError(
                    f"{dimension.key} is {value!r}; the scale is the integers "
                    f"{LIKERT_MIN} to {LIKERT_MAX}"
                )
        if self.seconds_to_usable_draft < 0:
            raise ResponseStoreError("time-to-usable-draft cannot be negative")
        if not self.presented_narrative.strip():
            raise ResponseStoreError(
                f"item {self.item_id!r} has an empty presented narrative; a rating of "
                "nothing is not a rating"
            )
        if self.timing_source not in {"browser", "server"}:
            raise ResponseStoreError(f"unknown timing_source {self.timing_source!r}")
        if not self.submitted_at:
            object.__setattr__(self, "submitted_at", datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised response.

        Returns:
            A JSON-serialisable mapping. Carries no system identity.
        """
        return {
            "schema_version": self.schema_version,
            "item_id": self.item_id,
            "rater_id": self.rater_id,
            "case_id": self.case_id,
            "position": self.position,
            "is_repeat": self.is_repeat,
            **{d.key: getattr(self, d.key) for d in LIKERT_DIMENSIONS},
            "would_file": self.would_file,
            "seconds_to_usable_draft": round(self.seconds_to_usable_draft, 3),
            "timing_source": self.timing_source,
            "hidden_seconds": round(self.hidden_seconds, 3),
            "n_blurs": self.n_blurs,
            "presented_narrative": self.presented_narrative,
            "corrected_narrative": self.corrected_narrative,
            "comment": self.comment,
            "submitted_at": self.submitted_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RatingResponse:
        """Rebuild a response from its serialised form.

        Args:
            payload: One line of a response file.

        Returns:
            The response.

        Raises:
            ResponseStoreError: If a required field is missing or invalid.
        """
        try:
            return cls(
                item_id=str(payload["item_id"]),
                rater_id=str(payload["rater_id"]),
                case_id=str(payload["case_id"]),
                position=int(payload["position"]),
                is_repeat=bool(payload.get("is_repeat", False)),
                factual_correctness=int(payload["factual_correctness"]),
                completeness=int(payload["completeness"]),
                actionability=int(payload["actionability"]),
                readability=int(payload["readability"]),
                regulatory_tone=int(payload["regulatory_tone"]),
                would_file=bool(payload["would_file"]),
                seconds_to_usable_draft=float(payload["seconds_to_usable_draft"]),
                presented_narrative=str(payload["presented_narrative"]),
                corrected_narrative=str(payload.get("corrected_narrative", "")),
                timing_source=str(payload.get("timing_source", "server")),
                hidden_seconds=float(payload.get("hidden_seconds", 0.0)),
                n_blurs=int(payload.get("n_blurs", 0)),
                comment=str(payload.get("comment", "")),
                submitted_at=str(payload.get("submitted_at", "")),
                schema_version=str(payload.get("schema_version", RESPONSE_SCHEMA_VERSION)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ResponseStoreError(f"malformed rating response: {exc}") from exc


@dataclass
class ResponseStore:
    """Append-only per-rater response storage, and the mechanism behind save-and-resume.

    A rater's workload is measured in hours, so a session *will* be interrupted, and the
    resume path is therefore load-bearing rather than a convenience.
    :meth:`completed_item_ids` is how the interface knows where to restart, and it is
    derived from what was actually written rather than from a separate progress file that
    could disagree with it.

    Append-only, and for the same reason as Phase 6's annotation store: a rater who returns
    to an item has changed their mind, and both judgements are history worth keeping. The
    analysis reads the last one per item.

    Attributes:
        root: Directory holding ``<rater_id>.jsonl``.
    """

    root: Path

    def path_for(self, rater_id: str) -> Path:
        """Return the file one rater's responses are written to.

        Args:
            rater_id: The pseudonym.

        Returns:
            The path.

        Raises:
            ResponseStoreError: If the identifier is not a valid pseudonym.
        """
        if not _RATER_RE.match(rater_id):
            raise ResponseStoreError(f"rater_id {rater_id!r} is not a valid pseudonym")
        return Path(self.root) / f"{rater_id}.jsonl"

    def append(self, response: RatingResponse) -> Path:
        """Append one response.

        Rewrites the whole file through the atomic writer, as Phase 6's store does: at a
        few hundred rows the cost is nothing, and a process killed mid-write leaves the
        previous complete file rather than a truncated final line that would make the whole
        rater unreadable.

        Args:
            response: The submitted judgement.

        Returns:
            The file written to.

        Raises:
            OSError: If the write fails.
        """
        path = self.path_for(response.rater_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(response.to_dict(), ensure_ascii=False, sort_keys=True)
        existing = path.read_text(encoding="utf-8") if path.is_file() else ""
        with atomic_path(path) as tmp:
            tmp.write_text(existing + line + "\n", encoding="utf-8")
        return path

    def read(self, rater_id: str) -> list[RatingResponse]:
        """Read one rater's responses in submission order.

        Args:
            rater_id: The pseudonym.

        Returns:
            Their responses, oldest first. Empty when they have submitted nothing.

        Raises:
            ResponseStoreError: If a line is malformed.
        """
        path = self.path_for(rater_id)
        if not path.is_file():
            return []
        out: list[RatingResponse] = []
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                out.append(RatingResponse.from_dict(json.loads(line)))
            except (json.JSONDecodeError, ResponseStoreError) as exc:
                raise ResponseStoreError(f"{path}:{n}: {exc}") from exc
        return out

    def completed_item_ids(self, rater_id: str) -> set[str]:
        """Return the items this rater has already submitted.

        Args:
            rater_id: The pseudonym.

        Returns:
            Their item ids.

        Raises:
            ResponseStoreError: If their file is malformed.
        """
        return {r.item_id for r in self.read(rater_id)}

    def raters(self) -> tuple[str, ...]:
        """Return every rater with a file in the store.

        Returns:
            Their pseudonyms, sorted.
        """
        root = Path(self.root)
        if not root.is_dir():
            return ()
        return tuple(sorted(p.stem for p in root.glob("*.jsonl")))

    def read_all(self) -> list[RatingResponse]:
        """Read every response in the store, keeping the latest per (rater, item).

        Returns:
            The responses, sorted by rater then position.

        Raises:
            ResponseStoreError: If any file is malformed.
        """
        latest: dict[tuple[str, str], RatingResponse] = {}
        for rater_id in self.raters():
            for response in self.read(rater_id):
                latest[response.rater_id, response.item_id] = response
        return sorted(latest.values(), key=lambda r: (r.rater_id, r.position))


def load_narratives(path: Path) -> dict[str, str]:
    """Read the narrative pool: item id to the text that item shows.

    The pool is built by ``scripts/12_build_study.py``, which is the one place that holds
    the design, the blind key and the generated corpora at the same time. It emits this
    file keyed by opaque item id and with no system field, so that the interface can be
    given it directly without the interface ever having had the opportunity to unblind.

    Args:
        path: The narrative pool, one JSON object per line with ``item_id`` and
            ``narrative``.

    Returns:
        Item id to narrative.

    Raises:
        ResponseStoreError: If the file is missing, malformed, carries a system field, or
            contains a duplicate item id.
    """
    path = Path(path)
    if not path.is_file():
        raise ResponseStoreError(f"no narrative pool at {path}")
    out: dict[str, str] = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ResponseStoreError(f"{path}:{n}: {exc}") from exc
        leaked = {"system", "system_id", "arm", "generator", "tier"} & set(payload)
        if leaked:
            raise ResponseStoreError(
                f"{path}:{n}: the narrative pool carries {sorted(leaked)}. This file is "
                "loaded by the rating interface; a system field in it unblinds the study "
                "whether or not the interface displays it."
            )
        item_id = str(payload.get("item_id", ""))
        if not item_id:
            raise ResponseStoreError(f"{path}:{n}: no item_id")
        if item_id in out:
            raise ResponseStoreError(f"{path}:{n}: duplicate item_id {item_id!r}")
        out[item_id] = str(payload.get("narrative", ""))
    return out


def render_item(
    item: StudyItem,
    case: AnnotationCase,
    narratives: dict[str, str],
    total: int,
    *,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> RenderedItem:
    """Assemble everything shown for one item.

    Args:
        item: The design's item. Carries no system identity.
        case: The case, loaded by :class:`~g2t_aml.human.caseloader.CaseSource`.
        narratives: The narrative pool from :func:`load_narratives`.
        total: How many items this rater has, for the progress indicator.
        max_nodes: Display cap passed to the graph view.

    Returns:
        The rendered item.

    Raises:
        ResponseStoreError: If the pool has no narrative for this item, or the item and
            case disagree about which case this is.
    """
    if item.case_id != case.case_id:
        raise ResponseStoreError(
            f"item {item.item_id!r} is for case {item.case_id!r} but was given case "
            f"{case.case_id!r}"
        )
    narrative = narratives.get(item.item_id)
    if not narrative:
        raise ResponseStoreError(
            f"the narrative pool has no text for item {item.item_id!r}. Rendering a blank "
            "item would collect a rating of nothing and record it as a rating of a system."
        )
    return RenderedItem(
        item_id=item.item_id,
        case_id=item.case_id,
        dataset=item.dataset,
        narrative=narrative,
        panel=build_fact_panel(case.facts),
        graph=build_graph_view(case.view, case.focal_id, max_nodes=max_nodes),
        position=item.position,
        total=total,
    )


def assert_no_system_identity(payload: Any, system_ids: list[str]) -> None:
    """Assert that a rendered payload names no system in the matrix.

    The blinding check, written as a function rather than living only in a test so that it
    can also run at session start against the first item: a study whose blinding broke is
    better discovered before the first rating than after the last.

    Matching is on word boundaries over the serialised payload. Substring matching would be
    useless here — ``"B1"`` occurs inside a hex item id roughly always — and the ids being
    searched for are short tokens that only appear as words when something has genuinely
    leaked.

    Args:
        payload: Anything JSON-serialisable. Usually
            :meth:`RenderedItem.to_dict`.
        system_ids: The system identifiers that must not appear.

    Raises:
        AssertionError: If any system id appears as a word in the payload.
    """
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    for system_id in system_ids:
        if re.search(rf"(?<![0-9A-Za-z_]){re.escape(system_id)}(?![0-9A-Za-z_])", blob):
            raise AssertionError(
                f"the rendered payload names system {system_id!r}. The rater can see it, "
                "and every rating collected in this session is unblinded."
            )


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for the Streamlit entrypoint.

    Returns:
        The parser. Arguments follow ``--`` on the ``streamlit run`` command line.
    """
    parser = argparse.ArgumentParser(description="Graph2Text AML Phase 12 rating interface")
    parser.add_argument("--rater", required=True, help="pseudonym, e.g. rater-01")
    parser.add_argument("--design", required=True, type=Path, help="design.json")
    parser.add_argument("--narratives", required=True, type=Path, help="narratives.jsonl")
    # No default. Every directory root lives in `configs/paths/` and is reached as
    # `cfg.paths.*`; a default here would be a hardcoded path in `src/`, which the repo
    # contract greps for. `make study-rate` supplies it.
    parser.add_argument("--responses", type=Path, required=True, help="response store root")
    parser.add_argument("--processed", type=Path, required=True, help="substrate processed dir")
    parser.add_argument("--interim", type=Path, required=True, help="substrate interim dir")
    parser.add_argument("--max-nodes", type=int, default=DEFAULT_MAX_NODES)
    return parser


def _require_streamlit() -> Any:
    """Return the Streamlit module, with a message naming the extra when it is absent.

    Returns:
        The ``streamlit`` module.

    Raises:
        RuntimeError: If Streamlit is not installed.
    """
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "the rating interface needs the `human` extra: uv sync --extra human"
        ) from exc
    return st


@dataclass
class _SessionState:
    """The interface's per-item working state, kept out of Streamlit's session dict.

    Attributes:
        timer: The server-side clock for the item currently on screen.
        item_id: Which item the timer belongs to, so a rerun that advances the item
            resets the clock rather than carrying the previous item's elapsed time.
        browser_active_ms: The browser component's latest reading, when it loaded.
        browser_hidden_ms: The browser component's hidden total.
        browser_blurs: The browser component's blur count.
    """

    timer: BlurAwareTimer = field(default_factory=BlurAwareTimer)
    item_id: str = ""
    browser_active_ms: float | None = None
    browser_hidden_ms: float = 0.0
    browser_blurs: int = 0


def _render_case(st: Any, rendered: RenderedItem) -> None:  # pragma: no cover - Streamlit
    """Render the left column: the graph and the fact record.

    Args:
        st: The Streamlit module.
        rendered: The item.
    """
    from g2t_aml.human.graphview import to_plotly_figure

    st.subheader("The case")
    st.plotly_chart(to_plotly_figure(rendered.graph), use_container_width=True)
    st.caption(f"{rendered.panel.case_id} · {rendered.panel.dataset}")
    if rendered.panel.masked_families:
        st.warning(
            "**Not available on this substrate:**\n\n"
            + "\n".join(f"- {m}" for m in rendered.panel.masked_families)
        )
    for section in rendered.panel.sections:
        with st.expander(section.name, expanded=section.name in ("Subject", "Scope")):
            for row in section.rows:
                st.markdown(f"{row.label}: **{row.value}**")


def _elapsed(state: _SessionState) -> tuple[float, str, float, int]:  # pragma: no cover
    """Return the item's timing, preferring the browser clock when it reported.

    Args:
        state: The session state, holding both clocks.

    Returns:
        ``(seconds, source, hidden_seconds, n_blurs)``.
    """
    if state.browser_active_ms is not None:
        return (
            state.browser_active_ms / 1000.0,
            "browser",
            state.browser_hidden_ms / 1000.0,
            state.browser_blurs,
        )
    return state.timer.active_seconds(), "server", state.timer.hidden_seconds, state.timer.n_blurs


def _render_rating_form(  # pragma: no cover - Streamlit
    st: Any,
    state: _SessionState,
    item: StudyItem,
    rendered: RenderedItem,
    store: ResponseStore,
    rater_id: str,
) -> None:
    """Render the right column: the narrative, the scales, the edit box and Submit.

    The draft is shown above the scales and the edit box below them, deliberately. A rater
    who edits before rating has already formed their judgement by fixing things, and their
    Likert row then describes the text they produced rather than the one they were given.

    Args:
        st: The Streamlit module.
        state: The session state, holding the clocks.
        item: The design's item.
        rendered: What is on screen.
        store: Where the response goes.
        rater_id: The rater's pseudonym.
    """
    import time

    st.subheader("Draft narrative")
    st.info("Read the draft and rate it. Only then edit it.")
    st.markdown(f"> {rendered.narrative}")

    values: dict[str, int] = {}
    for dimension in LIKERT_DIMENSIONS:
        st.markdown(f"**{dimension.label}** — {dimension.question}")
        st.caption(f"{dimension.anchor_low}  \n{dimension.anchor_mid}  \n{dimension.anchor_high}")
        values[dimension.key] = st.slider(
            dimension.label,
            LIKERT_MIN,
            LIKERT_MAX,
            (LIKERT_MIN + LIKERT_MAX) // 2,
            key=f"{item.item_id}-{dimension.key}",
            label_visibility="collapsed",
        )
    would_file = (
        st.radio(
            "Would you file this after review?",
            ["No", "Yes"],
            key=f"{item.item_id}-file",
            horizontal=True,
        )
        == "Yes"
    )

    st.subheader("Correct the draft to a filable state")
    st.caption(
        "Edit the text below until you would be willing to file it. The difference between "
        "what you were shown and what you leave here is the measurement."
    )
    corrected = st.text_area(
        "Corrected narrative",
        value=rendered.narrative,
        height=300,
        key=f"{item.item_id}-edit",
        label_visibility="collapsed",
    )
    comment = st.text_input("Anything else? (optional)", key=f"{item.item_id}-comment")

    if st.button("Submit and continue", type="primary"):
        state.timer.stop(time.monotonic())
        seconds, source_name, hidden, blurs = _elapsed(state)
        store.append(
            RatingResponse(
                item_id=item.item_id,
                rater_id=rater_id,
                case_id=item.case_id,
                position=item.position,
                is_repeat=item.is_repeat,
                would_file=would_file,
                seconds_to_usable_draft=seconds,
                presented_narrative=rendered.narrative,
                corrected_narrative=corrected,
                timing_source=source_name,
                hidden_seconds=hidden,
                n_blurs=blurs,
                comment=comment,
                **values,
            )
        )
        st.rerun()


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - the Streamlit shell
    """Run the rating interface.

    Everything decided here is decided elsewhere and only displayed: the assignment is
    :mod:`g2t_aml.human.study_design`, the fact panel is
    :mod:`g2t_aml.human.factpanel`, the graph is :mod:`g2t_aml.human.graphview`, and what
    is written down is :class:`ResponseStore`. Consequently this function is the only part
    of Phase 12 that is not unit-tested, and it is kept as thin as that split allows.

    Args:
        argv: Command-line arguments. Defaults to those following ``--``.

    Raises:
        RuntimeError: If Streamlit is not installed.
    """
    import time

    st = _require_streamlit()
    args = build_parser().parse_args(argv)

    st.set_page_config(page_title="SAR narrative review", layout="wide")

    design: StudyDesign = load_design(args.design)
    narratives = load_narratives(args.narratives)
    store = ResponseStore(root=args.responses)
    sequence = design.for_rater(args.rater)
    if not sequence:
        st.error(f"No items in this design for rater {args.rater!r}.")
        return

    done = store.completed_item_ids(args.rater)
    remaining = [i for i in sequence if i.item_id not in done]
    st.progress(len(done) / len(sequence), text=f"{len(done)} of {len(sequence)} items rated")
    if not remaining:
        st.success("All items rated. Thank you — you can close this window.")
        return

    item = remaining[0]
    if "study_state" not in st.session_state:
        st.session_state["study_state"] = _SessionState()
    state: _SessionState = st.session_state["study_state"]
    if state.item_id != item.item_id:
        st.session_state["study_state"] = state = _SessionState(item_id=item.item_id)

    source = CaseSource(processed_dir=args.processed, interim_dir=args.interim)
    rendered = render_item(
        item, source.load(item.case_id), narratives, len(sequence), max_nodes=args.max_nodes
    )
    assert_no_system_identity(rendered.to_dict(), list(design.systems))
    state.timer.start(time.monotonic())

    from g2t_aml.human.study_timer_component import visibility_timer

    reading = visibility_timer(key=f"timer-{item.item_id}")
    if reading is not None:
        state.browser_active_ms = float(reading.get("active_ms", 0.0))
        state.browser_hidden_ms = float(reading.get("hidden_ms", 0.0))
        state.browser_blurs = int(reading.get("n_blurs", 0))

    left, right = st.columns([1, 1])
    with left:
        _render_case(st, rendered)
    with right:
        _render_rating_form(st, state, item, rendered, store, args.rater)


if __name__ == "__main__":  # pragma: no cover
    main()
