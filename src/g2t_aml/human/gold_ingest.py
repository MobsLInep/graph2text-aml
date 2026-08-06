"""Turning reviewed annotations into ``gold.jsonl``, gated by the same ten checks.

Gold is written into ``training_record_v1`` with ``tier="gold"`` and passes through the
**same** ten-point harness as Bronze and Silver (D-037). That is the whole point of one
schema for three tiers: "Gold is verified" then means exactly what "Silver is verified"
means, and a reviewer can compare the tiers without first working out whether the two
claims are about the same thing.

**Human-written narratives have no slots, so the alignment produces them.** Bronze carries
a character-span alignment from every load-bearing value back to the fact field it came
from; a Gold narrative that states a fact faithfully states the same value, so
:class:`~g2t_aml.corpus.silver.claim_extraction.SlotAlignmentExtractor` locates it in the
Gold text and emits the slot there. The Bronze narrative is used **only here, at ingestion
time** — it is never loaded by the annotation interface, and
:class:`~g2t_aml.human.caseloader.AnnotationCase` has no field that could carry it. An
annotator who has seen a template rendering is editing it.

Three consequences of aligning rather than parsing, all of them intended:

1. A value the annotator wrote in their own format — "roughly 9,400 Canadian Dollars"
   against Bronze's "9,434.82 Canadian Dollar" — does not align. It is reported as a
   dropped fact and as an unbacked quantity, and the record's unverifiable rate rises.
   The conservative direction: a human paraphrase costs budget and shows up in the report
   where someone can look at it, rather than being silently accepted by a fuzzy matcher
   that would also silently accept a real error (D-048).
2. A quantity the annotator introduced that aligns to nothing becomes a claim naming no
   field, which the checker resolves to UNVERIFIABLE. Enough of them fail the record. That
   is correct: an invented figure has not been shown wrong, it has been shown unbacked.
3. Salience coverage is measured from the alignment, so it is the same measurement the
   automated metric makes on generated text.

**Nothing is ingested without an accepted second review.** An item awaiting review, or
one whose review disputes it without an adjudication, is held and reported — not dropped
quietly and not admitted on the grounds that the narrative looks fine.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from g2t_aml.corpus.graphref import build_graph_ref
from g2t_aml.corpus.record import BronzeNarrative, SlotAnnotation, TrainingRecord
from g2t_aml.corpus.silver.claim_extraction import canonicalise_narrative, extract_report
from g2t_aml.corpus.tokenization import TokenCounter, get_token_counter, word_count
from g2t_aml.corpus.validate import MAX_UNVERIFIABLE_RATE
from g2t_aml.facts.checkers import CheckContext, check_claim, check_narrative_text, summarise
from g2t_aml.facts.salience import salience_report
from g2t_aml.facts.schema import CaseFacts
from g2t_aml.facts.serialiser import serialise_facts
from g2t_aml.facts.vocab import ControlledVocabulary, load_vocabulary
from g2t_aml.human.review import Review, ReviewError
from g2t_aml.human.store import Annotation

__all__ = [
    "GOLD_INGEST_VERSION",
    "GOLD_TIER",
    "GoldIngestError",
    "GoldIngestReport",
    "IngestedItem",
    "bronze_narrative_from_record",
    "ingest_annotations",
]

#: The tier every record written here carries.
GOLD_TIER = "gold"

#: The version this ingestion writes into ``generator.renderer_version``.
#:
#: ``training_record_v1`` is FROZEN and requires that field on every record, with the
#: stated meaning "version of the code that produced the narrative, so a corpus can be
#: attributed to an exact renderer". For Gold the narrative was produced by a person, so
#: the honest reading of the required field is the version of the *pipeline that produced
#: the record* — this ingestion together with the annotation protocol it enforces. Bumping
#: the schema to make the field optional would invalidate every Bronze and Silver record
#: on disk (invariant 9) to accommodate one tier's vocabulary, which is the wrong trade.
#: ``generator.protocol`` names the guidelines document alongside it, and
#: ``generator.annotator_id`` names the author, so nothing is obscured by the reuse.
#: See D-053.
GOLD_INGEST_VERSION = "1.0.0"


class GoldIngestError(RuntimeError):
    """Raised when ingestion cannot proceed for a structural reason."""


def bronze_narrative_from_record(payload: dict[str, Any]) -> BronzeNarrative:
    """Rebuild the Bronze narrative and its slots from a corpus record.

    Args:
        payload: A line from ``bronze.jsonl``.

    Returns:
        The narrative, with its slot alignment. Only ``slots`` is load-bearing here; the
        text is carried so the alignment can be inspected when a case behaves oddly.

    Raises:
        GoldIngestError: If the record carries no slot annotations, which would make the
            alignment vacuous and every Gold record for that case slot-free — and a
            slot-free record passes ``zero_contradicted`` by having nothing to check.
    """
    slots = tuple(SlotAnnotation.from_dict(s) for s in payload.get("target_slots") or ())
    if not slots:
        raise GoldIngestError(
            f"the Bronze record for {payload.get('case_id')!r} carries no slot "
            "annotations, so there is nothing to align a Gold narrative against. A "
            "record with no slots makes no claims and would pass verification vacuously."
        )
    generator = payload.get("generator") or {}
    return BronzeNarrative(
        case_id=str(payload["case_id"]),
        text=str(payload["target_narrative"]),
        annotated="",
        slots=slots,
        family=str(generator.get("family", "unknown")),
        variant=int(generator.get("variant", 0)),
    )


@dataclass(frozen=True)
class IngestedItem:
    """One Gold record, and what the alignment found on the way to it.

    Attributes:
        record: The training record, ready to serialise.
        annotation: The annotation it came from.
        review: The accepted review that admitted it.
        aligned_paths: Fact fields the narrative demonstrably states.
        dropped_paths: Fields Bronze stated that this narrative does not. Not an error —
            a Gold narrative is not a paraphrase of Bronze and is free to select
            differently. Only a dropped *salient* field costs coverage.
        added_spans: Quantities the narrative carries that align to nothing.
    """

    record: TrainingRecord
    annotation: Annotation
    review: Review
    aligned_paths: tuple[str, ...]
    dropped_paths: tuple[str, ...]
    added_spans: tuple[tuple[int, int, str], ...]


@dataclass
class GoldIngestReport:
    """What ingestion produced, and everything it refused.

    Attributes:
        items: The ingested records.
        held: ``case_id -> reason`` for every annotation not ingested. Held, not dropped:
            each of these is a case the reservation is still holding out and that someone
            has already spent fifteen minutes on.
        by_annotator: Records ingested per annotator.
        flag_overrides: ``rule -> (raised, overridden)`` across every ingested item. The
            output that makes the live validation rules themselves measurable.
        salience_coverage: Per-record salience coverage, for the distribution.
        n_reviews: Reviews considered.
        n_adjudicated: Items whose review recorded an adjudication.
    """

    items: list[IngestedItem] = field(default_factory=list)
    held: dict[str, str] = field(default_factory=dict)
    by_annotator: dict[str, int] = field(default_factory=dict)
    flag_overrides: dict[str, tuple[int, int]] = field(default_factory=dict)
    salience_coverage: list[float] = field(default_factory=list)
    n_reviews: int = 0
    n_adjudicated: int = 0

    @property
    def n_ingested(self) -> int:
        """Return how many records were produced.

        Returns:
            The count.
        """
        return len(self.items)

    @property
    def mean_salience_coverage(self) -> float:
        """Return mean salience coverage over the ingested records.

        Returns:
            The mean, or 0.0 when nothing was ingested.
        """
        return (
            sum(self.salience_coverage) / len(self.salience_coverage)
            if self.salience_coverage
            else 0.0
        )

    def payloads(self) -> list[dict[str, Any]]:
        """Return the serialised training records.

        Returns:
            One mapping per ingested item, in case-id order.
        """
        return [i.record.to_dict() for i in sorted(self.items, key=lambda i: i.record.case_id)]

    def to_dict(self) -> dict[str, Any]:
        """Return the machine-readable quality report.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "n_ingested": self.n_ingested,
            "n_held": len(self.held),
            "held": dict(sorted(self.held.items())),
            "by_annotator": dict(sorted(self.by_annotator.items())),
            "n_reviews": self.n_reviews,
            "n_adjudicated": self.n_adjudicated,
            "mean_salience_coverage": round(self.mean_salience_coverage, 6),
            "flag_overrides": {
                rule: {
                    "raised": raised,
                    "overridden": overridden,
                    "override_rate": round(overridden / raised, 6) if raised else 0.0,
                }
                for rule, (raised, overridden) in sorted(self.flag_overrides.items())
            },
            "alignment": {
                "mean_aligned_paths": round(
                    sum(len(i.aligned_paths) for i in self.items) / self.n_ingested, 3
                )
                if self.items
                else 0.0,
                "mean_unaligned_quantities": round(
                    sum(len(i.added_spans) for i in self.items) / self.n_ingested, 3
                )
                if self.items
                else 0.0,
            },
        }

    def summary(self) -> str:
        """Return the human-readable summary.

        Returns:
            A short report suitable for a terminal or the phase log.
        """
        lines = [
            f"Gold ingestion: {self.n_ingested} records from {self.n_reviews} reviews "
            f"({self.n_adjudicated} adjudicated), {len(self.held)} held",
            f"  mean salience coverage {self.mean_salience_coverage:.3f}",
        ]
        if self.by_annotator:
            lines += ["", f"  {'annotator':<16} {'records':>8}"]
            lines += [f"  {name:<16} {n:>8}" for name, n in sorted(self.by_annotator.items())]
        overridden = {r: v for r, v in self.flag_overrides.items() if v[1]}
        if overridden:
            lines += ["", "  Overridden validation rules (a high rate means the rule is wrong):"]
            lines += [
                f"    {rule}: {ov} of {raised} overridden"
                for rule, (raised, ov) in sorted(overridden.items(), key=lambda kv: -kv[1][1])
            ]
        if self.held:
            lines += ["", "  Held:"]
            lines += [f"    {case}: {why}" for case, why in sorted(self.held.items())]
        return "\n".join(lines)


def _choose(annotations: list[Annotation], review: Review) -> Annotation:
    """Pick the annotation the review admits into the corpus.

    Args:
        annotations: Every annotation for the case.
        review: Its accepted review.

    Returns:
        The chosen annotation.

    Raises:
        GoldIngestError: If the review's choice does not match an annotation.
    """
    if len(annotations) == 1:
        return annotations[0]
    chosen = next((a for a in annotations if a.annotator_id == review.chosen_annotator), None)
    if chosen is None:
        raise GoldIngestError(
            f"review of {review.case_id!r} chooses {review.chosen_annotator!r}, who has "
            "no annotation for it"
        )
    return chosen


def _build_record(
    annotation: Annotation,
    facts: CaseFacts,
    bronze: BronzeNarrative,
    review: Review,
    *,
    split: str,
    case_store: Path,
    repo_root: Path,
    vocabulary: ControlledVocabulary,
    counter: TokenCounter,
) -> tuple[TrainingRecord, tuple[str, ...], tuple[str, ...], tuple[tuple[int, int, str], ...]]:
    """Turn one accepted annotation into a training record.

    Args:
        annotation: The accepted annotation.
        facts: Its fact record.
        bronze: The Bronze narrative for the same case, used only for its slot alignment.
        review: The accepted review.
        split: The split, from the frozen manifest. Always ``test`` for Gold.
        case_store: Directory holding the case membership tables.
        repo_root: Repository root, for the graph reference.
        vocabulary: The controlled vocabulary.
        counter: The token counter.

    Returns:
        ``(record, aligned_paths, dropped_paths, added_spans)``.
    """
    narrative = canonicalise_narrative(annotation.narrative)
    report = extract_report(narrative, facts, bronze, vocabulary=vocabulary)

    context = CheckContext(facts=facts, vocabulary=vocabulary)
    results = [check_claim(c, context) for c in report.claims]
    results += check_narrative_text(narrative, context)
    verdicts = summarise(results)

    salience = salience_report(facts, vocabulary)
    mentioned = [p for p in salience.required if p in report.aligned_paths]

    record = TrainingRecord(
        case_id=annotation.case_id,
        dataset=annotation.dataset,
        split=split,
        tier=GOLD_TIER,
        facts=facts,
        graph_ref=build_graph_ref(case_store, annotation.case_id, repo_root),
        serialised_facts=serialise_facts(facts, style="compact"),
        target_narrative=narrative,
        target_slots=report.aligned_slots,
        generator={
            "method": "human",
            "annotator_id": annotation.annotator_id,
            "reviewer_id": review.reviewer_id,
            "seconds_spent": round(annotation.seconds_spent, 1),
            "revision_count": annotation.revision_count,
            "typology_assigned": annotation.typology_assigned,
            "difficulty": annotation.difficulty,
            "n_flags_raised": len(annotation.flags),
            "n_flags_overridden": annotation.n_overridden,
            "adjudicated": review.adjudication is not None,
            "adjudication": review.adjudication.to_dict() if review.adjudication else None,
            "protocol": "docs/annotation/annotation_guidelines.md",
            "renderer_version": GOLD_INGEST_VERSION,
        },
        verification={
            "supported": verdicts["by_verdict"]["supported"],
            "contradicted": verdicts["by_verdict"]["contradicted"],
            "unverifiable": verdicts["by_verdict"]["unverifiable"],
            "unverifiable_rate": round(verdicts["unverifiable_rate"], 6),
            "n_claims": verdicts["n_claims"],
            "critical_error_rate": round(verdicts["critical_error_rate"], 6),
            "by_hallucination_class": verdicts["by_hallucination_class"],
        },
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
            "coverage": round(len(mentioned) / len(salience.required), 6)
            if salience.required
            else 1.0,
        },
    )
    return record, report.aligned_paths, report.dropped_paths, report.added_spans


def ingest_annotations(  # noqa: PLR0912, PLR0915 - one linear pass with one `continue`
    # per reason an item can be held. Extracting them would separate each refusal from the
    # message that explains it, and the messages are the report.
    annotations: list[Annotation],
    reviews: list[Review],
    facts_by_case: dict[str, CaseFacts],
    bronze_by_case: dict[str, BronzeNarrative],
    *,
    split_assignment: dict[str, str],
    case_store: Path,
    repo_root: Path,
    vocabulary: ControlledVocabulary | None = None,
    token_counter: TokenCounter | None = None,
    max_unverifiable_rate: float = MAX_UNVERIFIABLE_RATE,
) -> GoldIngestReport:
    """Build Gold training records from reviewed annotations.

    Args:
        annotations: Every non-calibration annotation.
        reviews: Every second-reviewer pass.
        facts_by_case: The fact record per case.
        bronze_by_case: The Bronze narrative per case, for its slot alignment. Used here
            and only here.
        split_assignment: Case id to split, from the frozen manifest.
        case_store: Directory holding the case membership tables.
        repo_root: Repository root, for graph references.
        vocabulary: The controlled vocabulary. Loaded from disk when omitted.
        token_counter: The token counter. The heuristic counter when omitted.
        max_unverifiable_rate: The per-record unverifiable budget, enforced here against
            the *extractor's* rate rather than the harness's — see the comment at the
            check, and :data:`~g2t_aml.corpus.validate.MAX_UNVERIFIABLE_RATE` for the
            budget itself.

    Returns:
        The report, holding both the records and the account of everything refused.
    """
    vocab = vocabulary if vocabulary is not None else load_vocabulary()
    counter = token_counter if token_counter is not None else get_token_counter()

    by_case: dict[str, list[Annotation]] = defaultdict(list)
    for annotation in annotations:
        by_case[annotation.case_id].append(annotation)

    latest_review: dict[str, Review] = {}
    for review in reviews:
        latest_review[review.case_id] = review

    report = GoldIngestReport(n_reviews=len(latest_review))
    flags: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for case_id in sorted(by_case):
        group = sorted(by_case[case_id], key=lambda a: a.annotator_id)
        review = latest_review.get(case_id)
        if review is None:
            report.held[case_id] = "awaiting second review"
            continue
        try:
            review.validate_against(tuple(a.annotator_id for a in group))
        except ReviewError as exc:
            report.held[case_id] = f"review is not independent: {exc}"
            continue
        if not review.accepted:
            report.held[case_id] = f"review verdict {review.verdict.value!r}" + (
                f"; adjudicated: {review.adjudication.decision}" if review.adjudication else ""
            )
            continue

        facts = facts_by_case.get(case_id)
        bronze = bronze_by_case.get(case_id)
        if facts is None:
            report.held[case_id] = "no fact record"
            continue
        if bronze is None:
            report.held[case_id] = "no Bronze record to align against"
            continue
        split = split_assignment.get(case_id)
        if split is None:
            report.held[case_id] = "absent from the frozen split manifest"
            continue

        try:
            chosen = _choose(group, review)
        except GoldIngestError as exc:
            report.held[case_id] = str(exc)
            continue

        record, aligned, dropped, added = _build_record(
            chosen,
            facts,
            bronze,
            review,
            split=split,
            case_store=case_store,
            repo_root=repo_root,
            vocabulary=vocab,
            counter=counter,
        )

        # The ten-point harness rebuilds a record's claims from its `target_slots`, and a
        # Gold record's slots are exactly the values that DID align — so an unaligned
        # quantity is invisible to check 5 there, however many of them there are. The
        # extractor saw them; its rate is on the record; and this is where it is enforced.
        # Without this, a Gold narrative could carry a dozen unbacked figures and pass the
        # gate by virtue of not having produced slots for any of them.
        rate = float(record.verification["unverifiable_rate"])
        if rate > max_unverifiable_rate:
            report.held[case_id] = (
                f"unverifiable rate {rate:.3f} exceeds {max_unverifiable_rate}: "
                f"{len(added)} quantities in the narrative align to no fact field "
                f"({[t for _, _, t in added[:4]]}). Either they restate a fact in a "
                "different format, or they are unbacked."
            )
            continue

        report.items.append(
            IngestedItem(
                record=record,
                annotation=chosen,
                review=review,
                aligned_paths=aligned,
                dropped_paths=dropped,
                added_spans=added,
            )
        )
        report.by_annotator[chosen.annotator_id] = (
            report.by_annotator.get(chosen.annotator_id, 0) + 1
        )
        report.salience_coverage.append(float(record.salience["coverage"]))
        if review.adjudication is not None:
            report.n_adjudicated += 1
        for outcome in chosen.flags:
            counts = flags[outcome.flag.rule]
            counts[0] += 1
            counts[1] += int(outcome.overridden)

    report.flag_overrides = {rule: (raised, ov) for rule, (raised, ov) in flags.items()}  # noqa: C416 - the tuple rebuild is the point: `flags` holds mutable lists
    return report
