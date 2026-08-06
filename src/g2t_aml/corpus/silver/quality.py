"""Deduplication and degeneracy filtering, after verification and before the corpus.

A verified record is faithful. It is not automatically *useful*. Three failure modes get
through the checker untouched, because none of them is a factual error:

- **Degenerate text.** A truncated completion, an n-gram loop, a missing section, a
  leaked "Here is the narrative:" preamble. All perfectly faithful; none is a SAR.
- **A near-duplicate of another Silver record.** Two cases with similar facts can be
  rewritten into near-identical prose, which inflates the corpus without adding signal
  and leaks between splits if the pair straddles one.
- **A near-verbatim copy of its own Bronze source.** This is the one that matters most and
  the one a faithfulness metric will never flag: a "rewrite" that reproduces the template
  contributes nothing beyond Bronze, while still being counted as a Silver record and
  still costing an API call. A corpus full of them would report excellent verification
  numbers and be, in substance, Bronze with a different ``tier`` field.

**Per-teacher drop asymmetry is recorded, not just the totals.** If one teacher's outputs
are dropped far more often than the other's, the surviving corpus is skewed however even
the assignment was — and that asymmetry is itself a reportable finding about the two
models, so it goes in the report rather than being silently corrected.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from g2t_aml.corpus.dedupe import DEFAULT_THRESHOLD, find_near_duplicates, jaccard, shingles

__all__ = [
    "BRONZE_VERBATIM_THRESHOLD",
    "DEGENERACY_CHECKS",
    "FilterReport",
    "QualityConfig",
    "degeneracy_reasons",
    "filter_records",
]

#: Jaccard against a record's *own* Bronze source at or above which the rewrite is
#: considered a copy rather than a rewrite. Higher than the corpus-wide duplicate
#: threshold because the two questions are different: 0.85 asks "is this the same text as
#: some other record", 0.90 asks "did this rewrite actually do anything".
BRONZE_VERBATIM_THRESHOLD = 0.90

#: The degeneracy checks, in report order. Names are stable and are quoted in the phase
#: log, on the same terms as the ten-point harness's check names.
DEGENERACY_CHECKS: tuple[str, ...] = (
    "truncated",
    "repetitive",
    "too_few_sections",
    "stub_section",
    "preamble_or_markdown",
    "low_lexical_variety",
)

#: A completion that stops mid-sentence. Terminal punctuation only; a closing quote or
#: bracket after it is fine.
_TERMINAL_RE = re.compile(r"[.!?][\"')\]]*$")

#: Markdown and chat-preamble leakage. The prompt forbids all of it, so an occurrence is a
#: model that ignored the output contract, and its text is not a narrative.
_MARKDOWN_RE = re.compile(r"^\s*(#{1,6}\s|[-*+]\s|\d+\.\s|```)", re.MULTILINE)
_PREAMBLE_RE = re.compile(
    r"^\s*(here(?:'s| is)|sure[,!]|certainly|below is|i(?:'ve| have)\b|as requested|"
    r"the following is|narrative:|rewritten narrative)",
    re.IGNORECASE,
)

#: Window used for the loop check. Five words is long enough that an ordinary repeated
#: phrase ("the focal account") does not trip it and short enough to catch a model that
#: has started cycling a clause.
_LOOP_NGRAM = 5

_WORD_RE = re.compile(r"[A-Za-z0-9|.,%-]+")


@dataclass(frozen=True)
class QualityConfig:
    """Thresholds for the filtering pass.

    Attributes:
        dedup_jaccard: Corpus-wide near-duplicate threshold, Silver against Silver and
            against Bronze.
        bronze_verbatim_jaccard: Own-source copy threshold.
        min_sections: Paragraphs a narrative must have. The SAR structure is four parts;
            three is the floor that tolerates a model merging two closely related ones.
        min_section_words: Words below which a paragraph is a stub rather than a section.
        max_ngram_repeats: How often one 5-gram may occur before the text is looping.
        min_type_token_ratio: Distinct words over total words, below which the text is
            degenerate regardless of what repeats.
    """

    dedup_jaccard: float = DEFAULT_THRESHOLD
    bronze_verbatim_jaccard: float = BRONZE_VERBATIM_THRESHOLD
    min_sections: int = 3
    min_section_words: int = 12
    max_ngram_repeats: int = 3
    min_type_token_ratio: float = 0.35


@dataclass
class FilterReport:
    """What the filtering pass dropped, and why.

    Attributes:
        n_input: Records examined.
        n_kept: Records surviving.
        dropped: Case id to the reasons it was dropped.
        by_reason: Reason to count.
        by_teacher: Teacher to how many of its records were dropped.
        assigned_by_teacher: Teacher to how many it had before filtering, so a drop count
            can be read as a rate.
        bronze_similarity: Summary statistics of each surviving record's Jaccard against
            its own Bronze source — the number that says whether Silver is meaningfully
            distinct from Bronze at all.
        duplicate_pairs: Confirmed near-duplicate pairs, truncated for the report.
    """

    n_input: int = 0
    n_kept: int = 0
    dropped: dict[str, list[str]] = field(default_factory=dict)
    by_reason: dict[str, int] = field(default_factory=dict)
    by_teacher: dict[str, int] = field(default_factory=dict)
    assigned_by_teacher: dict[str, int] = field(default_factory=dict)
    bronze_similarity: dict[str, float] = field(default_factory=dict)
    duplicate_pairs: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_dropped(self) -> int:
        """Return how many records were dropped.

        Returns:
            The count.
        """
        return len(self.dropped)

    def drop_rate_by_teacher(self) -> dict[str, float]:
        """Return each teacher's drop rate.

        Returns:
            Teacher to the share of its records this pass removed. The spread between
            teachers is the asymmetry finding; a large one means the surviving corpus is
            not the corpus that was assigned.
        """
        return {
            teacher: round(self.by_teacher.get(teacher, 0) / total, 6) if total else 0.0
            for teacher, total in sorted(self.assigned_by_teacher.items())
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the machine-readable report.

        Returns:
            A JSON-serialisable mapping for ``silver_quality.json``.
        """
        rates = self.drop_rate_by_teacher()
        return {
            "n_input": self.n_input,
            "n_kept": self.n_kept,
            "n_dropped": self.n_dropped,
            "drop_rate": round(self.n_dropped / self.n_input, 6) if self.n_input else 0.0,
            "by_reason": dict(sorted(self.by_reason.items())),
            "by_teacher": dict(sorted(self.by_teacher.items())),
            "drop_rate_by_teacher": rates,
            "teacher_drop_rate_spread": (
                round(max(rates.values()) - min(rates.values()), 6) if rates else 0.0
            ),
            "bronze_similarity": self.bronze_similarity,
            "duplicate_pairs": self.duplicate_pairs[:20],
            "dropped_examples": dict(sorted(self.dropped.items())[:20]),
            "thresholds": {
                "dedup_jaccard": DEFAULT_THRESHOLD,
                "bronze_verbatim_jaccard": BRONZE_VERBATIM_THRESHOLD,
            },
        }

    def summary(self) -> str:
        """Return the human-readable summary.

        Returns:
            A short report for a terminal or the phase log.
        """
        lines = [
            f"quality filtering: {self.n_kept}/{self.n_input} kept, {self.n_dropped} dropped",
        ]
        for reason, count in sorted(self.by_reason.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {reason:<28} {count:>6}")
        if self.bronze_similarity:
            stats = self.bronze_similarity
            lines.append(
                f"  jaccard vs own Bronze: median {stats.get('median', 0):.3f}  "
                f"p95 {stats.get('p95', 0):.3f}  max {stats.get('max', 0):.3f}"
            )
        rates = self.drop_rate_by_teacher()
        if rates:
            lines.append(
                "  drop rate by teacher: " + "  ".join(f"{k} {v:.3f}" for k, v in rates.items())
            )
        return "\n".join(lines)


def degeneracy_reasons(text: str, config: QualityConfig | None = None) -> tuple[str, ...]:
    """Return every degeneracy check a narrative fails.

    Args:
        text: The canonicalised narrative.
        config: Thresholds.

    Returns:
        Zero or more names from :data:`DEGENERACY_CHECKS`. Empty means the text is
        well-formed — which is not the same as good, only not broken.
    """
    cfg = config if config is not None else QualityConfig()
    reasons: list[str] = []
    stripped = text.strip()
    paragraphs = [p for p in re.split(r"\n\s*\n", stripped) if p.strip()]
    words = _WORD_RE.findall(stripped.lower())

    if not stripped or not _TERMINAL_RE.search(stripped):
        reasons.append("truncated")

    if len(words) >= _LOOP_NGRAM:
        grams = Counter(
            tuple(words[i : i + _LOOP_NGRAM]) for i in range(len(words) - _LOOP_NGRAM + 1)
        )
        if grams and max(grams.values()) >= cfg.max_ngram_repeats:
            reasons.append("repetitive")

    if len(paragraphs) < cfg.min_sections:
        reasons.append("too_few_sections")
    elif any(len(_WORD_RE.findall(p)) < cfg.min_section_words for p in paragraphs):
        reasons.append("stub_section")

    if _MARKDOWN_RE.search(stripped) or _PREAMBLE_RE.match(stripped):
        reasons.append("preamble_or_markdown")

    if words and len(set(words)) / len(words) < cfg.min_type_token_ratio:
        reasons.append("low_lexical_variety")

    return tuple(reasons)


def filter_records(
    records: Sequence[dict[str, Any]],
    bronze_texts: dict[str, str],
    *,
    config: QualityConfig | None = None,
) -> tuple[list[dict[str, Any]], FilterReport]:
    """Drop degenerate, duplicate and Bronze-copying records.

    The order is deliberate and cheap-first: degeneracy is per record, the own-Bronze
    comparison is one Jaccard per record, and only the survivors go into the LSH pass,
    which is the expensive one. A degenerate record that also duplicates another would
    otherwise be counted twice and confuse the per-reason table.

    Args:
        records: Serialised Silver records.
        bronze_texts: Case id to that case's Bronze narrative, for the own-source
            comparison and for the corpus-wide Silver-against-Bronze pass.
        config: Thresholds.

    Returns:
        ``(kept records, report)``.
    """
    cfg = config if config is not None else QualityConfig()
    report = FilterReport(n_input=len(records))
    dropped: dict[str, list[str]] = {}

    for record in records:
        teacher = str(record.get("generator", {}).get("teacher", "-"))
        report.assigned_by_teacher[teacher] = report.assigned_by_teacher.get(teacher, 0) + 1

    survivors: list[dict[str, Any]] = []
    similarities: list[float] = []
    for record in records:
        case_id = str(record["case_id"])
        text = str(record["target_narrative"])
        reasons = list(degeneracy_reasons(text, cfg))

        source = bronze_texts.get(case_id)
        if source is not None:
            similarity = jaccard(shingles(text), shingles(source))
            if similarity >= cfg.bronze_verbatim_jaccard:
                reasons.append("bronze_verbatim")
            else:
                similarities.append(similarity)

        if reasons:
            dropped[case_id] = reasons
        else:
            survivors.append(record)

    # Corpus-wide near duplicates, Silver against Silver and against every Bronze
    # narrative. Bronze ids sort before Silver ids, so when a Silver record duplicates a
    # Bronze one the Silver record is what gets dropped -- which is the right way round:
    # Bronze is the reference tier and is never removed by a Silver build.
    pool: dict[str, str] = {f"bronze:{k}": v for k, v in bronze_texts.items()}
    pool.update({f"silver:{r['case_id']}": str(r["target_narrative"]) for r in survivors})
    duplicates = find_near_duplicates(pool, threshold=cfg.dedup_jaccard)
    duplicate_ids = {d.split(":", 1)[1] for d in duplicates.dropped if d.startswith("silver:")}
    report.duplicate_pairs = [
        {"kept": kept, "dropped": drop, "jaccard": round(score, 4)}
        for kept, drop, score in duplicates.pairs
        if drop.startswith("silver:")
    ]

    kept: list[dict[str, Any]] = []
    for record in survivors:
        case_id = str(record["case_id"])
        if case_id in duplicate_ids:
            dropped.setdefault(case_id, []).append("near_duplicate")
        else:
            kept.append(record)

    by_teacher: Counter[str] = Counter()
    by_reason: Counter[str] = Counter()
    teacher_of = {
        str(r["case_id"]): str(r.get("generator", {}).get("teacher", "-")) for r in records
    }
    for case_id, reasons in dropped.items():
        by_reason.update(reasons)
        by_teacher[teacher_of.get(case_id, "-")] += 1

    report.dropped = dropped
    report.by_reason = dict(by_reason)
    report.by_teacher = dict(by_teacher)
    report.n_kept = len(kept)
    report.bronze_similarity = _distribution(similarities)
    return kept, report


def _distribution(values: list[float]) -> dict[str, float]:
    """Summarise a distribution of similarities.

    Args:
        values: The observations.

    Returns:
        Count, min, median, p95, max and mean. Empty input yields an empty mapping, so an
        absent distribution is distinguishable from an all-zero one.
    """
    if not values:
        return {}
    ordered = sorted(values)
    n = len(ordered)

    def percentile(p: float) -> float:
        return ordered[min(n - 1, int(p * n))]

    return {
        "n": float(n),
        "min": round(ordered[0], 6),
        "median": round(percentile(0.5), 6),
        "p95": round(percentile(0.95), 6),
        "max": round(ordered[-1], 6),
        "mean": round(sum(ordered) / n, 6),
    }
