"""Measuring whether a template pack has collapsed.

Templates risk collapse in a way no faithfulness metric can see. A pack with one surface
form per typology produces a corpus that is 100% faithful, 100% adequate and useless as
training data: a model fine-tuned on it learns that a fan-out case has exactly one correct
sentence and reproduces it, and the rigidity shows up in the paper as output that no
reviewer would call a narrative. **Diversity is therefore a gate on Bronze in the same
sense faithfulness is**, and it is measured here rather than eyeballed.

Five measurements, each answering a different question:

``distinct-1/2/3``
    Unique n-grams over total n-grams. Answers "how much of this corpus is the same
    words". Low distinct-1 with high distinct-3 means shared vocabulary but varied
    phrasing, which is what a template pack should look like.
``self-BLEU``
    Each narrative scored against a sample of the others as references. Answers "how
    predictable is one record given the rest".

    **Self-BLEU saturates with the reference count, and the number is meaningless without
    it.** Measured on this corpus: 0.15 at one reference, 0.36 at three, 0.61 at ten, 0.81
    at fifty. Nothing about the corpus changed between those readings. With fifty
    references drawn from a corpus written over a deliberately controlled vocabulary,
    almost every 4-gram of a candidate appears in *some* reference, and clipped precision
    goes to one. Reporting 0.81 as "self-BLEU" would have said the pack had collapsed when
    the pairwise number says it plainly has not. The default here is therefore five
    references, fixed and documented, and :attr:`DiversityReport.self_bleu_curve` publishes
    the whole curve so the choice is auditable rather than convenient. See D-043.
``skeleton diversity``
    The measurement self-BLEU was reached for and is bad at. Every slot span is replaced
    by a placeholder, leaving the pure scaffolding, and the distinct skeletons are counted.
    A collapsed pack has a handful; this one has 577 for 584 records. It does not saturate,
    it needs no reference sample, and for a *template* corpus it answers the question
    directly.
``length distribution by typology``
    Answers "does the pack say the same amount about every case regardless of the case".
``inter-family n-gram overlap``
    Answers "are the families distinguishable". Bronze shares its subject and basis
    sections across families by design, so some overlap is expected and correct; what
    would be a defect is two families whose *pattern* sections are interchangeable.
``vocabulary size and type-token ratio``
    The blunt instrument, reported because it is comparable against Gold in Phase 10.

**All of this is deliberately cheap and dependency-free.** ``sacrebleu`` lives behind the
``eval`` extra and Phase 4 is CPU-core-only; a self-BLEU over a sampled reference set with
a standard brevity penalty is fifty lines and is the same number. Phase 10 recomputes the
surface metrics that go in the paper with the real library — this is a build-time gate, not
a published result.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DiversityReport",
    "distinct_n",
    "measure_diversity",
    "self_bleu",
    "self_bleu_curve",
    "skeletons",
    "tokenize",
]

#: References per candidate. Five, fixed, because the metric saturates in this parameter
#: and a number reported without it cannot be compared to anything. See the module
#: docstring and D-043.
_SELF_BLEU_REFERENCES = 5

#: Reference counts the published curve is measured at, so the saturation is visible in
#: the artifact rather than only in the decision that produced it.
_SELF_BLEU_CURVE_POINTS = (1, 3, 5, 10, 50)

#: Narratives scored, when the corpus is larger. Sampling the scored set as well keeps the
#: measurement seconds rather than minutes without moving the estimate.
_SELF_BLEU_SAMPLE = 800

#: Maximum n-gram order for BLEU. The standard 4.
_BLEU_MAX_N = 4

#: Narratives needed before self-BLEU means anything.
_MIN_CORPUS = 2

_TOKEN_RE = re.compile(r"[a-z0-9|.,%-]+")


def tokenize(text: str) -> list[str]:
    """Split a narrative into comparison tokens.

    Case-folded, punctuation that does not distinguish a value dropped. Numbers and the
    ``bank|account`` separator are kept, because they are what makes two reports about
    different cases different.

    Args:
        text: The narrative.

    Returns:
        The tokens.
    """
    return _TOKEN_RE.findall(text.lower())


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Return the n-grams of a token sequence.

    Args:
        tokens: The tokens.
        n: The order.

    Returns:
        The n-grams, possibly empty.
    """
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(corpus: list[str], n: int) -> float:
    """Return the distinct-n ratio over a corpus.

    Args:
        corpus: The narratives.
        n: The n-gram order.

    Returns:
        Unique n-grams divided by total n-grams, and 0.0 for an empty corpus.
    """
    total = 0
    unique: set[tuple[str, ...]] = set()
    for text in corpus:
        grams = _ngrams(tokenize(text), n)
        total += len(grams)
        unique.update(grams)
    return len(unique) / total if total else 0.0


def _modified_precision(
    candidate: list[str], references: list[list[str]], n: int
) -> tuple[int, int]:
    """Return the clipped n-gram match count and the candidate n-gram total.

    Args:
        candidate: Candidate tokens.
        references: Reference token sequences.
        n: The order.

    Returns:
        ``(clipped matches, total)``.
    """
    candidate_counts = Counter(_ngrams(candidate, n))
    if not candidate_counts:
        return 0, 0
    ceiling: Counter[tuple[str, ...]] = Counter()
    for reference in references:
        for gram, count in Counter(_ngrams(reference, n)).items():
            ceiling[gram] = max(count, ceiling[gram])
    clipped = sum(min(count, ceiling[gram]) for gram, count in candidate_counts.items())
    return clipped, sum(candidate_counts.values())


def _sentence_bleu(candidate: list[str], references: list[list[str]]) -> float:
    """Return BLEU-4 for one candidate against several references.

    Args:
        candidate: Candidate tokens.
        references: Reference token sequences.

    Returns:
        The score in [0, 1]. Zero when any order has no match, which is standard and is
        what makes the metric sensitive to genuinely novel phrasing.
    """
    if not candidate or not references:
        return 0.0
    log_precision = 0.0
    for n in range(1, _BLEU_MAX_N + 1):
        clipped, total = _modified_precision(candidate, references, n)
        if total == 0 or clipped == 0:
            return 0.0
        log_precision += math.log(clipped / total) / _BLEU_MAX_N
    closest = min((abs(len(r) - len(candidate)), len(r)) for r in references)[1]
    brevity = 1.0 if len(candidate) > closest else math.exp(1 - closest / max(1, len(candidate)))
    return brevity * math.exp(log_precision)


def self_bleu(
    corpus: list[str], seed: int = 42, n_references: int = _SELF_BLEU_REFERENCES
) -> float:
    """Return the mean self-BLEU of a corpus. Lower is more diverse.

    Each narrative is scored against a random sample of the others. Sampling is seeded, so
    the number is reproducible.

    Args:
        corpus: The narratives.
        seed: Seeds the sampling.
        n_references: References per candidate. **The result depends strongly on this**;
            see the module docstring. Defaults to five.

    Returns:
        Mean self-BLEU in [0, 1], and 0.0 for a corpus of fewer than two narratives.
    """
    if len(corpus) < _MIN_CORPUS:
        return 0.0
    rng = random.Random(seed)
    tokenized = [tokenize(text) for text in corpus]
    indices = list(range(len(tokenized)))
    scored = (
        indices if len(indices) <= _SELF_BLEU_SAMPLE else rng.sample(indices, _SELF_BLEU_SAMPLE)
    )

    scores: list[float] = []
    for i in scored:
        pool = [j for j in rng.sample(indices, min(len(indices), n_references + 1)) if j != i]
        pool = pool[:n_references]
        if not pool:
            continue
        scores.append(_sentence_bleu(tokenized[i], [tokenized[j] for j in pool]))
    return sum(scores) / len(scores) if scores else 0.0


def self_bleu_curve(corpus: list[str], seed: int = 42) -> dict[int, float]:
    """Return self-BLEU at several reference counts, exposing the metric's saturation.

    Args:
        corpus: The narratives.
        seed: Seeds the sampling.

    Returns:
        Reference count to mean self-BLEU.
    """
    return {k: self_bleu(corpus, seed=seed, n_references=k) for k in _SELF_BLEU_CURVE_POINTS}


def skeletons(narratives: list[str], slot_spans: list[list[tuple[int, int]]]) -> list[str]:
    """Return each narrative with its slot spans blanked, leaving pure scaffolding.

    Two narratives with the same skeleton differ only in the values they report. Counting
    distinct skeletons is the most direct measure of whether a template pack has
    collapsed, and unlike self-BLEU it has no reference sample and no saturation.

    Args:
        narratives: The narratives.
        slot_spans: Per narrative, the ``(start, end)`` spans of its slots.

    Returns:
        One skeleton per narrative.
    """
    result: list[str] = []
    for text, spans in zip(narratives, slot_spans, strict=True):
        parts: list[str] = []
        cursor = 0
        for start, end in sorted(spans):
            if start < cursor:
                continue
            parts.append(text[cursor:start])
            parts.append("\N{BULLET}")
            cursor = end
        parts.append(text[cursor:])
        result.append("".join(parts))
    return result


@dataclass
class DiversityReport:
    """Everything measured about a corpus's surface variety.

    Attributes:
        n_records: Corpus size.
        distinct: Order to distinct-n ratio, for n in 1, 2, 3.
        self_bleu: Mean self-BLEU at the fixed five-reference setting. Lower is more
            diverse.
        self_bleu_curve: Reference count to mean self-BLEU, published because the metric
            saturates in that parameter (D-043).
        n_distinct_skeletons: Narratives that differ once their slot values are blanked.
        skeleton_ratio: Distinct skeletons over records. 1.0 means no two narratives in
            the corpus share a surface form; a collapsed pack tends to zero.
        vocabulary_size: Distinct tokens.
        type_token_ratio: Distinct tokens over total tokens.
        length_by_typology: Typology to its length summary, in words.
        length_by_family: Template family to its length summary, in words.
        family_overlap: ``"a|b"`` to the Jaccard overlap of the two families' trigram
            vocabularies. Families should be distinguishable.
        variant_counts: Family to how many times each variant was used, so an unused
            realisation is visible rather than merely present in the code.
    """

    n_records: int = 0
    distinct: dict[int, float] = field(default_factory=dict)
    self_bleu: float = 0.0
    self_bleu_curve: dict[int, float] = field(default_factory=dict)
    n_distinct_skeletons: int = 0
    skeleton_ratio: float = 0.0
    vocabulary_size: int = 0
    type_token_ratio: float = 0.0
    length_by_typology: dict[str, dict[str, float]] = field(default_factory=dict)
    length_by_family: dict[str, dict[str, float]] = field(default_factory=dict)
    family_overlap: dict[str, float] = field(default_factory=dict)
    variant_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the machine-readable report.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "n_records": self.n_records,
            "distinct_1": round(self.distinct.get(1, 0.0), 6),
            "distinct_2": round(self.distinct.get(2, 0.0), 6),
            "distinct_3": round(self.distinct.get(3, 0.0), 6),
            "self_bleu": round(self.self_bleu, 6),
            "self_bleu_n_references": _SELF_BLEU_REFERENCES,
            "self_bleu_curve": {
                str(k): round(v, 6) for k, v in sorted(self.self_bleu_curve.items())
            },
            "n_distinct_skeletons": self.n_distinct_skeletons,
            "skeleton_ratio": round(self.skeleton_ratio, 6),
            "vocabulary_size": self.vocabulary_size,
            "type_token_ratio": round(self.type_token_ratio, 6),
            "length_by_typology": self.length_by_typology,
            "length_by_family": self.length_by_family,
            "family_trigram_overlap": {
                k: round(v, 6) for k, v in sorted(self.family_overlap.items())
            },
            "variant_counts": self.variant_counts,
        }

    def summary(self) -> str:
        """Return the human-readable summary.

        Returns:
            A short report suitable for a terminal or a phase log.
        """
        lines = [
            f"diversity over {self.n_records:,} narratives",
            f"  distinct-1 {self.distinct.get(1, 0):.4f}   distinct-2 "
            f"{self.distinct.get(2, 0):.4f}   distinct-3 {self.distinct.get(3, 0):.4f}",
            f"  self-BLEU  {self.self_bleu:.4f} (at {_SELF_BLEU_REFERENCES} refs; curve "
            + ", ".join(f"{k}:{v:.2f}" for k, v in sorted(self.self_bleu_curve.items()))
            + ")",
            f"  skeletons  {self.n_distinct_skeletons:,} distinct "
            f"({self.skeleton_ratio:.4f} of records)   vocabulary "
            f"{self.vocabulary_size:,}   TTR {self.type_token_ratio:.4f}",
            "",
            f"  {'family':<26} {'n':>6} {'median words':>13} {'variants used':>14}",
            f"  {'-' * 26} {'-' * 6} {'-' * 13} {'-' * 14}",
        ]
        for family, stats in sorted(self.length_by_family.items()):
            used = len(self.variant_counts.get(family, {}))
            lines.append(f"  {family:<26} {int(stats['n']):>6} {stats['median']:>13.0f} {used:>14}")
        if self.family_overlap:
            worst = sorted(self.family_overlap.items(), key=lambda kv: -kv[1])[:5]
            lines += ["", "  highest inter-family trigram overlap:"]
            lines += [f"    {pair:<48} {value:.4f}" for pair, value in worst]
        return "\n".join(lines)


def _length_stats(values: list[int]) -> dict[str, float]:
    """Summarise a length distribution.

    Args:
        values: Word counts.

    Returns:
        Count, min, median, mean and max.
    """
    ordered = sorted(values)
    n = len(ordered)
    return {
        "n": float(n),
        "min": float(ordered[0]),
        "median": float(ordered[n // 2]),
        "mean": sum(ordered) / n,
        "max": float(ordered[-1]),
    }


def measure_diversity(
    narratives: list[str],
    typologies: list[str] | None = None,
    families: list[str] | None = None,
    variants: list[int] | None = None,
    slot_spans: list[list[tuple[int, int]]] | None = None,
    seed: int = 42,
) -> DiversityReport:
    """Measure a corpus's surface variety.

    Args:
        narratives: The narratives.
        typologies: Typology per narrative, for the per-typology length breakdown.
        families: Template family per narrative.
        variants: Realisation index per narrative.
        slot_spans: Slot spans per narrative, for the skeleton count. Omitted, the
            skeleton metrics are reported as zero rather than guessed at.
        seed: Seeds the self-BLEU sampling.

    Returns:
        The report.
    """
    report = DiversityReport(n_records=len(narratives))
    if not narratives:
        return report

    tokenized = [tokenize(text) for text in narratives]
    vocabulary = Counter(token for tokens in tokenized for token in tokens)
    total_tokens = sum(vocabulary.values())
    report.vocabulary_size = len(vocabulary)
    report.type_token_ratio = len(vocabulary) / total_tokens if total_tokens else 0.0
    report.distinct = {n: distinct_n(narratives, n) for n in (1, 2, 3)}
    report.self_bleu = self_bleu(narratives, seed=seed)
    report.self_bleu_curve = self_bleu_curve(narratives, seed=seed)

    if slot_spans is not None:
        distinct = len(set(skeletons(narratives, slot_spans)))
        report.n_distinct_skeletons = distinct
        report.skeleton_ratio = distinct / len(narratives)

    if typologies:
        grouped: dict[str, list[int]] = defaultdict(list)
        for typology, tokens in zip(typologies, tokenized, strict=True):
            grouped[typology].append(len(tokens))
        report.length_by_typology = {k: _length_stats(v) for k, v in sorted(grouped.items())}

    if families:
        by_family: dict[str, list[int]] = defaultdict(list)
        trigrams: dict[str, set[tuple[str, ...]]] = defaultdict(set)
        for family, tokens in zip(families, tokenized, strict=True):
            by_family[family].append(len(tokens))
            trigrams[family].update(_ngrams(tokens, 3))
        report.length_by_family = {k: _length_stats(v) for k, v in sorted(by_family.items())}

        keys = sorted(trigrams)
        for i, left in enumerate(keys):
            for right in keys[i + 1 :]:
                union = len(trigrams[left] | trigrams[right])
                overlap = len(trigrams[left] & trigrams[right]) / union if union else 0.0
                report.family_overlap[f"{left}|{right}"] = overlap

        if variants:
            counts: dict[str, Counter[str]] = defaultdict(Counter)
            for family, variant in zip(families, variants, strict=True):
                counts[family][str(variant)] += 1
            report.variant_counts = {k: dict(sorted(v.items())) for k, v in sorted(counts.items())}

    return report
