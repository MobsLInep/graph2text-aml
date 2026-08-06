"""Inter-annotator agreement, on the three things that can actually disagree.

Two people writing SAR narratives for one case will not write the same text, and that is
not a defect. So this module measures agreement on the things where disagreement *is*
meaningful, and reports the thing where it is not as evidence rather than as a failure.

**Typology assignment — Cohen's kappa.** A categorical judgement over a closed vocabulary
with exactly two raters per item, which is the case kappa was designed for. Chance
correction matters here more than usual: ``unclassified`` is 88% of the test split, so two
annotators who both answered "unclassified" every time would score 88% raw agreement and a
kappa of zero, which is the correct answer about how much they agreed on anything.

**Content selection — Jaccard over mentioned salient fields.** The question this answers is
"did they think the same facts mattered", which is closer to the thing being evaluated than
any surface metric. Jaccard rather than kappa because the unit is a *set* per item, not a
label.

**Text similarity — reported, never gated.** Two faithful narratives about one case share
their numbers and little else. A low score here is the finding that narrative writing has
legitimate variance, which is exactly why the Gold set cannot be evaluated by string
overlap and why the faithfulness machinery exists. Reporting it and saying so is more
honest than omitting it and letting a reviewer wonder.

**Krippendorff's alpha over the pooled typology judgements** is reported alongside kappa,
because ``docs/annotation/README.md`` promised alpha in Phase 3 and because alpha handles
the ragged design — most items have one annotator, 15% have two — where kappa can only
speak about the doubled subset. They answer different questions and both are printed.
The nominal-alpha implementation here is dependency-free and is checked against a
hand-computed fixture; there is no import of a statistics package that would have to be
installed to read an agreement report.

**Who double-annotates what is decided before anyone starts**, by
:func:`is_double_annotated`, which hashes the case id. Assigning it adaptively — "give the
hard ones to two people" — would make the agreement number a statement about the easy
items, and choosing after the fact would make it a statement about nothing.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from g2t_aml.human.store import Annotation

__all__ = [
    "AgreementReport",
    "PairAgreement",
    "cohens_kappa",
    "is_double_annotated",
    "jaccard",
    "krippendorff_alpha_nominal",
    "measure_agreement",
    "token_f1",
]

#: Words carrying no content for the surface-similarity comparison. Deliberately short: a
#: longer list would start removing AML vocabulary and make the number look better than it
#: is, which is the opposite of what this measurement is for.
_STOPWORDS = frozenset(
    "a an the of to in on at by for and or is was were be been are it its this that as with"
    " from into over under between within".split()
)

#: A unit coded by fewer raters than this contributes nothing to alpha.
_MIN_RATERS_FOR_ALPHA = 2

_WORD_RE = re.compile(r"[a-z0-9|.,]+")


def is_double_annotated(case_id: str, rate: float, *, salt: str = "gold-double") -> bool:
    """Decide, deterministically, whether a case gets two annotators.

    Hashing rather than drawing means the decision is a pure function of the case id: it
    is the same on every machine, is fixed before the first item is opened, and does not
    change when the sample is extended. An adaptively-chosen double-annotation set would
    make the agreement statistic a description of whichever items someone thought were
    worth checking.

    Args:
        case_id: The case.
        rate: The target share, in [0, 1].
        salt: Domain separator, so this decision is independent of any other hash-based
            assignment over the same ids.

    Returns:
        True when the case is in the double-annotated set.

    Raises:
        ValueError: If ``rate`` is outside [0, 1].
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"double-annotation rate {rate} is outside [0, 1]")
    digest = hashlib.sha256(f"{salt}:{case_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < rate


def cohens_kappa(a: list[str], b: list[str]) -> float:
    """Compute Cohen's kappa between two raters' label sequences.

    Args:
        a: The first rater's labels.
        b: The second rater's, index-aligned.

    Returns:
        Kappa. **1.0 when both raters agree on every item and use only one label**, which
        is the degenerate case where chance agreement is also 1.0 and the usual formula is
        0/0. Returning 1.0 there is the defensible reading — they did agree on everything
        — and it is stated rather than left to a NaN propagating into a report.

    Raises:
        ValueError: If the sequences differ in length or are empty.
    """
    if len(a) != len(b):
        raise ValueError(f"rater sequences differ in length: {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("cannot compute kappa over no items")

    n = len(a)
    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    counts_a, counts_b = Counter(a), Counter(b)
    expected = sum(
        (counts_a[label] / n) * (counts_b[label] / n) for label in set(counts_a) | set(counts_b)
    )
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def krippendorff_alpha_nominal(units: list[list[str]]) -> float:
    """Compute Krippendorff's alpha for nominal data over a ragged design.

    Args:
        units: One list of labels per unit, holding however many raters coded it. Units
            coded by fewer than two raters contribute nothing and are skipped, which is
            what makes alpha usable on a design where most items have one annotator.

    Returns:
        Alpha. 1.0 when every multiply-coded unit is unanimous *and* there is no
        disagreement to expect; 0.0 when no unit was coded twice.

    The standard nominal-difference form, with ``n`` the count of pairable values::

        Do = (1/n) * sum_u (1/(m_u - 1)) * sum_{i!=j in u} [v_i != v_j]
        De = (1/(n(n-1))) * (n^2 - sum_c n_c^2)
        alpha = 1 - Do/De

    The ``1/n`` cancels between the two, so the implementation below accumulates
    ``n·Do`` and ``n·De`` and divides once.
    """
    coded = [u for u in units if len(u) >= _MIN_RATERS_FOR_ALPHA]
    if not coded:
        return 0.0

    n_disagreements = sum(
        (x != y) / (len(unit) - 1) for unit in coded for x, y in itertools.permutations(unit, 2)
    )

    values = Counter(v for unit in coded for v in unit)
    n = sum(values.values())
    if n <= 1:
        return 1.0
    n_expected = sum(count * (n - count) for count in values.values()) / (n - 1)

    if n_expected == 0:
        return 1.0
    return 1.0 - n_disagreements / n_expected


def jaccard(a: set[str] | frozenset[str], b: set[str] | frozenset[str]) -> float:
    """Compute the Jaccard index of two sets.

    Args:
        a: The first set.
        b: The second.

    Returns:
        Intersection over union. **1.0 when both are empty**: two annotators who each
        mentioned none of a case's salient fields — because it has none — agreed
        completely, and scoring that 0.0 would penalise a case for its own structure.
    """
    if not a and not b:
        return 1.0
    union = set(a) | set(b)
    return len(set(a) & set(b)) / len(union) if union else 1.0


def token_f1(a: str, b: str) -> float:
    """Compute a content-word F1 between two narratives.

    A deliberately plain surface measure. Something more sophisticated — BERTScore, an
    embedding cosine — would report a higher number for the same two texts and invite the
    reading that the annotators agreed more than they did. The point of this figure is to
    be low.

    Args:
        a: The first narrative.
        b: The second.

    Returns:
        F1 over content-word multisets, 0.0 when either side has no content words.
    """
    tokens_a = Counter(t for t in _WORD_RE.findall(a.lower()) if t not in _STOPWORDS)
    tokens_b = Counter(t for t in _WORD_RE.findall(b.lower()) if t not in _STOPWORDS)
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = sum((tokens_a & tokens_b).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(tokens_a.values())
    recall = overlap / sum(tokens_b.values())
    return 2 * precision * recall / (precision + recall)


@dataclass(frozen=True)
class PairAgreement:
    """Agreement between two annotators on one case.

    Attributes:
        case_id: The case.
        annotators: The two pseudonyms, sorted.
        typology_agreed: Whether both assigned the same typology.
        typologies: What each assigned, in ``annotators`` order.
        content_jaccard: Jaccard over the salient fields each mentioned.
        mentioned: What each mentioned, in ``annotators`` order.
        text_f1: Content-word F1 between the two narratives.
        seconds_spent: Each annotator's time on this item.
    """

    case_id: str
    annotators: tuple[str, str]
    typology_agreed: bool
    typologies: tuple[str, str]
    content_jaccard: float
    mentioned: tuple[tuple[str, ...], tuple[str, ...]]
    text_f1: float
    seconds_spent: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised pair.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "case_id": self.case_id,
            "annotators": list(self.annotators),
            "typology_agreed": self.typology_agreed,
            "typologies": list(self.typologies),
            "content_jaccard": round(self.content_jaccard, 6),
            "mentioned": [list(self.mentioned[0]), list(self.mentioned[1])],
            "text_f1": round(self.text_f1, 6),
            "seconds_spent": [round(s, 1) for s in self.seconds_spent],
        }


@dataclass(frozen=True)
class AgreementReport:
    """The aggregate over every double-annotated item.

    Attributes:
        pairs: Per-item agreement, sorted by case id.
        kappa: Cohen's kappa on typology over the doubled subset.
        alpha: Krippendorff's alpha on typology, pooled over the whole set.
        mean_content_jaccard: Mean salient-field overlap.
        mean_text_f1: Mean content-word F1. Expected low.
        n_double_annotated: How many cases had two annotators.
        n_expected_double: How many the assignment rule selected.
        n_single_annotated: Cases with exactly one annotator.
        per_annotator: Per-annotator counts and mean time.
        typology_confusion: ``(a, b) -> count`` over disagreeing pairs, sorted. The most
            actionable output in the report: a systematic confusion between two typologies
            is a guidelines problem, and it is invisible in a single kappa.
    """

    pairs: tuple[PairAgreement, ...]
    kappa: float
    alpha: float
    mean_content_jaccard: float
    mean_text_f1: float
    n_double_annotated: int
    n_expected_double: int
    n_single_annotated: int
    per_annotator: dict[str, dict[str, float]]
    typology_confusion: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Return the machine-readable report.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "kappa_typology": round(self.kappa, 6),
            "krippendorff_alpha_typology": round(self.alpha, 6),
            "mean_content_jaccard": round(self.mean_content_jaccard, 6),
            "mean_text_f1": round(self.mean_text_f1, 6),
            "n_double_annotated": self.n_double_annotated,
            "n_expected_double": self.n_expected_double,
            "n_single_annotated": self.n_single_annotated,
            "per_annotator": self.per_annotator,
            "typology_confusion": self.typology_confusion,
            "pairs": [p.to_dict() for p in self.pairs],
        }

    def summary(self) -> str:
        """Return the human-readable summary.

        Returns:
            A short report suitable for a terminal or the phase log.
        """
        lines = [
            f"Inter-annotator agreement over {self.n_double_annotated} double-annotated "
            f"items ({self.n_expected_double} expected, "
            f"{self.n_single_annotated} single-annotated)",
            "",
            f"  Cohen's kappa (typology)      {self.kappa:>7.3f}  {_kappa_reading(self.kappa)}",
            f"  Krippendorff alpha (typology) {self.alpha:>7.3f}",
            f"  Content selection (Jaccard)   {self.mean_content_jaccard:>7.3f}",
            f"  Text similarity (token F1)    {self.mean_text_f1:>7.3f}  "
            "expected low; narrative writing has legitimate variance",
        ]
        if self.typology_confusion:
            lines += ["", "  Typology disagreements:"]
            lines += [
                f"    {pair}: {n}"
                for pair, n in sorted(self.typology_confusion.items(), key=lambda kv: -kv[1])
            ]
        if self.per_annotator:
            lines += ["", f"  {'annotator':<16} {'items':>6} {'mean min':>9}"]
            for name, stats in sorted(self.per_annotator.items()):
                lines.append(
                    f"  {name:<16} {int(stats['n_items']):>6} " f"{stats['mean_minutes']:>9.1f}"
                )
        return "\n".join(lines)


def _kappa_reading(kappa: float) -> str:
    """Return the conventional verbal reading of a kappa.

    Landis and Koch's bands, quoted because a reader expects them, and hedged because they
    are a convention rather than a result.

    Args:
        kappa: The statistic.

    Returns:
        The band name.
    """
    for threshold, name in (
        (0.81, "almost perfect"),
        (0.61, "substantial"),
        (0.41, "moderate"),
        (0.21, "fair"),
        (0.0, "slight"),
    ):
        if kappa >= threshold:
            return f"({name}, Landis & Koch)"
    return "(poor, Landis & Koch)"


def measure_agreement(
    annotations: list[Annotation],
    *,
    double_annotation_rate: float = 0.15,
    mentioned_by: dict[tuple[str, str], tuple[str, ...]] | None = None,
) -> AgreementReport:
    """Measure agreement across a set of annotations.

    Args:
        annotations: Every non-calibration annotation, from
            :meth:`~g2t_aml.human.store.AnnotationStore.read_all`.
        double_annotation_rate: The rate the assignment was made at, used only to report
            how many items *should* have been doubled against how many were.
        mentioned_by: ``(case_id, annotator_id) -> salient fields mentioned``, from
            ingestion's slot alignment. When omitted, content overlap is not measured and
            is reported as 1.0 over an empty comparison rather than guessed at from the
            text — a guess here would be a second, weaker salience metric competing with
            the real one.

    Returns:
        The report.

    Raises:
        ValueError: If any case carries two annotations from the same annotator, which
            would pair someone with themselves and inflate every statistic.
    """
    by_case: dict[str, list[Annotation]] = defaultdict(list)
    for annotation in annotations:
        by_case[annotation.case_id].append(annotation)

    for case_id, group in by_case.items():
        names = [a.annotator_id for a in group]
        if len(set(names)) != len(names):
            raise ValueError(
                f"case {case_id!r} has two annotations from one annotator; agreement "
                "would be measured against themselves"
            )

    pairs: list[PairAgreement] = []
    confusion: Counter[str] = Counter()
    for case_id in sorted(by_case):
        group = sorted(by_case[case_id], key=lambda a: a.annotator_id)
        if len(group) < _MIN_RATERS_FOR_ALPHA:
            continue
        first, second = group[0], group[1]
        mentioned = (
            (mentioned_by or {}).get((case_id, first.annotator_id), ()),
            (mentioned_by or {}).get((case_id, second.annotator_id), ()),
        )
        agreed = first.typology_assigned == second.typology_assigned
        if not agreed:
            confusion["/".join(sorted((first.typology_assigned, second.typology_assigned)))] += 1
        pairs.append(
            PairAgreement(
                case_id=case_id,
                annotators=(first.annotator_id, second.annotator_id),
                typology_agreed=agreed,
                typologies=(first.typology_assigned, second.typology_assigned),
                content_jaccard=jaccard(set(mentioned[0]), set(mentioned[1])),
                mentioned=mentioned,
                text_f1=token_f1(first.narrative, second.narrative),
                seconds_spent=(first.seconds_spent, second.seconds_spent),
            )
        )

    kappa = (
        cohens_kappa([p.typologies[0] for p in pairs], [p.typologies[1] for p in pairs])
        if pairs
        else 0.0
    )
    alpha = krippendorff_alpha_nominal(
        [[a.typology_assigned for a in group] for group in by_case.values()]
    )

    per_annotator: dict[str, dict[str, float]] = {}
    for annotation in annotations:
        stats = per_annotator.setdefault(
            annotation.annotator_id, {"n_items": 0.0, "total_seconds": 0.0}
        )
        stats["n_items"] += 1
        stats["total_seconds"] += annotation.seconds_spent
    for stats in per_annotator.values():
        stats["mean_minutes"] = round(stats["total_seconds"] / stats["n_items"] / 60, 3)

    return AgreementReport(
        pairs=tuple(pairs),
        kappa=kappa,
        alpha=alpha,
        mean_content_jaccard=(sum(p.content_jaccard for p in pairs) / len(pairs) if pairs else 0.0),
        mean_text_f1=sum(p.text_f1 for p in pairs) / len(pairs) if pairs else 0.0,
        n_double_annotated=len(pairs),
        n_expected_double=sum(
            1 for case_id in by_case if is_double_annotated(case_id, double_annotation_rate)
        ),
        n_single_annotated=sum(1 for group in by_case.values() if len(group) == 1),
        per_annotator=per_annotator,
        typology_confusion=dict(confusion),
    )
