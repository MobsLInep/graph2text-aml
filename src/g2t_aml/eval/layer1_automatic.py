"""Layer 1: surface-overlap metrics. Necessary, not sufficient, and reported that way.

These are the numbers a reviewer expects to see and the numbers that decide nothing. They
are computed against **Gold references only** — a BLEU score against Bronze measures how
closely a system reproduces a template, which is not a quality anyone wants maximised.

**The Bronze comparison is the point of this module, not an aside.** Bronze is a
deterministic template with no model in it at all. If it scores competitively against Gold
on ROUGE, then ROUGE does not distinguish a system that understands a case from one that
fills in blanks, and every overlap number in the AML-narrative literature is measuring
something other than what it claims to. :func:`template_baseline_finding` computes exactly
that comparison and flags it, and the flag is an output of the harness rather than a note
someone remembers to write. See D-076.

**Every metric here is individually optional and records why it is absent.** BERTScore
needs a 1.4 GB model, METEOR needs a WordNet download, a learned metric needs weights that
are not in this repository's dependency set at all. A harness that raised on a missing
model would make the whole of Layer 1 unrunnable in CI; a harness that silently returned
zero would put a zero in a results table. Both are worse than
:attr:`Layer1Metrics.unavailable`, which names the metric and the reason, and which
:mod:`g2t_aml.eval.report` prints as ``—`` rather than as a number.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from g2t_aml.corpus.diversity import distinct_n, self_bleu, tokenize

__all__ = [
    "SELF_BLEU_REFERENCES",
    "Layer1Metrics",
    "LearnedMetric",
    "PerplexityModel",
    "TemplateBaselineFinding",
    "bertscore_f1",
    "bleu",
    "compute_layer1",
    "length_stats",
    "meteor",
    "rouge",
    "template_baseline_finding",
]

#: The reference count self-BLEU is reported at, fixed by D-043. Self-BLEU without its
#: reference count is not a number: on the Bronze corpus it reads 0.16 at one reference
#: and 0.82 at fifty. Reported here at five, with the curve published beside it.
SELF_BLEU_REFERENCES = 5

#: How far below the best system Bronze may score before the overlap metrics are declared
#: non-discriminative. Two ROUGE points. Set here, before any system has been scored,
#: because a threshold chosen after seeing the results is not a threshold.
_TEMPLATE_COMPETITIVE_MARGIN = 0.02

#: Minimum hypotheses before a diversity number means anything. Distinct-n over three
#: narratives is a property of the sample, not of the system.
_MIN_DIVERSITY_CORPUS = 10


class LearnedMetric(Protocol):
    """A BLEURT- or COMET-class learned metric, supplied by the caller.

    Neither is in this repository's dependency set: BLEURT is TensorFlow-only and COMET
    pulls a second copy of PyTorch Lightning, and Phases 1—6 and 10 are declared CPU-only
    and dependency-light. The protocol is the seam — a caller with either installed
    passes an adapter, and the harness records the metric's name so the paper can say
    which one produced the number.
    """

    name: str

    def score(self, hypotheses: Sequence[str], references: Sequence[str]) -> list[float]:
        """Score each hypothesis against its reference.

        Args:
            hypotheses: The generated narratives.
            references: The Gold narratives, index-aligned with ``hypotheses``.

        Returns:
            One score per pair, higher is better.
        """
        ...


class PerplexityModel(Protocol):
    """A held-out language model that can score text.

    Supplied by the caller for the same reason as :class:`LearnedMetric`: perplexity is
    only interpretable against a stated model, and hard-coding one here would bake a
    1 GB download into a CPU-only phase.
    """

    name: str

    def perplexity(self, texts: Sequence[str]) -> list[float]:
        """Return the perplexity of each text.

        Args:
            texts: The narratives to score.

        Returns:
            One perplexity per text.
        """
        ...


@dataclass(frozen=True)
class Layer1Metrics:
    """Surface-overlap and diversity metrics for one system.

    Every metric is optional. ``None`` means *not computed*, and
    :attr:`unavailable` says why — never that the metric was computed and came out at
    zero.

    Attributes:
        system: The arm these describe.
        n_pairs: How many (hypothesis, Gold reference) pairs the overlap metrics ran on.
            Distinct from the number of narratives scored: Layer 1 can only run where a
            Gold reference exists, and Layer 2 runs everywhere.
        n_narratives: How many narratives the diversity and length metrics ran on.
        bleu: Corpus BLEU-4.
        bleu_signature: sacrebleu's signature. Reported because a BLEU number without
            its tokenisation, smoothing and case handling is not comparable to any
            other BLEU number, including a later one from this same harness.
        rouge1: ROUGE-1 F-measure, averaged over pairs.
        rouge2: ROUGE-2 F-measure.
        rouge_l: ROUGE-L F-measure.
        meteor: METEOR, averaged over pairs.
        bertscore_f1: BERTScore F1, **rescaled with baseline**.
        bertscore_model: The model that produced it.
        learned_metric: A BLEURT/COMET-class score, when one was supplied.
        learned_metric_name: Which metric produced it.
        distinct_1: Distinct unigram ratio over the system's own narratives.
        distinct_2: Distinct bigram ratio.
        self_bleu: Self-BLEU at :data:`SELF_BLEU_REFERENCES` references.
        self_bleu_references: The reference count, carried with the number (D-043).
        perplexity: Mean perplexity under the held-out LM, when one was supplied.
        perplexity_model: Which LM produced it.
        length_words_mean: Mean narrative length in words.
        length_words_std: Standard deviation of the same.
        reference_length_words_mean: Mean Gold length, for the length-distribution
            comparison.
        length_ratio: System mean length over Gold mean length. 1.0 is a match; a system
            at 0.6 is dropping content that ROUGE will not fully punish it for.
        unavailable: Metric name to the reason it was not computed.
    """

    system: str
    n_pairs: int = 0
    n_narratives: int = 0
    bleu: float | None = None
    bleu_signature: str | None = None
    rouge1: float | None = None
    rouge2: float | None = None
    rouge_l: float | None = None
    meteor: float | None = None
    bertscore_f1: float | None = None
    bertscore_model: str | None = None
    learned_metric: float | None = None
    learned_metric_name: str | None = None
    distinct_1: float | None = None
    distinct_2: float | None = None
    self_bleu: float | None = None
    self_bleu_references: int = SELF_BLEU_REFERENCES
    perplexity: float | None = None
    perplexity_model: str | None = None
    length_words_mean: float | None = None
    length_words_std: float | None = None
    reference_length_words_mean: float | None = None
    length_ratio: float | None = None
    unavailable: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the metrics as a JSON-serialisable mapping.

        Returns:
            Every field, with ``None`` preserved rather than coerced to zero.
        """
        return {
            "system": self.system,
            "n_pairs": self.n_pairs,
            "n_narratives": self.n_narratives,
            "bleu": self.bleu,
            "bleu_signature": self.bleu_signature,
            "rouge1": self.rouge1,
            "rouge2": self.rouge2,
            "rouge_l": self.rouge_l,
            "meteor": self.meteor,
            "bertscore_f1": self.bertscore_f1,
            "bertscore_model": self.bertscore_model,
            "learned_metric": self.learned_metric,
            "learned_metric_name": self.learned_metric_name,
            "distinct_1": self.distinct_1,
            "distinct_2": self.distinct_2,
            "self_bleu": self.self_bleu,
            "self_bleu_references": self.self_bleu_references,
            "perplexity": self.perplexity,
            "perplexity_model": self.perplexity_model,
            "length_words_mean": self.length_words_mean,
            "length_words_std": self.length_words_std,
            "reference_length_words_mean": self.reference_length_words_mean,
            "length_ratio": self.length_ratio,
            "unavailable": dict(sorted(self.unavailable.items())),
        }


# ------------------------------------------------------------------ metrics ---


def bleu(hypotheses: Sequence[str], references: Sequence[str]) -> tuple[float, str]:
    """Compute corpus BLEU-4 and return it with sacrebleu's signature.

    The signature comes back as a second return value rather than being logged, because a
    BLEU score whose tokenisation is not written down beside it cannot be compared to
    anything — including the same corpus scored by this function a year later under a
    different sacrebleu default.

    Args:
        hypotheses: The generated narratives.
        references: The Gold narratives, index-aligned.

    Returns:
        ``(score, signature)``. The score is on sacrebleu's 0—100 scale.

    Raises:
        ImportError: If sacrebleu is not installed. Caught by :func:`compute_layer1`,
            which records the absence rather than failing the run.
        ValueError: If the two sequences are different lengths.
    """
    if len(hypotheses) != len(references):
        raise ValueError(
            f"BLEU needs one reference per hypothesis; got {len(hypotheses)} and "
            f"{len(references)}"
        )
    import sacrebleu

    metric = sacrebleu.BLEU()
    result = metric.corpus_score(list(hypotheses), [list(references)])
    return float(result.score), str(metric.get_signature())


def rouge(hypotheses: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    """Compute ROUGE-1, ROUGE-2 and ROUGE-L F-measures, averaged over pairs.

    Args:
        hypotheses: The generated narratives.
        references: The Gold narratives, index-aligned.

    Returns:
        ``{"rouge1": …, "rouge2": …, "rougeL": …}``, each an F-measure in [0, 1].

    Raises:
        ImportError: If rouge-score is not installed.
        ValueError: If the two sequences are different lengths.
    """
    if len(hypotheses) != len(references):
        raise ValueError(
            f"ROUGE needs one reference per hypothesis; got {len(hypotheses)} and "
            f"{len(references)}"
        )
    from rouge_score import rouge_scorer

    keys = ("rouge1", "rouge2", "rougeL")
    scorer = rouge_scorer.RougeScorer(list(keys), use_stemmer=True)
    totals = dict.fromkeys(keys, 0.0)
    for hypothesis, reference in zip(hypotheses, references, strict=True):
        scores = scorer.score(reference, hypothesis)
        for key in keys:
            totals[key] += float(scores[key].fmeasure)
    n = max(len(hypotheses), 1)
    return {key: totals[key] / n for key in keys}


def meteor(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    """Compute METEOR, averaged over pairs.

    Args:
        hypotheses: The generated narratives.
        references: The Gold narratives, index-aligned.

    Returns:
        The mean METEOR score in [0, 1].

    Raises:
        ImportError: If nltk is not installed.
        LookupError: If the WordNet corpus has not been downloaded. Surfaced rather than
            downloaded on the caller's behalf: a metric function that reaches the network
            makes an offline CI run non-deterministic in a way that shows up as a score
            change rather than as a failure.
        ValueError: If the two sequences are different lengths.
    """
    if len(hypotheses) != len(references):
        raise ValueError(
            f"METEOR needs one reference per hypothesis; got {len(hypotheses)} and "
            f"{len(references)}"
        )
    from nltk.translate.meteor_score import meteor_score

    total = 0.0
    for hypothesis, reference in zip(hypotheses, references, strict=True):
        total += float(meteor_score([tokenize(reference)], tokenize(hypothesis)))
    return total / max(len(hypotheses), 1)


def bertscore_f1(
    hypotheses: Sequence[str],
    references: Sequence[str],
    *,
    model: str = "microsoft/deberta-xlarge-mnli",
    batch_size: int = 16,
) -> float:
    """Compute BERTScore F1, **rescaled with baseline**.

    Rescaling is not optional here and is not exposed as a parameter. Raw BERTScore
    occupies a narrow band near 0.85 whose floor is a property of the encoder rather than
    of the systems, so an unrescaled number is uninterpretable across papers and barely
    interpretable within one — two systems 0.02 apart on raw BERTScore may be 0.20 apart
    once the baseline is removed. Every number this harness reports is rescaled, and a
    parameter that let one not be would eventually be set.

    Args:
        hypotheses: The generated narratives.
        references: The Gold narratives, index-aligned.
        model: The encoder. Recorded on the result: BERTScore is only comparable within
            one model.
        batch_size: Encoder batch size.

    Returns:
        The mean rescaled F1.

    Raises:
        ImportError: If bert-score is not installed.
        ValueError: If the two sequences are different lengths.
    """
    if len(hypotheses) != len(references):
        raise ValueError(
            f"BERTScore needs one reference per hypothesis; got {len(hypotheses)} and "
            f"{len(references)}"
        )
    from bert_score import score as bert_score_fn

    _, _, f1 = bert_score_fn(
        list(hypotheses),
        list(references),
        model_type=model,
        rescale_with_baseline=True,
        batch_size=batch_size,
        verbose=False,
    )
    return float(f1.mean().item())


def length_stats(texts: Sequence[str]) -> dict[str, float]:
    """Return the word-length distribution of a set of narratives.

    Args:
        texts: The narratives.

    Returns:
        ``{"mean": …, "std": …, "min": …, "max": …, "median": …}``, all zero over an
        empty input rather than undefined.
    """
    lengths = [float(len(tokenize(text))) for text in texts]
    if not lengths:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    return {
        "mean": statistics.fmean(lengths),
        "std": statistics.pstdev(lengths) if len(lengths) > 1 else 0.0,
        "min": min(lengths),
        "max": max(lengths),
        "median": statistics.median(lengths),
    }


def compute_layer1(  # noqa: PLR0912, PLR0915 -- one guarded block per optional metric;
    # splitting them separates each metric from the reason its absence is recorded.
    system: str,
    hypotheses: Sequence[str],
    references: Sequence[str],
    *,
    all_narratives: Sequence[str] | None = None,
    bertscore_model: str | None = "microsoft/deberta-xlarge-mnli",
    learned: LearnedMetric | None = None,
    perplexity_model: PerplexityModel | None = None,
    seed: int = 42,
) -> Layer1Metrics:
    """Compute every Layer 1 metric that can be computed, and record why the rest cannot.

    Args:
        system: The arm being scored.
        hypotheses: Generated narratives that have a Gold reference.
        references: The Gold narratives, index-aligned with ``hypotheses``.
        all_narratives: Every narrative the system produced, including those with no Gold
            reference. Diversity and length are measured over this, because they need no
            reference and the Gold subset is far too small to characterise a system's
            output distribution. Defaults to ``hypotheses``.
        bertscore_model: Encoder for BERTScore, or None to skip it.
        learned: A BLEURT/COMET-class metric, or None.
        perplexity_model: A held-out LM, or None.
        seed: Seed for self-BLEU's reference sampling.

    Returns:
        The metrics, with :attr:`Layer1Metrics.unavailable` naming everything absent.

    Raises:
        ValueError: If ``hypotheses`` and ``references`` are different lengths. This is a
            caller bug rather than a missing dependency and is not softened into an
            ``unavailable`` entry.
    """
    if len(hypotheses) != len(references):
        raise ValueError(
            f"Layer 1 needs one reference per hypothesis; got {len(hypotheses)} and "
            f"{len(references)}"
        )
    corpus = list(all_narratives) if all_narratives is not None else list(hypotheses)
    unavailable: dict[str, str] = {}
    values: dict[str, Any] = {}

    if not hypotheses:
        unavailable.update(
            dict.fromkeys(
                ("bleu", "rouge", "meteor", "bertscore", "learned_metric"),
                "no case has a Gold reference; Layer 1 needs one and Gold is not written",
            )
        )
    else:
        try:
            values["bleu"], values["bleu_signature"] = bleu(hypotheses, references)
        except ImportError as exc:
            unavailable["bleu"] = f"sacrebleu is not installed ({exc})"

        try:
            scores = rouge(hypotheses, references)
            values["rouge1"] = scores["rouge1"]
            values["rouge2"] = scores["rouge2"]
            values["rouge_l"] = scores["rougeL"]
        except ImportError as exc:
            unavailable["rouge"] = f"rouge-score is not installed ({exc})"

        try:
            values["meteor"] = meteor(hypotheses, references)
        except ImportError as exc:
            unavailable["meteor"] = f"nltk is not installed ({exc})"
        except LookupError as exc:
            unavailable["meteor"] = f"the WordNet corpus is not downloaded ({exc})"

        if bertscore_model is None:
            unavailable["bertscore"] = "disabled by the caller"
        else:
            try:
                values["bertscore_f1"] = bertscore_f1(hypotheses, references, model=bertscore_model)
                values["bertscore_model"] = bertscore_model
            except ImportError as exc:
                unavailable["bertscore"] = f"bert-score is not installed ({exc})"
            except (OSError, RuntimeError) as exc:
                unavailable["bertscore"] = f"the {bertscore_model} weights are unavailable ({exc})"

        if learned is None:
            unavailable["learned_metric"] = (
                "no BLEURT/COMET-class metric was supplied; neither is in this "
                "repository's dependency set (see LearnedMetric)"
            )
        else:
            per_pair = learned.score(hypotheses, references)
            values["learned_metric"] = statistics.fmean(per_pair) if per_pair else None
            values["learned_metric_name"] = learned.name

    if len(corpus) < _MIN_DIVERSITY_CORPUS:
        unavailable["diversity"] = (
            f"{len(corpus)} narratives is below the {_MIN_DIVERSITY_CORPUS} needed for "
            "distinct-n and self-BLEU to describe the system rather than the sample"
        )
    else:
        values["distinct_1"] = distinct_n(corpus, 1)
        values["distinct_2"] = distinct_n(corpus, 2)
        values["self_bleu"] = self_bleu(corpus, n_references=SELF_BLEU_REFERENCES, seed=seed)

    if perplexity_model is None:
        unavailable["perplexity"] = "no held-out language model was supplied"
    else:
        scored = perplexity_model.perplexity(corpus)
        values["perplexity"] = statistics.fmean(scored) if scored else None
        values["perplexity_model"] = perplexity_model.name

    own = length_stats(corpus)
    values["length_words_mean"] = own["mean"]
    values["length_words_std"] = own["std"]
    if references:
        reference_lengths = length_stats(references)
        values["reference_length_words_mean"] = reference_lengths["mean"]
        if reference_lengths["mean"] > 0:
            values["length_ratio"] = own["mean"] / reference_lengths["mean"]

    # A metric library can hand back a NaN over a degenerate input -- an empty
    # hypothesis, a reference of one token -- and `NaN` is not valid JSON. Dropping it to
    # None here means the report prints an em dash instead of writing a file no parser
    # will read back.
    values = {k: _finite(v) if isinstance(v, float) else v for k, v in values.items()}

    return Layer1Metrics(
        system=system,
        n_pairs=len(hypotheses),
        n_narratives=len(corpus),
        unavailable=unavailable,
        **values,
    )


# ------------------------------------------------------- the Bronze finding ---


@dataclass(frozen=True)
class TemplateBaselineFinding:
    """Whether a deterministic template scores competitively on overlap metrics.

    Attributes:
        metric: The metric compared on.
        template_system: The template arm — Bronze.
        template_score: Its score against Gold.
        best_model_system: The highest-scoring non-template system.
        best_model_score: That system's score.
        margin: ``best_model_score - template_score``. Negative means the template won.
        threshold: The margin below which the metric is declared non-discriminative,
            fixed at :data:`_TEMPLATE_COMPETITIVE_MARGIN` before any system was scored.
        non_discriminative: True when the margin is below the threshold — the metric does
            not separate a template from a model, and any ranking it produces is not
            evidence about the systems.
    """

    metric: str
    template_system: str
    template_score: float
    best_model_system: str
    best_model_score: float
    margin: float
    threshold: float
    non_discriminative: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the finding as a JSON-serialisable mapping.

        Returns:
            Every field.
        """
        return {
            "metric": self.metric,
            "template_system": self.template_system,
            "template_score": self.template_score,
            "best_model_system": self.best_model_system,
            "best_model_score": self.best_model_score,
            "margin": self.margin,
            "threshold": self.threshold,
            "non_discriminative": self.non_discriminative,
        }

    @property
    def headline(self) -> str:
        """Return the finding as one sentence, for the report and the paper.

        Returns:
            A sentence stating what the comparison showed.
        """
        if not self.non_discriminative:
            return (
                f"{self.best_model_system} beats the {self.template_system} template by "
                f"{self.margin:.4f} {self.metric}, above the {self.threshold:.2f} margin: "
                f"{self.metric} distinguishes the two."
            )
        verb = "outscores" if self.margin < 0 else "is within"
        amount = f"{abs(self.margin):.4f}"
        return (
            f"The {self.template_system} template {verb} {amount} {self.metric} of "
            f"{self.best_model_system}, the best model arm. {self.metric} does not "
            "distinguish a deterministic template from a trained system, and a ranking "
            "produced by it is not evidence about the systems."
        )


def template_baseline_finding(
    metrics: Sequence[Layer1Metrics],
    *,
    metric: str = "rouge_l",
    template_system: str = "bronze",
    threshold: float = _TEMPLATE_COMPETITIVE_MARGIN,
) -> TemplateBaselineFinding | None:
    """Compare the deterministic template against the best model arm on one metric.

    The output the module docstring describes: direct evidence about whether overlap
    metrics separate real systems from templates. Run over ROUGE-L by default because
    that is the metric the SAR-generation literature reports most often, so a null result
    on it is the one that lands.

    Args:
        metrics: Layer 1 metrics, one per system, including the template arm.
        metric: The :class:`Layer1Metrics` field to compare. Must be a float field.
        template_system: The template arm's name.
        threshold: How far below the best model the template may score before the metric
            is declared non-discriminative.

    Returns:
        The finding, or None when the template arm is absent, no model arm is present, or
        either side lacks the metric.
    """
    by_name = {m.system: m for m in metrics}
    template = by_name.get(template_system)
    if template is None:
        return None
    template_score = getattr(template, metric, None)
    if not isinstance(template_score, int | float):
        return None

    candidates = [
        (m.system, float(getattr(m, metric)))
        for m in metrics
        if m.system != template_system and isinstance(getattr(m, metric, None), int | float)
    ]
    if not candidates:
        return None
    best_system, best_score = max(candidates, key=lambda pair: pair[1])
    margin = best_score - float(template_score)
    return TemplateBaselineFinding(
        metric=metric,
        template_system=template_system,
        template_score=float(template_score),
        best_model_system=best_system,
        best_model_score=best_score,
        margin=margin,
        threshold=threshold,
        non_discriminative=margin < threshold,
    )


def _finite(value: float | None) -> float | None:
    """Return a float only when it is finite.

    Args:
        value: The candidate.

    Returns:
        The value, or None when it is None, NaN or infinite. Used at the boundary where
        a metric library can hand back a NaN over a degenerate input and a JSON report
        would then carry ``NaN``, which is not valid JSON.
    """
    if value is None or not math.isfinite(value):
        return None
    return float(value)
