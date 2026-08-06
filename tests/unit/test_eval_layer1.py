"""Layer 1 metrics, and the template-baseline finding they exist to produce.

The overlap metrics themselves are thin wrappers over sacrebleu and rouge-score, so what
is worth testing is the two things this module adds: that a missing dependency or model
produces a *named absence* rather than a zero, and that the Bronze comparison computes and
flags the finding the paper quotes.
"""

from __future__ import annotations

import pytest

from g2t_aml.eval.layer1_automatic import (
    SELF_BLEU_REFERENCES,
    Layer1Metrics,
    compute_layer1,
    length_stats,
    template_baseline_finding,
)


class _ConstantLearnedMetric:
    """A stand-in for BLEURT/COMET, which are deliberately not dependencies."""

    name = "stub-learned-metric"

    def score(self, hypotheses, references):
        return [0.75] * len(hypotheses)


class _ConstantPerplexity:
    name = "stub-lm"

    def perplexity(self, texts):
        return [12.5] * len(texts)


def corpus(n: int = 12) -> list[str]:
    return [
        f"Narrative number {i} describes account {i} and its {i} counterparties." for i in range(n)
    ]


# ------------------------------------------------------- overlap metrics ---


def test_identical_hypotheses_and_references_score_at_the_ceiling():
    pytest.importorskip("sacrebleu")
    pytest.importorskip("rouge_score")
    texts = corpus(12)

    metrics = compute_layer1("perfect", texts, texts, bertscore_model=None)

    assert metrics.bleu == pytest.approx(100.0)
    assert metrics.rouge1 == pytest.approx(1.0)
    assert metrics.rouge2 == pytest.approx(1.0)
    assert metrics.rouge_l == pytest.approx(1.0)
    assert metrics.length_ratio == pytest.approx(1.0)


def test_the_bleu_signature_is_reported_with_the_score():
    # A BLEU number whose tokenisation is not written down beside it is not comparable to
    # any other BLEU number, including a later one from this same harness.
    pytest.importorskip("sacrebleu")
    metrics = compute_layer1("x", corpus(12), corpus(12), bertscore_model=None)
    assert metrics.bleu_signature
    assert "tok" in metrics.bleu_signature


def test_unrelated_text_scores_far_below_the_ceiling():
    pytest.importorskip("rouge_score")
    hypotheses = ["Completely different wording about unrelated subjects."] * 12
    metrics = compute_layer1("x", hypotheses, corpus(12), bertscore_model=None)
    assert metrics.rouge1 is not None
    assert metrics.rouge1 < 0.3


def test_length_stats_are_hand_computable():
    stats = length_stats(["one two three", "one two three four five"])
    assert stats["mean"] == 4.0
    assert stats["min"] == 3.0
    assert stats["max"] == 5.0
    assert stats["median"] == 4.0


def test_length_stats_over_nothing_are_zero_rather_than_undefined():
    assert length_stats([])["mean"] == 0.0


def test_a_mismatched_reference_count_is_a_caller_bug_not_a_missing_metric():
    with pytest.raises(ValueError, match="one reference per hypothesis"):
        compute_layer1("x", ["a", "b"], ["a"])


# ---------------------------------------------------- absence, not zero ---


def test_a_disabled_metric_is_named_absent_rather_than_scored_zero():
    metrics = compute_layer1("x", corpus(12), corpus(12), bertscore_model=None)

    assert metrics.bertscore_f1 is None
    assert "bertscore" in metrics.unavailable
    assert metrics.unavailable["bertscore"] == "disabled by the caller"


def test_no_gold_reference_disables_every_overlap_metric_and_says_why():
    # The state the project is actually in: Gold is not written. Layer 1 must report that
    # rather than crash or emit zeros, and Layer 2 must be unaffected.
    metrics = compute_layer1("x", [], [], all_narratives=corpus(12), bertscore_model=None)

    assert metrics.bleu is None
    assert metrics.rouge_l is None
    assert all(
        "Gold" in reason for key, reason in metrics.unavailable.items() if key != "perplexity"
    )
    # Diversity needs no reference, so it still runs.
    assert metrics.distinct_1 is not None
    assert metrics.n_pairs == 0
    assert metrics.n_narratives == 12


def test_diversity_is_withheld_on_a_corpus_too_small_to_characterise_a_system():
    metrics = compute_layer1("x", corpus(3), corpus(3), bertscore_model=None)
    assert metrics.distinct_1 is None
    assert "below the 10" in metrics.unavailable["diversity"]


def test_self_bleu_carries_its_reference_count():
    # Self-BLEU without its reference count is not a number: on the Bronze corpus it
    # reads 0.16 at one reference and 0.82 at fifty (D-043).
    metrics = compute_layer1("x", corpus(12), corpus(12), bertscore_model=None)
    assert metrics.self_bleu_references == SELF_BLEU_REFERENCES
    assert metrics.to_dict()["self_bleu_references"] == SELF_BLEU_REFERENCES


def test_a_supplied_learned_metric_is_used_and_named():
    metrics = compute_layer1(
        "x", corpus(12), corpus(12), bertscore_model=None, learned=_ConstantLearnedMetric()
    )
    assert metrics.learned_metric == pytest.approx(0.75)
    assert metrics.learned_metric_name == "stub-learned-metric"
    assert "learned_metric" not in metrics.unavailable


def test_an_absent_learned_metric_explains_why_it_is_absent():
    metrics = compute_layer1("x", corpus(12), corpus(12), bertscore_model=None)
    assert "dependency set" in metrics.unavailable["learned_metric"]


def test_a_supplied_perplexity_model_is_used_and_named():
    metrics = compute_layer1(
        "x", corpus(12), corpus(12), bertscore_model=None, perplexity_model=_ConstantPerplexity()
    )
    assert metrics.perplexity == pytest.approx(12.5)
    assert metrics.perplexity_model == "stub-lm"


def test_the_serialised_form_preserves_none_rather_than_coercing_to_zero():
    payload = compute_layer1("x", [], [], bertscore_model=None).to_dict()
    assert payload["bleu"] is None
    assert payload["unavailable"]


# ------------------------------------------------ the template baseline ---


def test_the_template_finding_flags_a_metric_that_does_not_separate_the_two():
    # The quotable result: a deterministic template within two ROUGE points of the best
    # model arm means ROUGE does not distinguish them, and any ranking it produces is not
    # evidence about the systems.
    finding = template_baseline_finding(
        [
            Layer1Metrics(system="bronze", rouge_l=0.41),
            Layer1Metrics(system="s1", rouge_l=0.42),
        ]
    )

    assert finding is not None
    assert finding.non_discriminative
    assert finding.margin == pytest.approx(0.01)
    assert "does not distinguish" in finding.headline


def test_the_template_finding_reports_a_clear_win_as_a_clear_win():
    finding = template_baseline_finding(
        [
            Layer1Metrics(system="bronze", rouge_l=0.30),
            Layer1Metrics(system="s1", rouge_l=0.55),
        ]
    )
    assert finding is not None
    assert not finding.non_discriminative
    assert "distinguishes the two" in finding.headline


def test_the_template_finding_handles_the_template_winning():
    finding = template_baseline_finding(
        [
            Layer1Metrics(system="bronze", rouge_l=0.50),
            Layer1Metrics(system="s1", rouge_l=0.45),
        ]
    )
    assert finding is not None
    assert finding.non_discriminative
    assert finding.margin < 0
    assert "outscores" in finding.headline


def test_the_template_finding_picks_the_best_model_arm():
    finding = template_baseline_finding(
        [
            Layer1Metrics(system="bronze", rouge_l=0.30),
            Layer1Metrics(system="s1", rouge_l=0.40),
            Layer1Metrics(system="b7", rouge_l=0.55),
        ]
    )
    assert finding is not None
    assert finding.best_model_system == "b7"


def test_the_template_finding_is_absent_when_there_is_nothing_to_compare():
    assert template_baseline_finding([Layer1Metrics(system="s1", rouge_l=0.4)]) is None
    assert template_baseline_finding([Layer1Metrics(system="bronze", rouge_l=0.4)]) is None
    assert (
        template_baseline_finding([Layer1Metrics(system="bronze"), Layer1Metrics(system="s1")])
        is None
    )


def test_the_template_finding_serialises_with_its_threshold():
    # The threshold is fixed before any system was scored, and it travels with the
    # finding so a reader can see what "competitive" meant.
    payload = template_baseline_finding(
        [Layer1Metrics(system="bronze", rouge_l=0.41), Layer1Metrics(system="s1", rouge_l=0.42)]
    ).to_dict()
    assert payload["threshold"] == pytest.approx(0.02)
    assert payload["non_discriminative"] is True
