"""The three-layer evaluation harness.

**Layer 1** — surface overlap against Gold references. Necessary for comparability with
prior work, sufficient for nothing. :mod:`~g2t_aml.eval.layer1_automatic` computes it and
also computes the Bronze template's scores, so the harness itself produces the evidence
about whether these metrics separate a template from a trained system.

**Layer 2** — faithfulness against the fact record. The defensible core, and the layer
that works on the fifteen thousand cases that will never have a human reference.
:mod:`~g2t_aml.eval.layer2_faithfulness` computes it from claims found by one of two
independent extractors in :mod:`~g2t_aml.eval.claim_extraction`, whose agreement is
measured rather than assumed. :mod:`~g2t_aml.eval.taxonomy_scorer` says which *kind* of
error, which is what the existing SAR literature does not report.

**Layer 3** — human decision-setting validation, in Phase 12.
:mod:`~g2t_aml.eval.metric_validation` is built now and runs the moment those ratings
land, because an analysis specified after seeing its data is an analysis chosen to fit it.

Everything above is glued together by :func:`~g2t_aml.eval.report.evaluate`, and every
number that reaches a table passes through :mod:`~g2t_aml.eval.statistics` — bootstrap
interval, paired test, Holm correction, effect size — because the shapes there are what
enforce "never a single seed" and "never a p-value without an effect size".

**This layer is a measurement instrument** and falls under invariant 1 along with
``facts/``: ``mypy --strict`` covers it, and a bug here silently corrupts every headline
number in the paper.
"""

from g2t_aml.eval.claim_extraction import (
    AgreementReport,
    DeterministicClaimExtractor,
    LLMClaimExtractor,
    cohens_kappa,
    measure_agreement,
)
from g2t_aml.eval.layer1_automatic import (
    Layer1Metrics,
    TemplateBaselineFinding,
    compute_layer1,
    template_baseline_finding,
)
from g2t_aml.eval.layer2_faithfulness import (
    CaseFaithfulness,
    FaithfulnessMetrics,
    aggregate_faithfulness,
    score_case,
    score_cases,
)
from g2t_aml.eval.metric_validation import MetricValidationReport, validate_metrics
from g2t_aml.eval.report import (
    HEADLINE_METRIC,
    PRIMARY_METRICS,
    EvaluationReport,
    SystemReport,
    evaluate,
)
from g2t_aml.eval.statistics import (
    Interval,
    PairedComparison,
    SeedSummary,
    bootstrap_ci,
    cliffs_delta,
    cohens_d,
    compare_systems,
    holm_bonferroni,
    publication_table,
    seed_summary,
    wilcoxon_signed_rank,
)
from g2t_aml.eval.taxonomy_scorer import (
    ClassifiedError,
    TaxonomyReport,
    ValidationReport,
    score_taxonomy,
    validate_against_hand_labels,
)
from g2t_aml.eval.types import (
    EvaluationInputError,
    ScoredCase,
    SystemOutput,
    load_system_outputs,
    pair_outputs_with_facts,
)

__all__ = [
    "HEADLINE_METRIC",
    "PRIMARY_METRICS",
    "AgreementReport",
    "CaseFaithfulness",
    "ClassifiedError",
    "DeterministicClaimExtractor",
    "EvaluationInputError",
    "EvaluationReport",
    "FaithfulnessMetrics",
    "Interval",
    "LLMClaimExtractor",
    "Layer1Metrics",
    "MetricValidationReport",
    "PairedComparison",
    "ScoredCase",
    "SeedSummary",
    "SystemOutput",
    "SystemReport",
    "TaxonomyReport",
    "TemplateBaselineFinding",
    "ValidationReport",
    "aggregate_faithfulness",
    "bootstrap_ci",
    "cliffs_delta",
    "cohens_d",
    "cohens_kappa",
    "compare_systems",
    "compute_layer1",
    "evaluate",
    "holm_bonferroni",
    "load_system_outputs",
    "measure_agreement",
    "pair_outputs_with_facts",
    "publication_table",
    "score_case",
    "score_cases",
    "score_taxonomy",
    "seed_summary",
    "template_baseline_finding",
    "validate_metrics",
    "validate_against_hand_labels",
    "wilcoxon_signed_rank",
]
